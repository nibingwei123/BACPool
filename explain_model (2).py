# -*- coding: utf-8 -*-
"""
分子性质预测 - Sum 池化 + 多骨架 (gine/gin/gcn/gat)
数据集: BBBP, BACE, HIV, Clintox, SIDER, Tox21, ToxCast, MUV, ESOL, FreeSolv, Lipophilicity, QM7, QM8
添加：内部特征缓存，目录 ./feature_cache
保存：每折最优模型，命名格式: 数据集_骨架_sum_batch32_best_fold1.pt
CSV记录每折性能，包含骨架列
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
import types
import math, random, csv, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, GATConv,GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool, LayerNorm
)
from sklearn.metrics import roc_auc_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import BondType, BondStereo
from collections import Counter
from itertools import product
from torch_geometric.utils import softmax, scatter

from torch_geometric.utils import softmax, scatter, to_dense_batch, to_dense_adj
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, GATConv, GATv2Conv,
    global_mean_pool, global_max_pool, global_add_pool, LayerNorm,
    SAGPooling as PyG_SAGPooling,
    TopKPooling as PyG_TopKPooling,
    ASAPooling as PyG_ASAPooling,
    EdgePooling as PyG_EdgePooling,
    DenseGINConv,DenseGCNConv,
    dense_diff_pool,
    dense_mincut_pool,
    Set2Set,            # ← 新增
    GlobalAttention,    # ← 新增
)
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.explain import Explainer, PGExplainer
# ==================== 常量与特征生成 ====================
ATOM_SYMBOLS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At", "metal"]
DEGREE_LIST = [0, 1, 2, 3, 4, 5]
HYBRID_LIST = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
HYDROGEN_LIST = [0, 1, 2, 3, 4]

import networkx as nx


def compute_atom_centrality(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms <= 1:
        return np.zeros((n_atoms, 3), dtype=np.float32)
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    deg_cent = nx.degree_centrality(G)
    clo_cent = nx.closeness_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    feats = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        feats[i, 0] = deg_cent.get(i, 0.0)
        feats[i, 1] = clo_cent.get(i, 0.0)
        feats[i, 2] = bet_cent.get(i, 0.0)
    return feats


def get_atom_features_39(mol):
    atoms = list(mol.GetAtoms())
    chiral_centers = {idx: tag for idx, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)}
    feats = []
    for atom in atoms:
        feat = []
        sym = atom.GetSymbol()
        if sym not in ATOM_SYMBOLS:
            type_vec = [0.0] * 15 + [1.0]
        else:
            type_vec = [1.0 if s == sym else 0.0 for s in ATOM_SYMBOLS[:-1]] + [0.0]
        feat.extend(type_vec)
        deg = atom.GetDegree()
        deg_vec = [1.0 if deg == d else 0.0 for d in DEGREE_LIST]
        feat.extend(deg_vec)
        feat.append(float(atom.GetFormalCharge()))
        feat.append(float(atom.GetNumRadicalElectrons()))
        hyb = str(atom.GetHybridization())
        if hyb in ("S", "UNSPECIFIED"):
            hyb = "OTHER"
        if hyb not in HYBRID_LIST:
            hyb = "OTHER"
        hyb_vec = [1.0 if hyb == h else 0.0 for h in HYBRID_LIST]
        feat.extend(hyb_vec)
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)
        h_count = atom.GetTotalNumHs()
        if h_count > 4:
            h_count = 4
        h_vec = [1.0 if h_count == c else 0.0 for c in HYDROGEN_LIST]
        feat.extend(h_vec)
        idx = atom.GetIdx()
        feat.append(1.0 if idx in chiral_centers else 0.0)
        if idx in chiral_centers:
            tag = chiral_centers[idx]
            if tag == "R":
                feat.extend([1.0, 0.0])
            elif tag == "S":
                feat.extend([0.0, 1.0])
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])
        feats.append(feat)
    return torch.tensor(feats, dtype=torch.float32)


def get_edge_features(mol):
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[u, v], [v, u]])
        bt = bond.GetBondType()
        attr = torch.tensor([
            1.0 if bt == BondType.SINGLE else 0.0,
            1.0 if bt == BondType.DOUBLE else 0.0,
            1.0 if bt == BondType.TRIPLE else 0.0,
            1.0 if bt == BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
            1.0 if bond.GetStereo() == BondStereo.STEREONONE else 0.0,
            1.0 if bond.GetStereo() == BondStereo.STEREOANY else 0.0,
            1.0 if bond.GetStereo() == BondStereo.STEREOZ else 0.0,
            1.0 if bond.GetStereo() == BondStereo.STEREOE else 0.0,
        ], dtype=torch.float)
        edge_attrs.extend([attr, attr])
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    if edge_attr.dim() != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        edge_attr = torch.zeros((edge_index.shape[1], 10), dtype=torch.float)
    return edge_index, edge_attr


# ==================== 环境标签 ====================
def generate_env_labels(smiles_list, n_clusters=5):
    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("INVALID")
        else:
            try:
                scaff = MurckoScaffold.GetScaffoldForMol(mol)
                scaff_smi = Chem.MolToSmiles(scaff) if scaff else "NO_SCAFFOLD"
            except:
                scaff_smi = "ERROR"
            scaffolds.append(scaff_smi)
    freq = Counter(scaffolds)
    top = [s for s, _ in freq.most_common(n_clusters - 1)]
    top_set = set(top)
    env_labels = []
    for s in scaffolds:
        if s in top_set:
            env_labels.append(top.index(s))
        else:
            env_labels.append(n_clusters - 1)
    return env_labels


def get_brics_frag_ids(mol):
    n_heavy = mol.GetNumAtoms()
    frag_ids = np.zeros(n_heavy, dtype=np.int64)
    try:
        broken = BRICS.BreakBRICSBonds(mol)
        if broken is None:
            return frag_ids
        raw_frags = Chem.GetMolFrags(broken, asMols=False)
    except:
        return frag_ids
    if not raw_frags or len(raw_frags) <= 1:
        return frag_ids
    clean_frags = []
    for f in raw_frags:
        heavy_idx = sorted([a for a in f if a < n_heavy])
        if heavy_idx:
            clean_frags.append(heavy_idx)
    if len(clean_frags) > 1:
        for frag_idx, atom_indices in enumerate(clean_frags):
            for a_idx in atom_indices:
                frag_ids[a_idx] = frag_idx
    return frag_ids


def get_rotatable_bond_frag_ids(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetDegree() > 1 and end.GetDegree() > 1:
            rot_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    if not rot_bonds:
        return [0] * n_atoms
    rw_mol = Chem.RWMol(mol)
    for a1, a2 in rot_bonds:
        bond = rw_mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            rw_mol.RemoveBond(a1, a2)
    broken = rw_mol.GetMol()
    frags = Chem.GetMolFrags(broken, asMols=False)
    frag_ids = [0] * n_atoms
    for i, frag in enumerate(frags):
        for atom_idx in frag:
            if atom_idx < n_atoms:
                frag_ids[atom_idx] = i
    return frag_ids


# ==================== 数据集 ====================
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels, task_type="regression", desc="",
                 env_labels=None, cache_path=None, recal=False):
        if cache_path is not None and os.path.isfile(cache_path) and recal == False:
            print(f"  Loading cached dataset from {cache_path}")
            self.data_list = torch.load(cache_path)
            self.failed = []
            print(f"  {desc}: {len(self.data_list)} molecules loaded from cache.")
            return

        self.data_list = []
        self.failed = []

        for idx, (smi, y) in enumerate(zip(smiles_list, labels)):
            if idx % 500 == 0:
                print(f"  Processing {desc}: {idx}/{len(smiles_list)}")

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                self.failed.append((idx, smi, "invalid"))
                continue
            mol = Chem.RemoveHs(mol)
            try:
                x = get_atom_features_39(mol)
                edge_index, edge_attr = get_edge_features(mol)

                if not (torch.isfinite(x).all() and torch.isfinite(edge_attr).all()):
                    self.failed.append((idx, smi, "non-finite"))
                    continue

                frag_ids = get_brics_frag_ids(mol)
                frag_ids = torch.tensor(frag_ids, dtype=torch.long)

                rot_frag_ids = get_rotatable_bond_frag_ids(mol)
                rot_frag_ids = torch.tensor(rot_frag_ids, dtype=torch.long)

                atom_centrality = compute_atom_centrality(mol)
                atom_centrality = torch.tensor(atom_centrality, dtype=torch.float32)

                if task_type in ["multitask"]:
                    y_tensor = torch.tensor(y, dtype=torch.float)
                elif task_type in ["classification", "multiclass"]:
                    y_tensor = torch.tensor(int(y), dtype=torch.long)
                else:
                    y_tensor = torch.tensor(float(y), dtype=torch.float)

                data = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y_tensor, frag_ids=frag_ids, rot_frag_ids=rot_frag_ids,
                    atom_centrality=atom_centrality,
                )
                if env_labels is not None:
                    data.env_id = torch.tensor(env_labels[idx], dtype=torch.long)
                else:
                    data.env_id = torch.tensor(0, dtype=torch.long)

                self.data_list.append(data)
            except Exception as e:
                self.failed.append((idx, smi, str(e)))

        print(f"  {desc}: {len(self.data_list)} kept, {len(self.failed)} failed.")

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            torch.save(self.data_list, cache_path)
            print(f"  Dataset cached to {cache_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


    
def my_collate(data_list):
    ys = [data.y for data in data_list]
    clean_list = [
        Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr)
        for d in data_list
    ]
    batch = Batch.from_data_list(clean_list)
    if ys[0].ndim == 1:
        batch.y = torch.stack(ys, dim=0)
    else:
        batch.y = torch.tensor([y.item() for y in ys], dtype=ys[0].dtype)
    batch.true_frag_ids = torch.cat([d.frag_ids for d in data_list], dim=0)
    batch.rot_frag_ids = torch.cat([d.rot_frag_ids for d in data_list], dim=0)
    batch.env_ids = torch.tensor([d.env_id for d in data_list], dtype=torch.long)
    batch.atom_centrality = torch.cat([d.atom_centrality for d in data_list], dim=0)
    return batch


# ==================== Sum 池化 ====================
class SumPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        return global_add_pool(x, batch), torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass
    
class MaxPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        return global_max_pool(x, batch), torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass
    
class MeanPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        return global_mean_pool(x, batch), torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass
    
class MeanMaxPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        return global_mean_pool(x, batch)+global_max_pool(x, batch), torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass        

# ==================== Readout 池化（baseline）====================

class Set2SetPooling(nn.Module):
    """
    Set2Set 读出（Vinyals et al. 2015）：
      - LSTM + attention 迭代读取
      - 输出维度 = 2 * in_channels（读出的 (q, r) 拼接）
    """
    def __init__(self, in_channels, processing_steps=2, num_layers=1, **kwargs):
        super().__init__()
        self.set2set = Set2Set(
            in_channels, processing_steps=processing_steps, num_layers=num_layers
        )
        self.out_dim = 2 * in_channels
        self.needs_frag_ids = False
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        mol_repr = self.set2set(x, batch)
        return mol_repr, torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass



class GMTPooling(nn.Module):
    """
    GMT 全局池化包装类（Graph Multiset Transformer, ICLR 2021）。

    与 SumPooling / Set2SetPooling 接口一致：
        forward(x, edge_index, edge_attr, batch, **kwargs) -> (mol_repr, pool_loss)

    GMT 结构：
        - GMPool（第一层）：把节点压缩成 k 个代表性节点
        - Self-Attention Block：k 个节点互相交互
        - GMPool（第二层）：压成 1 个图级向量

    参数：
        in_channels: 输入节点特征维度
        k:           代表性节点数（默认 10）
        num_encoder_blocks: Self-Attention 层数（默认 1）
        heads:       多头注意力头数（默认 4）
        layer_norm:  是否用 LayerNorm
        dropout:     dropout 比率
    """
    def __init__(self, in_channels, k=10, num_encoder_blocks=1,
                 heads=4, layer_norm=True, dropout=0.0):
        super().__init__()
        self.pool = GraphMultisetTransformer(
            channels=in_channels,
            k=k,
            num_encoder_blocks=num_encoder_blocks,
            heads=heads,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        self.out_dim = in_channels

        # 告诉骨架：不需要 frag_ids / rot_frag_ids
        self.needs_frag_ids = False
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        # GMT 只依赖 x 和 batch，不需要 edge_index / edge_attr
        mol_repr = self.pool(x, batch)
        return mol_repr, torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass    


class GlobalAttentionPooling(nn.Module):
    def __init__(self, in_channels, gate_hidden=None, **kwargs):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)
        gate_nn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )
        self.pool = GlobalAttention(gate_nn)
        self.out_dim = in_channels
        self.needs_frag_ids = False
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.last_atom_weight = None

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        size = int(batch.max().item()) + 1

        # 复现 GlobalAttention.forward 的内部逻辑
        gate = self.pool.gate_nn(x).view(-1, 1)
        x_in = self.pool.nn(x) if self.pool.nn is not None else x
        alpha = softmax(gate, batch, num_nodes=size)
        mol_repr = scatter(alpha * x_in, batch, dim=0, dim_size=size, reduce='sum')

        # 存下来供可解释性分析
        self.last_atom_weight = alpha.squeeze(-1).detach()
        return mol_repr, torch.tensor(0.0, device=x.device)
    
# ==================== DiffPool / MinCutPool 池化模块（DenseGINConv 版）====================


# ==================== DiffPool / MinCutPool 完整模型 ====================

# ==================== DiffPool / MinCutPool 独立模型 ====================

def _build_dense_gin_mlp(in_dim, out_dim, hidden_dim, is_assign=False):
    """DenseGINConv 内部 MLP。assign 最后一层不加激活（输出 logits）。"""
    if is_assign:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),#diffpool not use it
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

# ==================== AssignModule ====================
class AssignModule(nn.Module):
    """
    DiffPool 的 assign 分支：
        DenseGINConv / DenseGCNConv → ReLU → Linear → K 个 cluster logits

    只支持 backbone='gcn' 和 backbone='gin'。
    """
    def __init__(self, hidden_channels, num_clusters, backbone='gin'):
        super().__init__()
        self.backbone = backbone.lower()
        if self.backbone not in ('gcn', 'gin'):
            raise ValueError("AssignModule only supports backbone in {'gcn', 'gin'}")

        if self.backbone == 'gin':
            self.gnn = DenseGINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))
        else:  # gcn
            self.gnn = DenseGCNConv(hidden_channels, hidden_channels)

        self.mlp = nn.Linear(hidden_channels, num_clusters)

    def forward(self, x, adj, mask):
        s = self.gnn(x, adj, mask)
        if self.backbone == 'gin':
            s = F.layer_norm(s, (s.size(-1),))

        # DenseGINConv 内部 MLP 已有 ReLU；DenseGCNConv 内部没有，
        # 外层统一再 ReLU 一次，GCN 时这是唯一非线性
        s = F.relu(s)
        s = self.mlp(s)          # (B, N, K) logits
        return s


# ==================== DiffPoolModel ====================
class DiffPoolModel(nn.Module):
    """
    官方 DiffPool 结构（Ying et al. 2018）：
        conv1（稀疏 GNN 升维）
        → [assign_k + embed_k 并行 → dense_diff_pool] × num_pool_layers
        → embed_last（最后一层只 embed，不池化）
        → readout → Dense → Out

    - assign 分支：AssignModule（DenseGINConv / DenseGCNConv + Linear）
    - embed  分支：DenseGINConv / DenseGCNConv，可选 norm
    - 池化用 embed 的输出 h 和 assign 的输出 s
    - 只支持 backbone='gcn' 和 backbone='gin'
    """
    def __init__(self, in_channels, out_channels, edge_dim=10,
                 hidden_channels=128,
                 max_node=150, ratio=0.5,
                 num_layers=3,
                 min_clusters=2,
                 dense_layers=[128, 64],
                 dropout=0.0,
                 norm=None,
                 link_weight=0.5, entropy_weight=0.5,
                 readout='mean',
                 backbone='gin'):
        super().__init__()
        self.max_node = max_node
        self.ratio = ratio
        self.num_layers = num_layers
        self.num_pool_layers = max(num_layers - 1, 0)
        self.hidden_channels = hidden_channels
        self.link_weight = link_weight
        self.entropy_weight = entropy_weight
        self.readout = readout
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.backbone = backbone.lower()
        if self.backbone not in ('gcn', 'gin'):
            raise ValueError("DiffPoolModel only supports backbone in {'gcn', 'gin'}")
        self.assign_norm=True if backbone=="gin" else False
        self.edge_dim = edge_dim
        # GCN / GIN 都不使用 edge_attr
        self._use_edge_attr = False

        # 每层池化簇数
        self.num_clusters_list = self._make_cluster_list(
            self.num_pool_layers, max_node, ratio, min_clusters)

        # ==================== 第 1 层：稀疏 GNN 升维 ====================
        self.conv1 = self._build_sparse_conv(in_channels, hidden_channels)

        # ==================== assign / embed 双分支 ====================
        self.assign_modules = nn.ModuleList()
        self.embed_convs = nn.ModuleList()
        self.embed_norms = nn.ModuleList() if norm is not None else None

        for k in range(self.num_pool_layers):
            K = self.num_clusters_list[k]

            # assign：DenseGINConv / DenseGCNConv + Linear
            self.assign_modules.append(
                AssignModule(hidden_channels, K, backbone=self.backbone)
            )

            # embed：DenseGINConv / DenseGCNConv + 可选 norm
            self.embed_convs.append(
                self._build_dense_conv(hidden_channels, hidden_channels, hidden_channels)
            )
            if norm is not None:
                self.embed_norms.append(norm(hidden_channels))

        # ==================== 最后一层：只 embed，不池化 ====================
        self.last_embed = self._build_dense_conv(
            hidden_channels, hidden_channels, hidden_channels
        )
        self.last_norm = norm(hidden_channels) if norm is not None else None

        # ==================== 分类头 ====================
        if readout == 'mean_max':
            pool_out_dim = hidden_channels * 2
        else:
            pool_out_dim = hidden_channels
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    # ==================== 工具函数 ====================
    @staticmethod
    def _make_cluster_list(num_pool_layers, max_node, ratio, min_clusters):
        k_list = []
        k = max_node
        for _ in range(num_pool_layers):
            k = max(int(k * ratio), min_clusters)
            k_list.append(k)
        return k_list

    def _build_sparse_conv(self, in_dim, out_dim):
        if self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(nn=mlp)
        elif self.backbone == 'gcn':
            return GCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _build_dense_conv(self, in_dim, out_dim, hidden_dim):
        if self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            return DenseGINConv(mlp)
        elif self.backbone == 'gcn':
            return DenseGCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _readout_fn(self, x_dense):
        if self.readout == 'mean':
            return x_dense.mean(dim=1)
        elif self.readout == 'sum':
            return x_dense.sum(dim=1)
        elif self.readout == 'max':
            return x_dense.max(dim=1)[0]
        elif self.readout == 'mean_max':
            return torch.cat([x_dense.mean(dim=1),
                              x_dense.max(dim=1)[0]], dim=-1)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")

    # ==================== forward ====================
    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        # ==================== 稀疏 conv1 升维 ====================
        if self._use_edge_attr:
            h = self.conv1(x, edge_index, edge_attr)
        else:
            h = self.conv1(x, edge_index)
        h = F.relu(h)

        # ==================== 稠密化 ====================
        x_dense, mask = to_dense_batch(h, batch, max_num_nodes=self.max_node)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_node)

        total_link = torch.tensor(0.0, device=x.device)
        total_ent = torch.tensor(0.0, device=x.device)

        # ==================== assign + embed + pool ====================
        for k in range(self.num_pool_layers):
            # assign：AssignModule（DenseGINConv / DenseGCNConv + Linear）
            s = self.assign_modules[k](x_dense, adj, mask)   # (B, N, K)

            # embed：DenseGINConv / DenseGCNConv + 可选 norm
            h = self.embed_convs[k](x_dense, adj, mask)
            h = F.relu(h)
            if self.embed_norms is not None:
                h = self.embed_norms[k](h)

            # pool
            x_dense, adj, link_loss, ent_loss = dense_diff_pool(
                h, adj, s, mask)
            total_link = total_link + link_loss
            total_ent = total_ent + ent_loss
            mask = None

        # ==================== 最后一层 embed（不池化） ====================
        x_dense = self.last_embed(x_dense, adj, mask)
        x_dense = F.relu(x_dense)
        if self.last_norm is not None:
            x_dense = self.last_norm(x_dense)

        # ==================== readout ====================
        y = self._readout_fn(x_dense)

        # ==================== 分类头 ====================
        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)

        pool_loss = self.link_weight * total_link + self.entropy_weight * total_ent
        return y, pool_loss

    def set_epoch(self, epoch):
        pass


class MinCutPoolModel(nn.Module):
    """
    标准 MinCutPool（Bianchi et al. ICML 2020），参考 PyG 官方示例：
        conv1 → pool1 → conv2 → pool2 → conv3 → mean → Dense → Out

    结构：
      - 第 1 层：稀疏图卷积，负责升维
      - 之后稠密化，后续层用 DenseGCNConv 或 DenseGINConv
      - 前 N-1 层之后池化，最后一层不池化
      - assign: DenseGCNConv / DenseGINConv → ReLU → Linear
      - embed : 就是 conv 本身
      - 每次池化前后给 adj 加自环

    只支持 backbone='gcn' 和 backbone='gin'。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim=10,
        hidden_channels=128,
        max_node=150,
        ratio=0.5,
        num_layers=3,
        min_clusters=2,
        dense_layers=[128, 64],
        dropout=0.0,
        norm=None,
        mincut_weight=0.5,
        ortho_weight=0.5,
        temp=1.0,
        backbone='gin',
    ):
        super().__init__()
        self.max_node = max_node
        self.ratio = ratio
        self.num_layers = num_layers
        self.num_pool_layers = max(num_layers - 1, 0)
        self.hidden_channels = hidden_channels
        self.mincut_weight = mincut_weight
        self.ortho_weight = ortho_weight
        self.temp = temp
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.add_loop=False if backbone=='gin' else True
        self.backbone = backbone.lower()
        if self.backbone not in ('gcn', 'gin'):
            raise ValueError("Only support backbone in {'gcn', 'gin'}")

        self.edge_dim = edge_dim
        # GCN / GIN 都不使用 edge_attr
        self._use_edge_attr = False

        # 每层簇数
        self.num_clusters_list = self._make_cluster_list(
            self.num_pool_layers, max_node, ratio, min_clusters
        )

        # ==================== 第 1 层：稀疏 GNN（升维） ====================
        self.conv1 = self._build_sparse_conv(in_channels, hidden_channels)

        # ==================== 第 2 到第 N 层：稠密 GNN ====================
        self.dense_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.dense_convs.append(
                self._build_dense_conv(
                    hidden_channels, hidden_channels, hidden_channels
                )
            )

        # ==================== assign 分支（每层池化对应一个） ====================
        self.assign_gnns = nn.ModuleList()
        self.assign_mlps = nn.ModuleList()
        for K in self.num_clusters_list:
            self.assign_gnns.append(
                self._build_dense_conv(
                    hidden_channels, hidden_channels, hidden_channels
                )
            )
            self.assign_mlps.append(nn.Linear(hidden_channels, K))

        # ==================== norm ====================
        if norm is not None:
            self.norm1 = norm(hidden_channels)
            self.norms_dense = nn.ModuleList(
                [norm(hidden_channels) for _ in range(num_layers - 1)]
            )
        else:
            self.norm1 = None
            self.norms_dense = None

        # ==================== 分类头 ====================
        self.pool_out_dim = hidden_channels
        _dense = [self.pool_out_dim] + dense_layers
        self.lins = nn.ModuleList(
            [
                nn.Linear(_dense[i], _dense[i + 1])
                for i in range(len(_dense) - 1)
            ]
        )
        self.out = nn.Linear(_dense[-1], out_channels)

    @staticmethod
    def _make_cluster_list(num_pool_layers, max_node, ratio, min_clusters):
        k_list = []
        k = max_node
        for _ in range(num_pool_layers):
            k = max(int(k * ratio), min_clusters)
            k_list.append(k)
        return k_list

    def _build_sparse_conv(self, in_dim, out_dim):
        if self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(nn=mlp)
        elif self.backbone == 'gcn':
            return GCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _build_dense_conv(self, in_dim, out_dim, hidden_dim):
        if self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            return DenseGINConv(mlp)
        elif self.backbone == 'gcn':
            return DenseGCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _add_self_loops(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        return adj + eye.unsqueeze(0)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        # ==================== 第 1 层：稀疏 GNN ====================
        if self._use_edge_attr:
            h = self.conv1(x, edge_index, edge_attr)
        else:
            h = self.conv1(x, edge_index)

        if self.norm1 is not None:
            h = self.norm1(h)
        h = F.relu(h)

        # ==================== 稠密化 ====================
        x_dense, mask = to_dense_batch(h, batch, max_num_nodes=self.max_node)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_node)
        adj = self._add_self_loops(adj)

        total_mincut = torch.tensor(0.0, device=x.device)
        total_ortho = torch.tensor(0.0, device=x.device)

        # ============ 第 1 到 N-1 次池化 + 第 2 到 N 层 conv ============
        for k in range(self.num_pool_layers):
            # ---- pool_{k+1} ----
            K = self.num_clusters_list[k]
            if K < x_dense.size(1):
                x_bar = self.assign_gnns[k](x_dense, adj, mask,add_loop=self.add_loop)
                
                if self.backbone == 'gin':
                    x_bar = F.layer_norm(x_bar, (x_bar.size(-1),))
                # DenseGINConv 内部 MLP 已有 ReLU；
                # DenseGCNConv 内部没有，所以 GCN 时补一个 ReLU
                if self.backbone == 'gcn':
                    x_bar = F.relu(x_bar)

                s = self.assign_mlps[k](x_bar)
                x_dense, adj, ml, ol = dense_mincut_pool(
                    x_dense, adj, s, mask, temp=self.temp
                )
                adj = self._add_self_loops(adj)
                total_mincut = total_mincut + ml
                total_ortho = total_ortho + ol
                mask = None

            # ---- conv_{k+2}（稠密） ----
            h = self.dense_convs[k](x_dense, adj, mask,add_loop=self.add_loop)
            h = F.relu(h)
            if self.norms_dense is not None:
                h = self.norms_dense[k](h)

            x_dense = h

        # ==================== mean 聚合 ====================
        y = x_dense.sum(dim=1)

        # ==================== 分类头 ====================
        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)

        pool_loss = (
            self.mincut_weight * total_mincut
            + self.ortho_weight * total_ortho
        )
        return y, pool_loss

    def set_epoch(self, epoch):
        pass

class BRICSFragmentPoolOriginal(nn.Module):
    """
    原版 BRICS 片段池化（单视图）：
    - 片段内：原子注意力 + 门控融合（sum/mean/max）
    - 片段间：跟随 model_type 的 GNN
    - 分子级：注意力聚合
    - 辅助损失：门控熵正则
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, aggr='attention', gate_hidden=None,
                 model_type='gine'):     # ★ 新增 model_type
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout
        self.use_layernorm = use_layernorm
        self.aggr = aggr
        self.model_type = model_type.lower()   # ★ 保存

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        self.attn_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1)
        )

        gate_input_dim = in_channels + 1 + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3)
        )

        # ★ 片段间 GNN（跟随 model_type）
        self.frag_convs = nn.ModuleList()
        self.frag_norms = nn.ModuleList() if use_layernorm else None
        for _ in range(num_gnn_layers):
            conv = _build_frag_conv(self.model_type, in_channels, hidden_channels, dropout)
            self.frag_convs.append(conv)
            if use_layernorm:
                self.frag_norms.append(LayerNorm(in_channels))

        if aggr == 'attention':
            self.mol_attn = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 2, 1)
            )
        else:
            self.mol_attn = None

        self.dropout = nn.Dropout(dropout)
        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 可解释性缓存
        self.last_atom_weight = None
        self.last_frag_weight = None
        self.last_gate_weights = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSFragmentPoolOriginal 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 构建全局片段 ID ====================
        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # ==================== 2. 片段内聚合 ====================
        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        # ==================== 3. 片段间 GNN（★ 跟随 model_type）====================
        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask], frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        # ==================== 4. 分子级聚合 ====================
        if self.aggr == 'attention':
            attn_score = self.mol_attn(h_group).squeeze(-1)
            frag_attn_weight = softmax(attn_score, frag_batch)
            mol_repr = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                               frag_batch, dim=0, dim_size=B, reduce='sum')
        elif self.aggr == 'mean':
            frag_attn_weight = torch.ones(num_frags, device=device) / \
                scatter(torch.ones(num_frags, device=device), frag_batch,
                        dim=0, dim_size=B, reduce='sum')[frag_batch]
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='mean')
        elif self.aggr == 'sum':
            frag_attn_weight = torch.ones(num_frags, device=device)
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='sum')
        elif self.aggr == 'max':
            frag_attn_weight = torch.ones(num_frags, device=device)
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='max')
        else:
            raise ValueError("aggr must be 'mean', 'sum', 'max', or 'attention'")

        # ==================== 5. 辅助损失 + 可解释性缓存 ====================
        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        self.last_atom_weight = attn_weights.detach()
        self.last_frag_weight = frag_attn_weight.detach()
        self.last_gate_weights = gate_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass    
    
class BRICSAtomCrossAttnFusionMultiStatPool(BRICSFragmentPoolOriginal):
    """
    BRICS 片段视图 + 原子视图，cross-attention 融合（双多统计量版）。

    与 v34 / v2 / signed 的核心差异：
      - ★ 片段内聚合（原子 → 片段）：
          原来是 attention + 门控融合(sum/mean/max)
          改为：[mean, max, sum, attn] concat → MLP → h_group
      - ★ 片段级聚合（片段 → 分子）：
          原来是 softmax attention 加权
          改为：[mean, max, sum, attn] concat → MLP → mol_repr_frag
      - 其他保持一致：片段间 GNN、原子视图、cross-attention 融合、alpha 融合

    和原子视图完全对称：
      - 片段内：4 统计量 concat → frag_intra_proj
      - 片段级：4 统计量 concat → frag_stat_proj
      - 原子级：4 统计量 concat → atom_proj

    可解释性缓存：
      - last_atom_weight : (N,)        片段内原子 attention 权重
      - last_frag_weight : (num_frags,) 片段级 attention 权重
      - last_alpha       : 标量
      - last_attn        : (B, 2, 2)   2-token attention 权重
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',          # 保留父类 mol_attn（片段级打分）
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        self.in_channels = in_channels
        self.de_corr_weight = de_corr_weight

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ★ 父类的 gate_mlp 不再使用，从参数表中移除（省参数）
        self.gate_mlp = None

        # ==================== ★ 片段内聚合：四统计量 concat → MLP ====================
        # 输入 [mean, max, sum, attn] 共 4*in_channels，映射回 in_channels
        self.frag_intra_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== ★ 片段级聚合：四统计量 concat → MLP ====================
        self.frag_stat_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 原子视图 ====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 2-token 视图类型 embedding ====================
        self.register_buffer(
            'view_type_embed',
            torch.stack([
                torch.ones(in_channels),
                -torch.ones(in_channels)
            ]) * 0.1
        )

        # ==================== 2-token MHA + FFN ====================
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

        # ==================== 融合权重 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # ==================== 缓存 ====================
        self.last_alpha = None
        self.last_attn = None
        self.last_atom_weight = None
        self.last_frag_weight = None

    # ==================== 片段池化 ====================
    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 全局片段 ID ====================
        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # ==================== 2. ★ 片段内聚合：四统计量 concat → MLP ====================
        # 原子级 attention（片段内 softmax）
        attn_logits = self.attn_mlp(x).squeeze(-1)      # (N,)
        atom_weights = softmax(attn_logits, frag_idx)   # (N,) 片段内归一化

        a_mean = scatter(x, frag_idx, dim=0,
                         dim_size=num_frags, reduce='mean')
        a_max  = scatter(x, frag_idx, dim=0,
                         dim_size=num_frags, reduce='max')
        a_sum  = scatter(x, frag_idx, dim=0,
                         dim_size=num_frags, reduce='sum')
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, frag_idx,
                         dim=0, dim_size=num_frags, reduce='sum')

        h_group = self.frag_intra_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )  # (num_frags, in_channels)

        # ==================== 3. 片段间 GNN ====================
        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        # ==================== 4. ★ 片段级聚合：四统计量 concat → MLP ====================
        mol_mean = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='mean')
        mol_max  = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='max')
        mol_sum  = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='sum')

        frag_attn_logits = self.mol_attn(h_group).squeeze(-1)
        frag_attn_weight = softmax(frag_attn_logits, frag_batch)
        mol_attn = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                           frag_batch, dim=0, dim_size=B, reduce='sum')

        mol_repr_frag = self.frag_stat_proj(
            torch.cat([mol_mean, mol_max, mol_sum, mol_attn], dim=-1)
        )

        # 没有门控熵损失了
        pool_loss = torch.tensor(0.0, device=device)

        # 缓存
        self.last_atom_weight = atom_weights.detach()
        self.last_frag_weight = frag_attn_weight.detach()

        return mol_repr_frag, pool_loss

    # ==================== 主 forward ====================
    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomCrossAttnFusionMultiStatPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图 ====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max  = global_max_pool(x, batch)
        a_sum  = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. 2-token cross-attention ====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)   # (B, 2, d)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :]

        # ==================== 4. 融合（与 v34 一致）====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== 5. 正则 ====================
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + self.de_corr_weight * sim ** 2

        # ==================== 6. 缓存 ====================
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass    
    
class GNN_Base_HierarchicalEdgePool(nn.Module):
    """
    EdgePool 专用层次化骨干：GNN → EdgePool → GNN → EdgePool → ... → 全局读出。

    与 GNN_Base_HierarchicalSAGTopK 的差异：
      - 池化用 PyG_EdgePooling（边收缩）
      - EdgePooling 内部会计算边分数，dropout 需要传给 pool
      - EdgePool 返回 (x, edge_index, edge_attr, batch, unpool_info, edge_score)
      - 不需要 ratio 参数，EdgePool 通过边分数迭代收缩到目标比例

    EdgePool 关键参数：
      - add_to_edge_score: 加到边分数上的常数，默认 0.5，文档说"极大地帮助 unpool 稳定性"
      - dropout: 建议 0.2（原论文配置）
    """
    def __init__(self, in_channels, out_channels, edge_dim,
                 ratio=0.5,
                 convs_layers=[128, 128, 128],
                 dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0,
                 backbone='gin', heads=4, **kwargs):
        super().__init__()
        self.backbone = backbone.lower()
        self.ratio = ratio
        self.dropout_p = dropout_p
        self.edge_dim = edge_dim
        self._use_edge_attr = self.backbone in ('gine', 'gat', 'gatv2')

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(len(_convs) - 1):
            self.convs.append(self._build_conv(
                _convs[i], _convs[i + 1], edge_dim, dropout_p, heads))
            # EdgePool：in_channels = 该层输出维度
            # add_to_edge_score=0.5（默认），dropout 用 0.2（原论文）
            self.pools.append(PyG_EdgePooling(
                in_channels=_convs[i + 1],
                dropout=0.2,
                add_to_edge_score=0.5,
            ))

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        _dense = [_convs[-1]] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def _build_conv(self, in_dim, out_dim, edge_dim, dropout, heads):
        if self.backbone == 'gine':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn=mlp, edge_dim=edge_dim)
        elif self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(nn=mlp)
        elif self.backbone == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif self.backbone == 'gat':
            return GATConv(in_dim, out_dim, heads=heads, concat=False,
                           edge_dim=edge_dim, dropout=dropout)
        elif self.backbone == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _apply_conv(self, conv, x, edge_index, edge_attr):
        if self._use_edge_attr:
            return conv(x, edge_index, edge_attr)
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = self._apply_conv(conv, x, edge_index, edge_attr)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            # ==================== EdgePool 池化（GIN 不传 edge_attr）====================
            if self._use_edge_attr:
                # GINE / GAT / GATv2：需要 edge_attr
                out = pool(x, edge_index, edge_attr=edge_attr, batch=batch)
                if len(out) == 6:
                    x, edge_index, edge_attr, batch, _, _ = out
                else:
                    # 老版本不返回 edge_attr
                    x, edge_index, batch, _, _ = out
                    edge_attr = torch.zeros(edge_index.size(1), self.edge_dim,
                                            device=x.device)
            else:
                # GIN / GCN：不需要 edge_attr
                out = pool(x, edge_index, batch=batch)
                if len(out) == 5:
                    # 老版本：返回 (x, edge_index, batch, unpool_info, edge_score)
                    x, edge_index, batch, _, _ = out
                elif len(out) == 6:
                    # 新版本：返回 (x, edge_index, edge_attr, batch, unpool_info, edge_score)
                    x, edge_index, _, batch, _, _ = out
                else:
                    x, edge_index, batch = out[0], out[1], out[2]
                # GIN 不用 edge_attr，重建一个零向量占位，防止后续传参报错
                edge_attr = torch.zeros(edge_index.size(1), self.edge_dim,
                                        device=x.device)

        y = global_add_pool(x, batch)
        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass    
    
# ==================== 片段间 GNN 动态构建 ====================

def _build_frag_conv(model_type, in_channels, hidden_channels, dropout=0.0):
    """根据 model_type 构建片段间 GNN 层"""
    model_type = model_type.lower()
    if model_type == "gine":
        return GINEConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels)
            ),
            edge_dim=10
        )
    elif model_type == "gin":
        return GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels)
            )
        )
    elif model_type == "gcn":
        return GCNConv(in_channels, in_channels)
    elif model_type == "gat":
        return GATConv(
            in_channels, in_channels,
            heads=4, concat=False, edge_dim=10, dropout=dropout
        )
    elif model_type == "gatv2":
        return GATv2Conv(
            in_channels, in_channels,
            heads=4, concat=False, edge_dim=10, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _apply_frag_conv(conv, h, edge_index, edge_attr, model_type):
    """根据 model_type 决定是否传 edge_attr"""
    model_type = model_type.lower()
    if model_type in ("gine", "gat", "gatv2"):
        return conv(h, edge_index, edge_attr)
    else:  # gin, gcn
        return conv(h, edge_index)


# ==================== 四个池化模块 ====================

class BRICSFragmentPoolOriginal(nn.Module):
    """
    原版 BRICS 片段池化（单视图）：
    - 片段内：原子注意力 + 门控融合（sum/mean/max）
    - 片段间：跟随 model_type 的 GNN
    - 分子级：注意力聚合
    - 辅助损失：门控熵正则
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, aggr='attention', gate_hidden=None,
                 model_type='gine'):     # ★ 新增 model_type
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout
        self.use_layernorm = use_layernorm
        self.aggr = aggr
        self.model_type = model_type.lower()   # ★ 保存

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        self.attn_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1)
        )

        gate_input_dim = in_channels + 1 + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3)
        )

        # ★ 片段间 GNN（跟随 model_type）
        self.frag_convs = nn.ModuleList()
        self.frag_norms = nn.ModuleList() if use_layernorm else None
        for _ in range(num_gnn_layers):
            conv = _build_frag_conv(self.model_type, in_channels, hidden_channels, dropout)
            self.frag_convs.append(conv)
            if use_layernorm:
                self.frag_norms.append(LayerNorm(in_channels))

        if aggr == 'attention':
            self.mol_attn = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 2, 1)
            )
        else:
            self.mol_attn = None

        self.dropout = nn.Dropout(dropout)
        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 可解释性缓存
        self.last_atom_weight = None
        self.last_frag_weight = None
        self.last_gate_weights = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSFragmentPoolOriginal 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 构建全局片段 ID ====================
        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # ==================== 2. 片段内聚合 ====================
        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        # ==================== 3. 片段间 GNN（★ 跟随 model_type）====================
        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask], frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        # ==================== 4. 分子级聚合 ====================
        if self.aggr == 'attention':
            attn_score = self.mol_attn(h_group).squeeze(-1)
            frag_attn_weight = softmax(attn_score, frag_batch)
            mol_repr = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                               frag_batch, dim=0, dim_size=B, reduce='sum')
        elif self.aggr == 'mean':
            frag_attn_weight = torch.ones(num_frags, device=device) / \
                scatter(torch.ones(num_frags, device=device), frag_batch,
                        dim=0, dim_size=B, reduce='sum')[frag_batch]
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='mean')
        elif self.aggr == 'sum':
            frag_attn_weight = torch.ones(num_frags, device=device)
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='sum')
        elif self.aggr == 'max':
            frag_attn_weight = torch.ones(num_frags, device=device)
            mol_repr = scatter(h_group, frag_batch, dim=0, dim_size=B, reduce='max')
        else:
            raise ValueError("aggr must be 'mean', 'sum', 'max', or 'attention'")

        # ==================== 5. 辅助损失 + 可解释性缓存 ====================
        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        self.last_atom_weight = attn_weights.detach()
        self.last_frag_weight = frag_attn_weight.detach()
        self.last_gate_weights = gate_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass
    

   
class BRICSAtomFusionPool(nn.Module):
    """
    BRICS 片段视图（brics_frag_original_23）+ 原子视图，可学习方向加法残差融合：

    - 片段视图：完全复用 BRICSFragmentPoolOriginal
        · 片段内：attention + 门控(sum/mean/max)
        · 片段间：GNN
        · 分子级：attention
    - 原子视图：原子级 mean/max/sum/attn 拼接 → 投影
    - 融合：
        g = sigmoid(view_gate([mol_repr_frag, mol_repr_atom]))   # 逐样本
        base     = g * mol_repr_frag + (1-g) * mol_repr_atom
        residual = (1-g) * mol_repr_frag + g * mol_repr_atom
        mol_repr = base + alpha * residual
      alpha = sigmoid(raw_alpha) * alpha_max，受 L2 稀疏惩罚
      view_gate 初始化为 0 → 初始 50/50，训练中自己决定哪个当主干
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', alpha_init=-2.0, alpha_max=1.0,
                 alpha_reg=0.01):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ==================== 片段视图（主干候选 1）====================
        self.frag_pool = BRICSFragmentPoolOriginal(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        # ==================== 原子视图（主干候选 2）====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 逐样本视图门控 ====================
        self.view_gate = nn.Sequential(
            nn.Linear(2 * in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        nn.init.constant_(self.view_gate[-1].bias, 0.0)   # 初始 50/50

        # ==================== 残差强度 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 缓存（分析用）
        self.last_alpha = None
        self.last_view_gate = None
        self.last_atom_weight = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomFusionPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 片段视图 ====================
        mol_repr_frag, pool_loss = self.frag_pool(
            x, edge_index, edge_attr, batch, true_frag_ids=true_frag_ids
        )

        # ==================== 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 逐样本门控 + 加法残差 ====================
        cat = torch.cat([mol_repr_frag, mol_repr_atom], dim=-1)
        g = torch.sigmoid(self.view_gate(cat))            # (B, 1)

        base = g * mol_repr_frag + (1 - g) * mol_repr_atom
        residual = (1 - g) * mol_repr_frag + g * mol_repr_atom

        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = base + alpha * residual

        # ==================== 正则 + 缓存 ====================
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2

        self.last_alpha = alpha.detach()
        self.last_view_gate = g.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        self.frag_pool.set_epoch(epoch)
        
class BRICSAtomConcatFusionPool(nn.Module):
    """
    BRICS 片段视图（brics_frag_original_23）+ 原子视图，逐分子门控融合。
    门控特征仅用 [mol_repr_frag, mol_repr_atom] 的简单拼接（concat）。

    - 片段视图：完全复用 BRICSFragmentPoolOriginal
        · 片段内：attention + 门控(sum/mean/max)
        · 片段间：GNN
        · 分子级：attention
    - 原子视图：原子级 mean/max/sum/attn 拼接 → 投影
    - 门控：g = sigmoid(MLP([frag, atom]))            # (B, 1) 逐分子
    - 融合：
        base     = g * frag + (1-g) * atom
        residual = (1-g) * frag + g * atom
        mol_repr = base + alpha * residual
      alpha = sigmoid(raw_alpha) * alpha_max，L2 稀疏惩罚
    - view_gate bias 初始化为 0 → 初始 50/50
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', alpha_init=-2.0, alpha_max=1.0,
                 alpha_reg=0.01):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ==================== 片段视图（主干候选 1）====================
        self.frag_pool = BRICSFragmentPoolOriginal(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        # ==================== 原子视图（主干候选 2）====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 逐分子门控（仅 concat）====================
        self.view_gate = nn.Sequential(
            nn.Linear(2 * in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        nn.init.constant_(self.view_gate[-1].bias, 0.0)   # 初始 50/50

        # ==================== 残差强度 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 缓存（分析用）
        self.last_alpha = None
        self.last_view_gate = None
        self.last_atom_weight = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomConcatFusionPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 片段视图 ====================
        mol_repr_frag, pool_loss = self.frag_pool(
            x, edge_index, edge_attr, batch, true_frag_ids=true_frag_ids
        )

        # ==================== 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 门控：仅 concat ====================
        gate_in = torch.cat([mol_repr_frag, mol_repr_atom], dim=-1)
        g = torch.sigmoid(self.view_gate(gate_in))            # (B, 1)

        # ==================== 加法残差融合 ====================
        base = g * mol_repr_frag + (1 - g) * mol_repr_atom
        residual = (1 - g) * mol_repr_frag + g * mol_repr_atom

        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = base + alpha * residual

        # ==================== 正则 + 缓存 ====================
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2

        self.last_alpha = alpha.detach()
        self.last_view_gate = g.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        self.frag_pool.set_epoch(epoch)
        
        
class BRICSAtomCrossAttnFusionPool(nn.Module):
    """
    BRICS 片段视图（brics_frag_original_23）+ 原子视图，用 cross-attention 融合。

    设计：
      - 片段视图：完全复用 BRICSFragmentPoolOriginal
          · 片段内：attention + 门控(sum/mean/max)
          · 片段间：GNN
          · 分子级：attention
      - 原子视图：原子级 mean/max/sum/attn 拼接 → atom_proj
      - 融合：
          seq = stack([frag, atom], dim=1)          # (B, 2, d)
          seq = seq + view_type_embed               # 区分两个视图
          attn_out, attn_w = mha(seq, seq, seq)     # 2-token 双向 self-attention
          seq = LayerNorm(seq + attn_out)           # 残差 1
          seq = LayerNorm(seq + ffn(seq))           # 残差 2（FFN）
          h_frag, h_atom = seq[:, 0], seq[:, 1]
          mol_repr = frag + alpha * h_frag + (1 - alpha) * h_atom    # 加权融合
      - alpha = sigmoid(raw_alpha) * alpha_max，L2 稀疏惩罚，初始偏片段

    输出 (B, d)，不改骨架。
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gin', n_heads=4, ffn_ratio=2,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        self.n_heads = n_heads
        self.in_channels = in_channels

        # ==================== 片段视图（主干候选 1）====================
        self.frag_pool = BRICSFragmentPoolOriginal(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        # ==================== 原子视图（主干候选 2）====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 视图类型 embedding ====================
        # 让 attention 能区分"我是片段还是原子"
        self.view_type_embed = nn.Parameter(torch.zeros(2, in_channels))

        # ==================== 视图间 Cross-Attention ====================
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        # FFN：d → ffn_ratio*d → d
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

        # ==================== 融合权重 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 缓存（分析用）
        self.last_alpha = None
        self.last_attn = None
        self.last_atom_weight = None
        self.last_frag_weight = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomCrossAttnFusionPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 片段视图 ====================
        mol_repr_frag, pool_loss = self.frag_pool(
            x, edge_index, edge_attr, batch, true_frag_ids=true_frag_ids
        )
        # 片段视图的 attention 缓存（可解释性）
        self.last_frag_weight = self.frag_pool.last_frag_weight

        # ==================== 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 2-token 序列 + view-type embedding ====================
        # (B, 2, d)
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)
        seq = seq + self.view_type_embed.unsqueeze(0)     # (1, 2, d) → 广播

        # ==================== Cross-Attention ====================
        # 2-token 双向 self-attention
        attn_out, attn_w = self.mha(seq, seq, seq)        # attn_w: (B, 2, 2)
        seq = self.norm1(seq + attn_out)

        # ==================== FFN ====================
        seq = self.norm2(seq + self.ffn(seq))

        # ==================== 拆开两个视图的更新表示 ====================
        h_frag = seq[:, 0, :]     # (B, d)
        h_atom = seq[:, 1, :]     # (B, d)

        # ==================== 加权融合（带稀疏残差）====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== 正则 + 缓存 ====================
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2

        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        self.frag_pool.set_epoch(epoch)

              
class ManualMultiheadAttention(nn.Module):
    """
    手写 2-token Multi-Head Attention。

    参考 PyTorch 1.x 的 multi_head_attention_forward 手动展开：
        Q/K/V = fused in_proj(x)        # 一次 Linear 出 3*d
        reshape → (B, H, L, head_dim)
        q *= scaling                     # 1/sqrt(head_dim)
        attn = q @ k^T                   # (B, H, L, L)
        attn = softmax(attn, dim=-1)
        attn = dropout(attn)
        out = attn @ v                   # (B, H, L, head_dim)
        merge heads → (B, L, d)
        out = out_proj(out)

    与 nn.MultiheadAttention 的差异：
      - 只支持 L == 2（这正是 2-token cross-attention 的场景）
      - 不接受 attn_mask / key_padding_mask（当前用法不需要）
      - 其余行为（初始化、need_weights 返回多头平均）保持一致

    初始化跟 nn.MultiheadAttention 完全一致：
      - in_proj_weight: xavier_uniform_
      - in_proj_bias / out_proj.bias: 0
      - out_proj.weight: nn.Linear 默认 kaiming_uniform_
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim({embed_dim}) 必须能被 num_heads({num_heads}) 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # 与老版一致：一个 fused in_proj 出 Q/K/V
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # 与 nn.MultiheadAttention._reset_parameters 对齐
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, need_weights=True, **kwargs):
        """
        Args:
            query, key, value: (B, 2, d)  本实现只支持 L=2 的 self-attention
            need_weights:     是否返回 attention 权重
        Returns:
            out:   (B, 2, d)
            attn_w:(B, 2, 2) 多头平均后的权重；need_weights=False 时返回 None
        """
        B, L, D = query.shape
        assert L == 2, f"ManualMultiheadAttention 专为 2-token 场景，L={L} 不支持"
        assert D == self.embed_dim, f"输入维度 {D} != embed_dim {self.embed_dim}"
        # 本类只做 self-attention，用同一张量即可
        assert query.shape == key.shape == value.shape

        # ---------- 1. QKV 线性投影（fused in_proj 拆三段） ----------
        if self.in_proj_bias is not None:
            q = F.linear(query, self.in_proj_weight[:D],      self.in_proj_bias[:D])
            k = F.linear(key,   self.in_proj_weight[D:2*D],   self.in_proj_bias[D:2*D])
            v = F.linear(value, self.in_proj_weight[2*D:],    self.in_proj_bias[2*D:])
        else:
            q = F.linear(query, self.in_proj_weight[:D])
            k = F.linear(key,   self.in_proj_weight[D:2*D])
            v = F.linear(value, self.in_proj_weight[2*D:])

        # ---------- 2. reshape 成多头 (B, H, L, head_dim) ----------
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- 3. scale + bmm ----------
        q = q * self.scaling
        attn = torch.matmul(q, k.transpose(-2, -1))       # (B, H, L, L)

        # ---------- 4. softmax + dropout（只作用在 attn 上，跟老版一致） ----------
        attn = F.softmax(attn, dim=-1)
        attn_for_v = self.dropout(attn)

        # ---------- 5. attn @ v → 合并多头 ----------
        out = torch.matmul(attn_for_v, v)                 # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # ---------- 6. 输出投影 ----------
        out = self.out_proj(out)

        # ---------- 7. attention 权重（多头平均，与 nn.MultiheadAttention 默认行为一致） ----------
        if need_weights:
            attn_w = attn.mean(dim=1)                     # (B, L, L)
        else:
            attn_w = None

        return out, attn_w
            
            
class BRICSAtomCrossAttnFusionV2Pool(BRICSFragmentPoolOriginal):
    """
    cross_attn_fusion_34 的保守改进版。

    与 v34 的区别（三处）：
      1. view_type_embed 固定正交初始化
      2. n_heads=1（2-token 不需要多头）
      3. h_atom 加残差（保留原始原子视图信息）

    融合公式完全不变：
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gin', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        self.in_channels = in_channels
        self.de_corr_weight = de_corr_weight

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ==================== 原子视图 ====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== ★ 改动1：固定正交 view_type_embed ====================
        self.register_buffer(
            'view_type_embed',
            torch.stack([
                torch.ones(in_channels),
                -torch.ones(in_channels)
            ]) * 0.1
        )
        
        # ==================== ★ 改动2：n_heads=1 ====================
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=n_heads,      # ★ 默认 1
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

        # ==================== 融合权重 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 缓存
        self.last_alpha = None
        self.last_attn = None
        self.last_atom_weight = None
        self.last_frag_weight = None

    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """BRICS 片段池化（复用父类组件）。"""
        device = x.device
        B = int(batch.max().item()) + 1

        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        attn_score = self.mol_attn(h_group).squeeze(-1)
        frag_attn_weight = softmax(attn_score, frag_batch)
        mol_repr_frag = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        self.last_frag_weight = frag_attn_weight.detach()
        return mol_repr_frag, pool_loss

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图 ====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. 2-token cross-attention ====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)   # (B, 2, d)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        # ==================== ★ 改动3：h_atom 加残差 ====================
        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :] + mol_repr_atom   # ★ 加残差

        # ==================== 4. 融合（主公式不变）====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== 5. 正则 ====================
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + self.de_corr_weight * sim ** 2

        # ==================== 6. 缓存 ====================
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass    

class BRICSAtomAlignAttnPool(BRICSFragmentPoolOriginal):
    """
    BRICS 片段视图 + 原子视图：
    - 片段视图：BRICSFragmentPoolOriginal → mol_repr_frag, p_frag
    - 原子视图：原子 attention → p_atom, mol_repr_atom
    - 对齐：p_atom 按片段聚合 → p_atom_frag，与 p_frag 对齐
    - cross-attention 融合 → fused
    - 门控：mol_repr = g * mol_repr_frag + (1-g) * fused
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=1, ffn_ratio=1,
                 align_weight=0.05, gate_init=-1.0):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        self.in_channels = in_channels
        self.align_weight = align_weight

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ==================== 原子视图 ====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 2-token cross-attention ====================
        self.register_buffer(
            'view_type_embed',
            torch.stack([torch.ones(in_channels),
                         -torch.ones(in_channels)]) * 0.1
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

        # ==================== 融合权重 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
        self.alpha_max = 1.0

        # ==================== 门控：纯 BRICS vs 融合 ====================
        self.final_gate = nn.Sequential(
            nn.Linear(2 * in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        nn.init.constant_(self.final_gate[-1].bias, float(gate_init))

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # 缓存
        self.last_p_frag = None
        self.last_p_atom_frag = None
        self.last_align_loss = None
        self.last_g = None
        self.last_alpha = None
        self.last_attn = None

    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """片段池化，返回 mol_repr_frag, p_frag, frag_idx, pool_loss"""
        device = x.device
        B = int(batch.max().item()) + 1

        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # 片段内聚合
        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        # 片段间 GNN
        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        # 分子级聚合
        attn_score = self.mol_attn(h_group).squeeze(-1)
        p_frag = softmax(attn_score, frag_batch)   # (num_frags,) 片段注意力
        mol_repr_frag = scatter(p_frag.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        return mol_repr_frag, p_frag, frag_idx, frag_batch, num_frags, pool_loss

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图 ====================
        mol_repr_frag, p_frag, frag_idx, frag_batch, num_frags, pool_loss = \
            self._fragment_pooling(x, edge_index, edge_attr, batch, true_frag_ids)

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        p_atom = softmax(atom_logits, batch)   # (N,) 原子注意力

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(p_atom.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. ★ 注意力对齐 ====================
        # 把原子注意力按片段聚合
        p_atom_frag = scatter(p_atom, frag_idx, dim=0,
                              dim_size=num_frags, reduce='sum')  # (num_frags,)

        # 两者都是片段上的概率分布，用 KL 散度对齐
        # 加 eps 防止 log(0)
        eps = 1e-8
        p_frag_norm = p_frag / (p_frag.sum() + eps)
        p_atom_frag_norm = p_atom_frag / (p_atom_frag.sum() + eps)

        # 双向 KL（对称化）
        kl_1 = F.kl_div((p_frag_norm + eps).log(),
                        p_atom_frag_norm + eps, reduction='batchmean')
        kl_2 = F.kl_div((p_atom_frag_norm + eps).log(),
                        p_frag_norm + eps, reduction='batchmean')
        loss_align = (kl_1 + kl_2) / 2.0

        # ==================== 4. cross-attention 融合 ====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)
        seq = seq + self.view_type_embed.unsqueeze(0)
        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :]

        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        fused = alpha * h_frag + (1 - alpha) * h_atom

        # ==================== 5. 门控：纯 BRICS vs 融合 ====================
        g = torch.sigmoid(self.final_gate(
            torch.cat([mol_repr_frag, fused], dim=-1)
        ))
        mol_repr = g * mol_repr_frag + (1 - g) * fused

        # ==================== 6. 总损失 ====================
        pool_loss = pool_loss + self.align_weight * loss_align

        # ==================== 7. 缓存 ====================
        self.last_p_frag = p_frag.detach()
        self.last_p_atom_frag = p_atom_frag.detach()
        self.last_align_loss = loss_align.detach()
        self.last_g = g.detach()
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass


    
# ==================== 有符号注意力 ====================
class SignedAttentionScorer(nn.Module):
    """
    有符号注意力：输出可正可负的权重。

    设计：
      - 正通道：MLP → softmax → pos_weight ∈ [0,1]
      - 负通道：MLP → softmax → neg_weight ∈ [0,1]
      - signed_weight = pos_weight - neg_weight ∈ [-1, 1]
      - 每个分子内 sum(signed_weight) 不一定为 1（因为正负可抵消）

    用途：
      - 如果片段倾向于支持预测 → signed_weight > 0
      - 如果片段倾向于反对预测 → signed_weight < 0
      - 这使得片段注意力本身具有"方向性"，可直接作为正负贡献的候选
    """
    def __init__(self, in_channels, hidden=None, dropout=0.0):
        super().__init__()
        if hidden is None:
            hidden = max(in_channels // 2, 8)

        self.pos_scorer = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.neg_scorer = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        # 可学习温度，控制 softmax 尖锐程度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, batch):
        """
        Args:
            h: (N, d) 节点/片段表示
            batch: (N,) 每个节点属于哪个图
        Returns:
            signed_w: (N,) 有符号权重
            pos_w:    (N,) 正通道权重（总和为每个图内为1）
            neg_w:    (N,) 负通道权重（总和为每个图内为1）
        """
        temp = self.temperature.abs().clamp(min=1e-3)
        pos_logits = self.pos_scorer(h).squeeze(-1) / temp
        neg_logits = self.neg_scorer(h).squeeze(-1) / temp

        pos_w = softmax(pos_logits, batch)
        neg_w = softmax(neg_logits, batch)

        signed_w = pos_w - neg_w
        return signed_w, pos_w, neg_w


# ==================== 基于有符号注意力的 34 版本 ====================
class BRICSAtomCrossAttnFusionSignedPool(BRICSFragmentPoolOriginal):
    """
    brics_atom_cross_attn_fusion_34 的有符号注意力版本。

    与 v34 唯一的区别：
      - 片段级聚合（mol_attn）改为有符号注意力
      - 原来：softmax(score) ∈ [0,1]，权重全为正
      - 现在：pos_softmax - neg_softmax ∈ [-1,1]，权重可正可负

    其余部分完全一致（cross-attention 融合、原子视图、alpha 融合）。
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=4, ffn_ratio=2,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 signed_attn_temperature_init=1.0):
        # 父类初始化时，aggr='attention' 会创建原始 mol_attn
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',  # 暂时保留，后面会被覆盖使用
            gate_hidden=gate_hidden,
            model_type=model_type
        )

        self.in_channels = in_channels

        if gate_hidden is None:
            gate_hidden = max(in_channels // 2, 8)

        # ==================== 用有符号注意力替换 mol_attn ====================
        # 丢弃原有的 mol_attn（不影响训练，不放进 optimizer）
        self.mol_attn = None
        self.signed_mol_attn = SignedAttentionScorer(
            in_channels,
            hidden=gate_hidden,
            dropout=dropout
        )
        with torch.no_grad():
            self.signed_mol_attn.temperature.fill_(signed_attn_temperature_init)

        # ==================== 原子视图 ====================
        self.atom_attn = nn.Sequential(
            nn.Linear(in_channels, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1)
        )
        self.atom_proj = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==================== 2-token cross-attention ====================
        self.view_type_embed = nn.Parameter(torch.zeros(2, in_channels))
        with torch.no_grad():
            self.view_type_embed[0] = 1.0
            self.view_type_embed[1] = -1.0

        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

        # ==================== 融合权重 ====================
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_max = alpha_max
        self.alpha_reg = alpha_reg

        self.out_dim = in_channels
        self.needs_frag_ids = True
        self.needs_rot_frag_ids = False
        self.needs_env_ids = False
        self.FRAG_OFFSET = 10000

        # ==================== 缓存 ====================
        self.last_alpha = None
        self.last_attn = None
        self.last_atom_weight = None
        # 有符号注意力的缓存
        self.last_signed_frag_weight = None   # (num_frags,) 有符号
        self.last_pos_frag_weight = None      # (num_frags,) 正通道
        self.last_neg_frag_weight = None      # (num_frags,) 负通道

    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """BRICS 片段池化，但使用有符号注意力聚合。"""
        device = x.device
        B = int(batch.max().item()) + 1

        # 片段 ID
        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # ==================== 片段内聚合（原子 → 片段）====================
        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        # ==================== 片段间 GNN ====================
        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        # ==================== 分子级聚合：有符号注意力 ====================
        signed_w, pos_w, neg_w = self.signed_mol_attn(h_group, frag_batch)

        # 用有符号权重加权求和（可能有正有负）
        mol_repr_frag = scatter(signed_w.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        # 辅助损失：门控熵
        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        # 缓存
        self.last_signed_frag_weight = signed_w.detach()
        self.last_pos_frag_weight = pos_w.detach()
        self.last_neg_frag_weight = neg_w.detach()
        self.last_frag_weight = signed_w.detach()  # 兼容原有属性名

        return mol_repr_frag, pool_loss

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图（有符号注意力）====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. 2-token cross-attention ====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)   # (B, 2, d)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :]

        # ==================== 4. 融合 ====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== 5. 正则 ====================
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + 0.01 * sim ** 2

        # ==================== 6. 缓存 ====================
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass

    # ==================== 可解释性辅助方法 ====================
    def get_signed_frag_weights(self):
        """返回片段的有符号权重、正权重、负权重。"""
        return {
            'signed': self.last_signed_frag_weight,
            'pos': self.last_pos_frag_weight,
            'neg': self.last_neg_frag_weight,
        }
    
class BRICSAtomCrossAttnFusionV2AlignPool(BRICSAtomCrossAttnFusionV2Pool):
    """
    v2 + align 结合：
      - 主路径：完整继承 v2 的 cross-attention + alpha 融合
      - 辅助正则：片段注意力 vs 原子重要性聚合的对齐损失

    关键区别（相对 brics_atom_align）：
      - 最终表示来自片段+原子融合，不是只用片段
      - 对齐损失是辅助项，权重小，不主导训练
      - 对齐损失的片段分支保留梯度（原版可能两边都 detach）

    可解释性增益：
      - last_align_loss 可直接监控
      - 训练后片段注意力 = 原子重要性聚合，两者可交叉验证
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05, align_weight=0.05):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=gate_hidden,
            model_type=model_type,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            alpha_reg=alpha_reg,
            de_corr_weight=de_corr_weight,
        )
        self.align_weight = align_weight

        # 缓存带梯度的中间量（用于对齐损失）
        self._frag_attn_weight = None
        self._frag_idx = None
        self._frag_batch = None
        self._num_frags = None
        self.last_align_loss = None

    # ==================== 重写：保留带梯度的片段注意力 ====================
    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """与父类一致，但把 frag_attn_weight / frag_idx 等保留下来。"""
        device = x.device
        B = int(batch.max().item()) + 1

        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        attn_score = self.mol_attn(h_group).squeeze(-1)
        frag_attn_weight = softmax(attn_score, frag_batch)
        mol_repr_frag = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        # ★ 缓存：带梯度版本用于对齐，detach 版本兼容旧接口
        self._frag_attn_weight = frag_attn_weight
        self._frag_idx = frag_idx
        self._frag_batch = frag_batch
        self._num_frags = num_frags
        self.last_frag_weight = frag_attn_weight.detach()

        return mol_repr_frag, pool_loss

    # ==================== 主 forward ====================
    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomCrossAttnFusionV2AlignPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图 ====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. cross-attention（v2 原样）====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :] + mol_repr_atom   # v2 的残差

        # ==================== 4. 融合（v2 原样）====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== ★ 5. 对齐损失（新增）====================
        # 原子重要性聚合到片段级
        atom_imp_frag = scatter(atom_weights, self._frag_idx, dim=0,
                                dim_size=self._num_frags, reduce='sum')
        # 片段注意力作为目标（保留梯度，但用 detach 版本作为对齐目标）
        align_loss = F.mse_loss(self._frag_attn_weight,
                                atom_imp_frag.detach())
        pool_loss = pool_loss + self.align_weight * align_loss
        self.last_align_loss = align_loss.detach()

        # ==================== 6. v2 原有正则 ====================
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + self.de_corr_weight * sim ** 2

        # ==================== 7. 缓存 ====================
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    # ==================== 可解释性辅助方法 ====================
    def get_align_stats(self):
        """返回对齐损失 + 片段注意力 + 原子重要性聚合，供分析用。"""
        if self._frag_attn_weight is None:
            return None
        return {
            'align_loss':    float(self.last_align_loss.item())
                             if self.last_align_loss is not None else None,
            'p_frag':        self._frag_attn_weight.detach().cpu().numpy(),
            'atom_imp':      self.last_atom_weight,   # (N,) 已在 forward 缓存
            'frag_idx':      self._frag_idx.cpu().numpy(),
            'num_frags':     self._num_frags,
        }
    
    
class BRICSAtomCrossAttnFusionV2AblationPool(BRICSAtomCrossAttnFusionV2Pool):
    """
    v2_align 的分支消融版本。

    branch_mode 控制用哪些分支：
      - 'frag_only'      : 只用片段分支。返回 mol_repr_frag
      - 'atom_only'      : 只用原子分支。返回 mol_repr_atom
      - 'fused_no_align' : 片段+原子融合，但**不加对齐损失**（= v2）
      - 'concat'         : ★ 新增，直接 concat 片段+原子分支，过一层投影

    继承自 BRICSAtomCrossAttnFusionV2Pool：
      - 无对齐损失（父类本来就没有）
      - 保留完整的片段/原子/cross-attn/alpha 融合路径

    可解释性缓存（继承自父类）：
      - last_atom_weight : (N,)      原子级 softmax 注意力
      - last_frag_weight : (K,)      片段级 softmax 注意力
      - last_alpha       : 标量      融合权重
      - last_attn        : (B,2,2)   2-token attention 权重
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gin', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05,
                 branch_mode='fused_no_align'):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=gate_hidden,
            model_type=model_type,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            alpha_reg=alpha_reg,
            de_corr_weight=de_corr_weight,
        )

        assert branch_mode in ('frag_only', 'atom_only',
                               'fused_no_align', 'concat'), \
            f"Unknown branch_mode: {branch_mode}"
        self.branch_mode = branch_mode

        # ★ 新增：concat 分支的降维投影层
        if self.branch_mode == 'concat':
            self.concat_proj = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels)
            )
        else:
            self.concat_proj = None

    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 2. 分支消融：atom_only ====================
        if self.branch_mode == 'atom_only':
            self.last_atom_weight = atom_weights.detach()
            self.last_alpha = None
            self.last_attn = None
            self.last_frag_weight = None
            return mol_repr_atom, torch.tensor(0.0, device=device)

        # ==================== 3. 片段视图 ====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        if self.branch_mode == 'frag_only':
            self.last_atom_weight = atom_weights.detach()
            self.last_alpha = None
            self.last_attn = None
            return mol_repr_frag, pool_loss

        # ==================== 4. ★ 新增：直接 concat ====================
        if self.branch_mode == 'concat':
            mol_repr = self.concat_proj(
                torch.cat([mol_repr_frag, mol_repr_atom], dim=-1)
            )
            self.last_atom_weight = atom_weights.detach()
            self.last_alpha = None
            self.last_attn = None
            return mol_repr, pool_loss

        # ==================== 5. fused_no_align：cross-attn 融合，无对齐 ====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :] + mol_repr_atom   # v2 的原子残差

        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # v2 原有正则（无对齐损失）
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + self.de_corr_weight * sim ** 2

        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    def set_epoch(self, epoch):
        pass    


class BRICSAtomCrossAttnFusionV2AlignNoFragGNNPool(BRICSAtomCrossAttnFusionV2AlignPool):
    """
    与 BRICSAtomCrossAttnFusionV2AlignPool 完全一致，唯一区别：
    ★ 删掉片段间 GNN（self.frag_convs）★

    改动动机（sanity check）：
      - BRICS 切割边极稀疏，1 层 GNN 感受野只有 1~2 跳
      - 下游 mol_attn 已经做了全片段全局聚合
      - 当前已过拟合，删掉 GNN 若能不掉性能 → 直接省参数

    保留的部分（一字未改）：
      - 片段内聚合：原子 attention + 门控 sum/mean/max
      - 片段级聚合：mol_attn（打分 + 加权和）
      - 原子视图：mean / max / sum / attn 四统计量
      - 2-token cross-attention 融合 + h_atom 残差
      - alpha 融合 + 对齐损失
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05, align_weight=0.05):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=gate_hidden,
            model_type=model_type,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            alpha_reg=alpha_reg,
            de_corr_weight=de_corr_weight,
            align_weight=align_weight,
        )

        # ★ 清空片段间 GNN（父类构造时已建好，这里替换为空，
        #   既省参数也不会被 optimizer / state_dict 带上）
        self.frag_convs = nn.ModuleList()
        self.frag_norms = None

    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """
        与父类 _fragment_pooling 完全一致，
        只是删掉了 [frag_edge_index 构建 + frag GNN 前向传播] 那一段。
        """
        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段 ID ====================
        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        # ==================== 2. 片段内聚合（同父类）====================
        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        # ==================== ★★★ 片段间 GNN：整段删除 ★★★ ====================
        # 父类这里原本有：
        #   u, v = edge_index[0], edge_index[1]
        #   frag_u = frag_idx[u];  frag_v = frag_idx[v]
        #   cross_mask = (frag_u != frag_v)
        #   frag_edge_index = torch.stack([frag_u[cross_mask], frag_v[cross_mask]], dim=0)
        #   frag_edge_attr  = edge_attr[cross_mask]
        #   for i, conv in enumerate(self.frag_convs):
        #       h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
        #                                  frag_edge_attr, self.model_type)
        #       h_group = F.relu(h_group)
        #       if self.use_layernorm:
        #           h_group = self.frag_norms[i](h_group)
        #       h_group = self.dropout(h_group)
        #
        # 全部删掉。h_group 直接进入片段级聚合。

        # ==================== 3. 片段级聚合（同父类）====================
        attn_score = self.mol_attn(h_group).squeeze(-1)
        frag_attn_weight = softmax(attn_score, frag_batch)
        mol_repr_frag = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)) \
                       .sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        # ==================== 4. 缓存带梯度版本用于对齐损失 ====================
        self._frag_attn_weight = frag_attn_weight
        self._frag_idx = frag_idx
        self._frag_batch = frag_batch
        self._num_frags = num_frags
        self.last_frag_weight = frag_attn_weight.detach()

        return mol_repr_frag, pool_loss
    
    

class BRICSAtomCrossAttnFusionV2AlignPool(BRICSAtomCrossAttnFusionV2Pool):
    """
    v2 + align 结合：
      - 主路径：完整继承 v2 的 cross-attention + alpha 融合
      - 辅助正则：片段注意力 vs 原子重要性聚合的对齐损失

    关键区别（相对 brics_atom_align）：
      - 最终表示来自片段+原子融合，不是只用片段
      - 对齐损失是辅助项，权重小，不主导训练
      - 对齐损失的片段分支保留梯度（原版可能两边都 detach）

    可解释性增益：
      - last_align_loss 可直接监控
      - 训练后片段注意力 = 原子重要性聚合，两者可交叉验证
    """
    def __init__(self, in_channels, hidden_channels=None, num_gnn_layers=1,
                 dropout=0.0, use_layernorm=True, gate_hidden=None,
                 model_type='gine', n_heads=1, ffn_ratio=1,
                 alpha_init=-2.0, alpha_max=1.0, alpha_reg=0.01,
                 de_corr_weight=0.05, align_weight=0.05):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=gate_hidden,
            model_type=model_type,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            alpha_reg=alpha_reg,
            de_corr_weight=de_corr_weight,
        )
        self.align_weight = align_weight

        # 缓存带梯度的中间量（用于对齐损失）
        self._frag_attn_weight = None
        self._frag_idx = None
        self._frag_batch = None
        self._num_frags = None
        self.last_align_loss = None

    # ==================== 重写：保留带梯度的片段注意力 ====================
    def _fragment_pooling(self, x, edge_index, edge_attr, batch, true_frag_ids):
        """与父类一致，但把 frag_attn_weight / frag_idx 等保留下来。"""
        device = x.device
        B = int(batch.max().item()) + 1

        frag_global = batch * self.FRAG_OFFSET + true_frag_ids
        unique_frags, frag_idx = torch.unique(frag_global, return_inverse=True)
        num_frags = unique_frags.size(0)
        frag_batch = unique_frags // self.FRAG_OFFSET

        attn_logits = self.attn_mlp(x).squeeze(-1)
        attn_weights = softmax(attn_logits, frag_idx)
        group_feat = scatter(attn_weights.unsqueeze(-1) * x, frag_idx,
                             dim=0, dim_size=num_frags, reduce='sum')
        group_size = scatter(torch.ones_like(frag_idx, dtype=torch.float),
                             frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        ent = -(attn_weights * torch.log(attn_weights + 1e-8))
        group_entropy = scatter(ent, frag_idx, dim=0,
                                dim_size=num_frags, reduce='sum')

        gate_input = torch.cat([group_feat,
                                group_size.unsqueeze(-1),
                                group_entropy.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        h_sum = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='sum')
        h_mean = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='mean')
        h_max = scatter(x, frag_idx, dim=0, dim_size=num_frags, reduce='max')
        h_group = (gate_weights[:, 0:1] * h_sum +
                   gate_weights[:, 1:2] * h_mean +
                   gate_weights[:, 2:3] * h_max)

        u, v = edge_index[0], edge_index[1]
        frag_u = frag_idx[u]
        frag_v = frag_idx[v]
        cross_mask = (frag_u != frag_v)
        if cross_mask.sum() > 0:
            frag_edge_index = torch.stack([frag_u[cross_mask],
                                           frag_v[cross_mask]], dim=0)
            frag_edge_attr = edge_attr[cross_mask]
        else:
            frag_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            frag_edge_attr = torch.zeros((0, 10), dtype=torch.float, device=device)

        for i, conv in enumerate(self.frag_convs):
            h_group = _apply_frag_conv(conv, h_group, frag_edge_index,
                                       frag_edge_attr, self.model_type)
            h_group = F.relu(h_group)
            if self.use_layernorm:
                h_group = self.frag_norms[i](h_group)
            h_group = self.dropout(h_group)

        attn_score = self.mol_attn(h_group).squeeze(-1)
        frag_attn_weight = softmax(attn_score, frag_batch)
        mol_repr_frag = scatter(frag_attn_weight.unsqueeze(-1) * h_group,
                                frag_batch, dim=0, dim_size=B, reduce='sum')

        entropy_gate = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        pool_loss = 0.01 * entropy_gate

        # ★ 缓存：带梯度版本用于对齐，detach 版本兼容旧接口
        self._frag_attn_weight = frag_attn_weight
        self._frag_idx = frag_idx
        self._frag_batch = frag_batch
        self._num_frags = num_frags
        self.last_frag_weight = frag_attn_weight.detach()

        return mol_repr_frag, pool_loss

    # ==================== 主 forward ====================
    def forward(self, x, edge_index, edge_attr, batch,
                true_frag_ids=None, **kwargs):
        if true_frag_ids is None:
            raise ValueError("BRICSAtomCrossAttnFusionV2AlignPool 需要 true_frag_ids")

        device = x.device
        B = int(batch.max().item()) + 1

        # ==================== 1. 片段视图 ====================
        mol_repr_frag, pool_loss = self._fragment_pooling(
            x, edge_index, edge_attr, batch, true_frag_ids
        )

        # ==================== 2. 原子视图 ====================
        atom_logits = self.atom_attn(x).squeeze(-1)
        atom_weights = softmax(atom_logits, batch)

        a_mean = global_mean_pool(x, batch)
        a_max = global_max_pool(x, batch)
        a_sum = global_add_pool(x, batch)
        a_attn = scatter(atom_weights.unsqueeze(-1) * x, batch,
                         dim=0, dim_size=B, reduce='sum')

        mol_repr_atom = self.atom_proj(
            torch.cat([a_mean, a_max, a_sum, a_attn], dim=-1)
        )

        # ==================== 3. cross-attention（v2 原样）====================
        seq = torch.stack([mol_repr_frag, mol_repr_atom], dim=1)
        seq = seq + self.view_type_embed.unsqueeze(0)

        attn_out, attn_w = self.mha(seq, seq, seq)
        seq = seq + self.norm1(attn_out)
        seq = seq + self.norm2(self.ffn(seq))

        h_frag = seq[:, 0, :]
        h_atom = seq[:, 1, :] + mol_repr_atom   # v2 的残差

        # ==================== 4. 融合（v2 原样）====================
        alpha = torch.sigmoid(self.raw_alpha) * self.alpha_max
        mol_repr = mol_repr_frag + alpha * h_frag + (1 - alpha) * h_atom

        # ==================== ★ 5. 对齐损失（新增）====================
        # 原子重要性聚合到片段级
        atom_imp_frag = scatter(atom_weights, self._frag_idx, dim=0,
                                dim_size=self._num_frags, reduce='sum')
        # 片段注意力作为目标（保留梯度，但用 detach 版本作为对齐目标）
        align_loss = F.mse_loss(self._frag_attn_weight,
                                atom_imp_frag.detach())
        pool_loss = pool_loss + self.align_weight * align_loss
        self.last_align_loss = align_loss.detach()

        # ==================== 6. v2 原有正则 ====================
        sim = F.cosine_similarity(h_frag, h_atom, dim=-1).mean()
        pool_loss = pool_loss + self.alpha_reg * alpha ** 2 \
                  + self.de_corr_weight * sim ** 2

        # ==================== 7. 缓存 ====================
        self.last_alpha = alpha.detach()
        self.last_attn = attn_w.detach()
        self.last_atom_weight = atom_weights.detach()

        return mol_repr, pool_loss

    # ==================== 可解释性辅助方法 ====================
    def get_align_stats(self):
        """返回对齐损失 + 片段注意力 + 原子重要性聚合，供分析用。"""
        if self._frag_attn_weight is None:
            return None
        return {
            'align_loss':    float(self.last_align_loss.item())
                             if self.last_align_loss is not None else None,
            'p_frag':        self._frag_attn_weight.detach().cpu().numpy(),
            'atom_imp':      self.last_atom_weight,   # (N,) 已在 forward 缓存
            'frag_idx':      self._frag_idx.cpu().numpy(),
            'num_frags':     self._num_frags,
        } 

    
class GNN_Base_HierarchicalSAGTopK(nn.Module):
    """
    SAG/TopK 专用的层次化骨干：GNN → Pool → GNN → Pool → ... → 全局读出。

    与 GNN_Base_GINE/GIN/GCN/GAT/GATv2 的区别：
      - 那些是 [GNN × L] → PoolingHead → Dense → Out
      - 本类是 [GNN → Pool] × L → GlobalAdd → Dense → Out
      - SAG/TopK 需要每层 GNN 之后立即池化，所以不能塞进 pooling_module
    """
    def __init__(self, in_channels, out_channels, edge_dim,
                 pool_type='sag',          # 'sag' or 'topk'
                 ratio=0.7,
                 convs_layers=[128, 128, 128],
                 dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0,
                 backbone='gin', heads=4, **kwargs):
        super().__init__()
        self.pool_type = pool_type
        self.backbone = backbone.lower()
        self.ratio = ratio
        self.dropout_p = dropout_p
        self.edge_dim = edge_dim
        self._use_edge_attr = self.backbone in ('gine', 'gat', 'gatv2')

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(len(_convs) - 1):
            self.convs.append(self._build_conv(
                _convs[i], _convs[i + 1], edge_dim, dropout_p, heads))
            self.pools.append(self._build_pool(pool_type, _convs[i + 1], ratio))

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        _dense = [_convs[-1]] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def _build_conv(self, in_dim, out_dim, edge_dim, dropout, heads):
        if self.backbone == 'gine':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn=mlp, edge_dim=edge_dim)
        elif self.backbone == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(nn=mlp)
        elif self.backbone == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif self.backbone == 'gat':
            return GATConv(in_dim, out_dim, heads=heads, concat=False,
                           edge_dim=edge_dim, dropout=dropout)
        elif self.backbone == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    @staticmethod
    def _build_pool(pool_type, in_dim, ratio):
        if pool_type == 'sag':
            return PyG_SAGPooling(in_dim, ratio=ratio)
        elif pool_type == 'topk':
            return PyG_TopKPooling(in_dim, ratio=ratio)
        elif pool_type == 'asap':                        # ← 新增
            return PyG_ASAPooling(in_dim, ratio=ratio)
        raise ValueError(f"Unknown pool_type: {pool_type}")

    def _apply_conv(self, conv, x, edge_index, edge_attr):
        if self._use_edge_attr:
            return conv(x, edge_index, edge_attr)
        return conv(x, edge_index)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = self._apply_conv(conv, x, edge_index, edge_attr)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            # ★ 关键：每层卷积后立即池化
        # ★ 关键：每层卷积后立即池化
            if self.pool_type == 'asap':
                # ASAP 不接受 edge_attr
                x, edge_index,_, batch, _ = pool(x, edge_index, batch=batch)
                # 池化后 edge_attr 维度可能变了，重建占位
                edge_attr = torch.zeros(edge_index.size(1), self.edge_dim,
                                        device=x.device)
            else:
                # SAG / TopK 走原来的调用
                x, edge_index, edge_attr, batch, _, _ = pool(
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )

        y = global_add_pool(x, batch)
        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, torch.tensor(0.0, device=x.device)

    def set_epoch(self, epoch):
        pass        
        
# ==================== 骨架 ====================
class GNN_Base_GINE(nn.Module):
    """GINE 骨架（默认）"""
    def __init__(self, in_channels, out_channels, edge_dim,
                 pooling_module, convs_layers=[128, 128, 128], dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0, **kwargs):
        super().__init__()
        self.pooling = pooling_module
        self.dropout_p = dropout_p

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        for i in range(len(_convs) - 1):
            mlp = nn.Sequential(
                nn.Linear(_convs[i], _convs[i + 1]),
                nn.ReLU(),
                nn.Linear(_convs[i + 1], _convs[i + 1])
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=edge_dim, **kwargs))

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        pool_out_dim = pooling_module.out_dim if hasattr(pooling_module, 'out_dim') else _convs[-1]
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids
        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch,**extra_kwargs)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    def set_epoch(self, epoch):
        if hasattr(self.pooling, 'set_epoch'):
            self.pooling.set_epoch(epoch)


class GNN_Base_GIN(nn.Module):
    """GIN 骨架（不使用 edge_attr）"""
    def __init__(self, in_channels, out_channels, edge_dim,
                 pooling_module, convs_layers=[128, 128, 128], dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0, **kwargs):
        super().__init__()
        self.pooling = pooling_module
        self.dropout_p = dropout_p

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        for i in range(len(_convs) - 1):
            mlp = nn.Sequential(
                nn.Linear(_convs[i], _convs[i + 1]),
                nn.ReLU(),
                nn.Linear(_convs[i + 1], _convs[i + 1])
            )
            self.convs.append(GINConv(nn=mlp))

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        pool_out_dim = pooling_module.out_dim if hasattr(pooling_module, 'out_dim') else _convs[-1]
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids
        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch,**extra_kwargs)
        #y, pool_loss = self.pooling(x, edge_index, edge_attr, batch)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    def set_epoch(self, epoch):
        if hasattr(self.pooling, 'set_epoch'):
            self.pooling.set_epoch(epoch)


class GNN_Base_GCN(nn.Module):
    """GCN 骨架（不使用 edge_attr）"""
    def __init__(self, in_channels, out_channels, edge_dim,
                 pooling_module, convs_layers=[128, 128, 128], dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0, **kwargs):
        super().__init__()
        self.pooling = pooling_module
        self.dropout_p = dropout_p

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        for i in range(len(_convs) - 1):
            self.convs.append(GCNConv(_convs[i], _convs[i + 1]))

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        pool_out_dim = pooling_module.out_dim if hasattr(pooling_module, 'out_dim') else _convs[-1]
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids
        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch,**extra_kwargs)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    def set_epoch(self, epoch):
        if hasattr(self.pooling, 'set_epoch'):
            self.pooling.set_epoch(epoch)


class GNN_Base_GAT(nn.Module):
    """GAT 骨架（使用 edge_attr）"""
    def __init__(self, in_channels, out_channels, edge_dim,
                 pooling_module, convs_layers=[128, 128, 128], dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0, heads=4, **kwargs):
        super().__init__()
        self.pooling = pooling_module
        self.dropout_p = dropout_p
        self.heads = heads

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        for i in range(len(_convs) - 1):
            self.convs.append(
                GATConv(
                    _convs[i], _convs[i + 1],
                    heads=heads, concat=False,
                    edge_dim=edge_dim,
                    dropout=dropout_p
                )
            )

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        pool_out_dim = pooling_module.out_dim if hasattr(pooling_module, 'out_dim') else _convs[-1]
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids
        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch,**extra_kwargs)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    def set_epoch(self, epoch):
        if hasattr(self.pooling, 'set_epoch'):
            self.pooling.set_epoch(epoch)

class GNN_Base_GATv2(nn.Module):
    """
    GATv2 骨架（使用 edge_attr）
    - 动态注意力：注意力分数依赖查询节点，表达力严格强于 GAT
    - 与 GAT 唯一的区别是 GATConv → GATv2Conv，其余结构完全一致
    """
    def __init__(self, in_channels, out_channels, edge_dim,
                 pooling_module, convs_layers=[128, 128, 128], dense_layers=[128, 64],
                 batch_norms=None, dropout_p=0.0, heads=4, **kwargs):
        super().__init__()
        self.pooling = pooling_module
        self.dropout_p = dropout_p
        self.heads = heads

        _convs = [in_channels] + convs_layers
        self.convs = nn.ModuleList()
        for i in range(len(_convs) - 1):
            self.convs.append(
                GATv2Conv(
                    _convs[i], _convs[i + 1],
                    heads=heads, concat=False,
                    edge_dim=edge_dim,
                    dropout=dropout_p
                )
            )

        if batch_norms is not None:
            self.batch_norms = nn.ModuleList([
                batch_norms(_convs[i + 1]) for i in range(len(_convs) - 1)
            ])
        else:
            self.batch_norms = None

        pool_out_dim = pooling_module.out_dim if hasattr(pooling_module, 'out_dim') else _convs[-1]
        _dense = [pool_out_dim] + dense_layers
        self.lins = nn.ModuleList([
            nn.Linear(_dense[i], _dense[i + 1]) for i in range(len(_dense) - 1)
        ])
        self.out = nn.Linear(_dense[-1], out_channels)

    def forward(self, x, edge_index, edge_attr, batch, data_batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids

        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch, **extra_kwargs)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    def set_epoch(self, epoch):
        if hasattr(self.pooling, 'set_epoch'):
            self.pooling.set_epoch(epoch)            

def build_model(backbone_name, in_dim, out_ch, edge_dim, pooling,
                convs, dense_layers, dropout, norm):
    """根据 backbone_name 构建骨架 + Sum 池化。"""
    backbone_name = backbone_name.lower()
    if backbone_name == "gine":
        return GNN_Base_GINE(in_dim, out_ch, edge_dim=edge_dim, pooling_module=pooling,
                             convs_layers=convs, dense_layers=dense_layers,
                             dropout_p=dropout, batch_norms=norm)
    elif backbone_name == "gin":
        return GNN_Base_GIN(in_dim, out_ch, edge_dim=edge_dim, pooling_module=pooling,
                            convs_layers=convs, dense_layers=dense_layers,
                            dropout_p=dropout, batch_norms=norm)
    elif backbone_name == "gcn":
        return GNN_Base_GCN(in_dim, out_ch, edge_dim=edge_dim, pooling_module=pooling,
                            convs_layers=convs, dense_layers=dense_layers,
                            dropout_p=dropout, batch_norms=norm)
    elif backbone_name == "gat":
        return GNN_Base_GAT(in_dim, out_ch, edge_dim=edge_dim, pooling_module=pooling,
                            convs_layers=convs, dense_layers=dense_layers,
                            dropout_p=dropout, batch_norms=norm, heads=4)
    elif backbone_name == "gatv2":
        return GNN_Base_GATv2(in_dim, out_ch, edge_dim=edge_dim, pooling_module=pooling,
                              convs_layers=convs, dense_layers=dense_layers,
                              dropout_p=dropout, batch_norms=norm, heads=4)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
        
        
def _build_frag_conv(model_type, in_channels, hidden_channels, dropout=0.0):
    """根据 model_type 构建片段间 GNN 层"""
    model_type = model_type.lower()
    if model_type == "gine":
        return GINEConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels)
            ),
            edge_dim=10
        )
    elif model_type == "gin":
        return GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels)
            )
        )
    elif model_type == "gcn":
        return GCNConv(in_channels, in_channels)
    elif model_type == "gat":
        return GATConv(
            in_channels, in_channels,
            heads=4, concat=False, edge_dim=10, dropout=dropout
        )
    elif model_type == "gatv2":
        return GATv2Conv(
            in_channels, in_channels,
            heads=4, concat=False, edge_dim=10, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _apply_frag_conv(conv, h, edge_index, edge_attr, model_type):
    """根据 model_type 决定是否传 edge_attr"""
    model_type = model_type.lower()
    if model_type in ("gine", "gat", "gatv2"):
        return conv(h, edge_index, edge_attr)
    else:  # gin, gcn
        return conv(h, edge_index)        


    
    
# ==================== 训练与评估 ====================
def train_epoch(model, loader, optimizer, criterion, task_type, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred, pool_loss = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, data_batch=batch)
        if task_type == "multitask":
            y_true = batch.y.float()
            mask = ~torch.isnan(y_true)
            if mask.sum() == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                n = 0
            else:
                loss = criterion(pred[mask], y_true[mask]).mean()
                n = mask.sum().item()
        elif task_type in ["classification", "multiclass"]:
            loss = criterion(pred, batch.y)
            n = batch.num_graphs
        else:
            loss = criterion(pred.view(-1), batch.y.view(-1))
            n = batch.num_graphs

        if isinstance(pool_loss, torch.Tensor) and torch.isfinite(pool_loss):
            loss = loss + pool_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * n
        total_samples += n
    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate(model, loader, task_type, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, data_batch=batch)
            if task_type == "classification":
                preds.extend(F.softmax(pred, 1)[:, 1].cpu().tolist())
            elif task_type == "multiclass":
                preds.append(F.softmax(pred, 1).cpu().numpy())
            elif task_type == "multitask":
                preds.append(torch.sigmoid(pred).cpu().numpy())
            else:
                preds.extend(pred.view(-1).cpu().tolist())
            targets.extend(batch.y.cpu().tolist())
    if task_type == "classification":
        if len(set(targets)) < 2:
            return float("nan")
        return roc_auc_score(targets, preds)
    elif task_type == "multiclass":
        preds = np.concatenate(preds)
        targets = np.array(targets)
        if len(np.unique(targets)) < 2:
            return float("nan")
        return roc_auc_score(targets, preds, multi_class="ovr", average="macro")
    elif task_type == "multitask":
        preds = np.concatenate(preds, axis=0)
        targets = np.array(targets, dtype=np.float32)
        aucs = []
        for i in range(targets.shape[1]):
            mask = ~np.isnan(targets[:, i])
            if mask.sum() < 2:
                continue
            yt = targets[mask, i]
            yp = preds[mask, i]
            if not np.isfinite(yp).all():
                raise ValueError(
                    f"Task {i} predictions contain NaN or Inf. "
                    f"Invalid count: {(~np.isfinite(yp)).sum()}/{len(yp)}"
                )
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, yp))
        if not aucs:
            return float("nan")
        return np.mean(aucs)
    else:
        return np.sqrt(mean_squared_error(targets, preds))


def _patch_gcn_forward_for_edge_weight(model):
    """让 GNN_Base_GCN.forward 接受 edge_weight（IG 要对 edge_weight 求梯度必须）"""
    def new_forward(self, x, edge_index, edge_attr, batch,
                    data_batch=None, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)

            x = F.relu(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

        extra_kwargs = {}
        if getattr(self.pooling, 'needs_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'true_frag_ids'):
                extra_kwargs['true_frag_ids'] = data_batch.true_frag_ids
        if getattr(self.pooling, 'needs_rot_frag_ids', False) and data_batch is not None:
            if hasattr(data_batch, 'rot_frag_ids'):
                extra_kwargs['rot_frag_ids'] = data_batch.rot_frag_ids

        y, pool_loss = self.pooling(x, edge_index, edge_attr, batch, **extra_kwargs)

        for lin in self.lins:
            y = F.relu(lin(y))
            y = F.dropout(y, p=self.dropout_p, training=self.training)
        y = self.out(y)
        return y, pool_loss

    model.forward = types.MethodType(new_forward, model)
    return model



def _edge_to_node_importance(edge_mask, edge_index, num_nodes):
    """边重要性 → 原子重要性：两端所有入射边的 mean（保留正负）"""
    device = edge_mask.device
    both_idx  = torch.cat([edge_index[0], edge_index[1]])   # (2E,)
    both_mask = torch.cat([edge_mask, edge_mask])           # (2E,)

    num_sum = torch.zeros(num_nodes, device=device)
    num_cnt = torch.zeros(num_nodes, device=device)
    num_sum.scatter_add_(0, both_idx, both_mask)
    num_cnt.scatter_add_(0, both_idx, torch.ones_like(both_mask))

    return num_sum / num_cnt.clamp(min=1.0)

def _manual_ig_batch(model, x, edge_index, edge_attr, node_batch, data_batch,
                     num_graphs, target_class=1, n_steps=50):
    """
    单 batch IG（一次前向处理 batch 中所有图）：
      - baseline = edge_weight 全 0（所有边断开）
      - input    = edge_weight 全 1（原图）
      - 沿 α ∈ (0,1] 采样 n_steps 点，累积 ∂(Σ_g logits[g, tc]) / ∂edge_weight
      - 每条边的梯度天然来自其所属图（不同图 logits 相互独立）
    返回：
        list[np.ndarray]，长度 = num_graphs，每个元素 (n_atoms_g,)，带符号
    """
    model.eval()
    num_edges = edge_index.size(1)
    if num_edges == 0:
        raise ValueError("num_edges is 0, cannot proceed")

    total_grad = torch.zeros(num_edges, device=x.device)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp = torch.full((num_edges,), alpha,
                            device=x.device, requires_grad=True)
        logits, _ = model(x, edge_index, edge_attr, node_batch,
                          data_batch=data_batch, edge_weight=interp)

        # ★ 关键：对 batch 中所有图的目标类 logit 求和，一次反传
        grad = torch.autograd.grad(logits[:, target_class].sum(), interp)[0]
        total_grad += grad

    edge_imp = total_grad / n_steps                         # (E,)
    node_imp = _edge_to_node_importance(
        edge_imp, edge_index, x.size(0)
    ).cpu().numpy()                                         # (N,)

    # 按图拆分回每个分子
    node_batch_np = node_batch.cpu().numpy()
    return [node_imp[node_batch_np == g] for g in range(num_graphs)]


def evaluate_ig(model, test_loader, device, n_steps=50, target_class=1):
    """
    遍历 test_loader（batch_size 任意），用 IG 计算每个分子的原子重要性。

    返回：
        atom_importances: list[np.ndarray]
            长度 = 分子数，顺序与 test_loader 遍历顺序一致
            （即 test_set.data_list 的顺序）
            每个元素形状 (n_atoms,)，带符号（正=支持预测，负=反对）
    """
    model.eval()
    model = _patch_gcn_forward_for_edge_weight(model)
    print("[IG] GCN forward patched for edge_weight")

    atom_importances = []
    skipped = {"err": 0}

    for batch_idx_outer, batch in enumerate(test_loader):
        batch = batch.to(device)

        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        node_batch = batch.batch                       # ★ 正确取 batch.batch
        num_graphs = int(node_batch.max().item()) + 1

        try:
            with torch.enable_grad():
                per_mol = _manual_ig_batch(
                    model, x, ei, ea, node_batch, batch,
                    num_graphs,
                    target_class=target_class,
                    n_steps=n_steps,
                )
            atom_importances.extend(per_mol)
        except Exception as e:
            print(f"  [IG batch={batch_idx_outer}] {type(e).__name__}: {e}")
            # 回退：每个分子填 0，保证长度对齐
            node_batch_np = node_batch.cpu().numpy()
            for g in range(num_graphs):
                n_atoms_g = int((node_batch_np == g).sum())
                atom_importances.append(np.zeros(n_atoms_g, dtype=np.float32))
            skipped["err"] += 1

    print(f"[IG] 完成，共 {len(atom_importances)} 个分子，跳过: {skipped}")
    return atom_importances

'''
def _manual_ig(model, x, edge_index, edge_attr, batch_vec, data_batch,
               target_class=1, n_steps=50):
    """
    单分子 IG：
      - baseline = edge_weight 全 0（所有边断开）
      - input    = edge_weight 全 1（原图）
      - 沿 α ∈ (0,1] 采样 n_steps 点，累积 ∂F/∂edge_weight
    返回：原子重要性 (numpy, 带符号)
    """
    model.eval()
    num_edges = edge_index.size(1)
    print("num_edges")
    print(num_edges)
    if num_edges == 0:
        raise ValueError("num_edges is 0, cannot proceed")
    total_grad = torch.zeros(num_edges, device=x.device)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp = torch.full((num_edges,), alpha,
                            device=x.device, requires_grad=True)
        #print("interp")
        #print(interp)
        logits, _ = model(x, edge_index, edge_attr, batch_vec,
                          data_batch=data_batch, edge_weight=interp)
        
        #print("logits.shape")
        #print(logits.shape)
        grad = torch.autograd.grad(logits[0, target_class], interp)[0]
        #print("grad.shape")
        #print(grad.shape)
        total_grad += grad

    edge_imp = total_grad / n_steps
    return _edge_to_node_importance(edge_imp, edge_index, x.size(0)).cpu().numpy()


def evaluate_ig(model, test_loader, device, n_steps=50, target_class=1):
    """
    遍历 test_loader（batch_size=1），用 IG 计算每个分子的原子重要性。

    返回：
        atom_importances: list[np.ndarray]
            长度 = 分子数，顺序与 test_loader 遍历顺序一致
            （即 test_set.data_list 的顺序）
            每个元素形状 (n_atoms,)，带符号（正=支持有毒，负=反对有毒）
    """
    model.eval()
    model = _patch_gcn_forward_for_edge_weight(model)
    print("[IG] GCN forward patched for edge_weight")


    atom_importances = []
    skipped = {"no_edge": 0, "err": 0}

    for global_idx, batch in enumerate(test_loader):
        batch = batch.to(device)

        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        bv = torch.zeros(x.size(0), dtype=torch.long, device=device)

        try:
            with torch.enable_grad():
                atom_imp = _manual_ig(
                    model, x, ei, ea, bv,
                    data_batch=batch,
                    target_class=target_class,
                    n_steps=n_steps,
                )
            atom_importances.append(atom_imp)
        except Exception as e:
            print(f"  [IG idx={global_idx}] {type(e).__name__}: {e}")
            atom_importances.append(np.zeros(x.size(0), dtype=np.float32))
            skipped["err"] += 1

    print(f"[IG] 完成，共 {len(atom_importances)} 个分子，跳过: {skipped}")
    return atom_importances
'''
'''
def evaluate_attention(model, dataloader, device):
    """
    method='Attention'：
      - 跑一次 forward，从 model.pooling.last_atom_weight 拿原子权重
      - 逐分子算指标
    """
    model.eval()
    atom_importances = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # 1) 触发前向，拿到原子权重
            _ = model(batch.x, batch.edge_index, batch.edge_attr,
                      batch.batch, data_batch=batch)
            atom_weights_all = model.pooling.last_atom_weight.cpu().numpy()

            # 2) 拿到每个分子在 batch 里的原子范围
            batch_idx = batch.batch.cpu().numpy()
            num_mols = int(batch_idx.max()) + 1

            # 3) 逐分子计算
            for b in range(num_mols):
                mask_bool = (batch_idx == b)
                atom_w   = atom_weights_all[mask_bool]
                atom_importances.append(atom_w)
 
    return atom_importances
    
'''
'''
def evaluate_attention(model, dataloader, device,
                       use_frag_weighting=True,
                       frag_count_threshold=2):
    """
    method='Attention'：
      - 跑一次 forward，从 model.pooling 拿原子权重 & 片段权重
      - 逐分子算重要性：
          · 若该分子片段数 <= frag_count_threshold：直接用原子权重
          · 否则：原子权重 × 该原子所属片段的权重
    """
    model.eval()
    atom_importances = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # 1) 触发一次前向，填充 pooling 缓存
            _ = model(batch.x, batch.edge_index, batch.edge_attr,
                      batch.batch, data_batch=batch)

            atom_weights_all = model.pooling.last_atom_weight.cpu().numpy()   # (N,)
            batch_idx        = batch.batch.cpu().numpy()                      # (N,)
            num_mols         = int(batch_idx.max()) + 1

            # 2) 尝试取片段级信息（仅当 pooling 是 BRICS 类且已缓存）
            frag_weights  = None
            frag_idx_all  = None
            if (use_frag_weighting
                    and hasattr(model.pooling, 'last_frag_weight')
                    and model.pooling.last_frag_weight is not None
                    and hasattr(model.pooling, '_frag_idx')
                    and model.pooling._frag_idx is not None):
                frag_weights = model.pooling.last_frag_weight.cpu().numpy()   # (num_frags_global,)
                frag_idx_all = model.pooling._frag_idx.cpu().numpy()          # (N,) 全局片段索引

            # 3) 逐分子计算
            for b in range(num_mols):
                mask_bool = (batch_idx == b)
                atom_w    = atom_weights_all[mask_bool]

                if frag_weights is not None:
                    frag_ids_b = frag_idx_all[mask_bool]
                    n_frags_b  = len(np.unique(frag_ids_b))
                    if n_frags_b > frag_count_threshold:
                        frag_w_for_atoms = frag_weights[frag_ids_b]
                        atom_w = atom_w * frag_w_for_atoms

                atom_importances.append(atom_w)

    return atom_importances
'''
def evaluate_attention(model, dataloader, device, **kwargs):
    """
    获取原子级别的注意力重要性。
    针对 BRICS 类池化方法，最终重要性 = (片段内原子 softmax 权重) × (片段在分子级的 softmax 权重)。
    这样既考虑了原子在片段里的相对重要性，又考虑了片段对整个分子的重要程度。
    """
    model.eval()
    atom_importances = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # 1) 触发一次前向，触发 pooling 内部缓存 last_atom_weight / last_frag_weight / _frag_idx
            _ = model(batch.x, batch.edge_index, batch.edge_attr,
                      batch.batch, data_batch=batch)

            # 2) 获取原子级权重
            # 对于 BRICS 池化：这是片段内部的 softmax 相对权重
            # 对于 GlobalAttention：这是整个分子的全局 softmax 权重
            atom_weights_all = model.pooling.last_atom_weight.cpu().numpy()   # (N,)
            batch_idx        = batch.batch.cpu().numpy()                      # (N,)
            num_mols         = int(batch_idx.max()) + 1

            # 3) 检查池化模块是否带有片段级信息
            frag_weights = None
            frag_idx_all = None
            
            if (hasattr(model.pooling, 'last_frag_weight')
                    and model.pooling.last_frag_weight is not None
                    and hasattr(model.pooling, '_frag_idx')
                    and model.pooling._frag_idx is not None):
                # 获取片段级 softmax 权重，以及每个原子属于哪个全局片段
                frag_weights = model.pooling.last_frag_weight.cpu().numpy()   # (num_frags_global,)
                frag_idx_all = model.pooling._frag_idx.cpu().numpy()          # (N,)

            # 4) 逐分子计算最终重要性
            for b in range(num_mols):
                mask_bool = (batch_idx == b)
                atom_w    = atom_weights_all[mask_bool]

                # ★ 核心修改：如果是片段池化，计算 片段内原子贡献 × 片段重要性
                if frag_weights is not None:
                    frag_ids_b = frag_idx_all[mask_bool]
                    # 找到该分子中每个原子对应的片段权重
                    frag_w_for_atoms = frag_weights[frag_ids_b] 
                    # 相乘得到原子对分子的绝对贡献度
                    atom_w = atom_w * frag_w_for_atoms

                atom_importances.append(atom_w)

    return atom_importances
# ==================== PGExplainer 模型包装器 ====================
class PGExplainerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                data_batch=None, **kwargs):
        edge_weight = kwargs.pop('edge_weight', None)
        if edge_weight is not None:
            try:
                y, _ = self.base_model(
                    x, edge_index, edge_attr, batch,
                    data_batch=data_batch, edge_weight=edge_weight
                )
            except TypeError:
                y, _ = self.base_model(
                    x, edge_index, edge_attr, batch, data_batch=data_batch
                )
        else:
            y, _ = self.base_model(
                x, edge_index, edge_attr, batch, data_batch=data_batch
            )
        return y[:, 1]

# ==================== PGExplainer 评估（支持 batch_size > 1）====================
def evaluate_pgexplainer(model, test_loader, device,
                         epochs=30, lr=0.003,
                         in_dim=256, hidden_dim=64,
                         target_class=1):
    model.eval()

    # 1. 接口对齐：让 GNN_Base_GCN.forward 支持 edge_weight
    model = _patch_gcn_forward_for_edge_weight(model)

    wrapped = PGExplainerWrapper(model).to(device)
    wrapped.eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=PGExplainer(
            epochs=epochs, lr=lr,
            in_dim=in_dim, hidden_dim=hidden_dim,
            bias=0.01,
        ),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    explainer.algorithm.to(device)

    # 训练
    print(f"[PGExplainer] 在测试集上训练 {epochs} 轮...")
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in test_loader:
            batch = batch.to(device)
            wrapped.eval()
            target = torch.full((batch.num_graphs,), float(target_class),
                                device=device, dtype=torch.float)
            explainer.algorithm.train(
                epoch, wrapped,
                batch.x, batch.edge_index,
                target=target,
                index=batch.batch,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                data_batch=batch,
            )
            torch.nn.utils.clip_grad_norm_(
                explainer.algorithm.mlp.parameters(), 1.0
            )
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"  [PGExplainer] epoch {epoch+1}/{epochs} done ({n_batches} batches)")

    # 推理
    atom_importances = []
    for global_idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch_idx = batch.batch.cpu().numpy()
        num_mols = int(batch_idx.max()) + 1
        try:
            target = torch.full((batch.num_graphs,), float(target_class),
                                device=device, dtype=torch.float)
            explanation = explainer(
                batch.x, batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                data_batch=batch,
                target=target,
                index=batch.batch,
            )
            edge_mask = explanation.edge_mask
            if edge_mask is None:
                raise RuntimeError("PGExplainer 返回 None edge_mask")
            node_imp = _edge_to_node_importance(
                edge_mask, batch.edge_index, batch.x.size(0)
            ).cpu().numpy()
            for b in range(num_mols):
                mask_b = (batch_idx == b)
                atom_importances.append(
                    node_imp[mask_b].astype(np.float32)
                )
        except Exception as e:
            print(f"  [PGExplainer idx={global_idx}] {type(e).__name__}: {e}")
            for b in range(num_mols):
                n_atoms_b = int((batch_idx == b).sum())
                atom_importances.append(
                    np.zeros(n_atoms_b, dtype=np.float32)
                )

    print(f"[PGExplainer] 完成，共 {len(atom_importances)} 个分子")
    return atom_importances

def calculate_metrics(importance_scores, true_mask, explain_type,top_k_ratio=None):
    """
    计算单个分子的所有指标。
    - importance_scores: 模型输出的原子权重 (numpy array)
    - true_mask: 真实的 alert 掩码 (0/1 numpy array)
    - top_k_ratio: 选取前百分之多少作为预测结果
    """
    n_atoms = len(true_mask)
    n_pos = int(true_mask.sum())
    print("true_mask")
    print(len(true_mask))
    print("importance_scores")
    print(len(importance_scores))
    # 如果全阳性或全阴性，指标无意义
    if n_pos == 0 or n_pos == n_atoms:
        raise ValueError(
            f"无法计算指标: n_pos={n_pos}, n_atoms={n_atoms}. ")

    # 1. 计算 AUC（用连续分数，直接算）
    try:
        auc = roc_auc_score(true_mask, importance_scores)
    except Exception:
        raise ValueError(
            f"全阴性(n_pos=0)或全阳性(n_pos=n_atoms)分子无法计算 AUC/ACC。")

    # 2. 基于 Top-K 截断，生成预测掩码 (0/1)
    if explain_type in ['moepool','global_attention','pgexplainer']:
        k = max(1, int(round(n_atoms * top_k_ratio)))
        #k = min(k, n_atoms - 1) # 防止 k 等于 n_atoms
        topk_idx = np.argsort(-importance_scores)[:k]
        pred_mask = np.zeros(n_atoms, dtype=int)
        pred_mask[topk_idx] = 1
    elif explain_type=="ig":
        # IG：分数 > 0 的原子判为阳性
        pred_mask = (importance_scores > 0).astype(int)
    else:
        raise ValueError(
            f"explain type not exist:{explain_type}")

    # 3. 计算混淆矩阵指标
    tp = int(((pred_mask == 1) & (true_mask == 1)).sum())
    fp = int(((pred_mask == 1) & (true_mask == 0)).sum())
    fn = int(((pred_mask == 0) & (true_mask == 1)).sum())
    tn = int(((pred_mask == 0) & (true_mask == 0)).sum())

    acc  = (tp + tn) / n_atoms
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "AUC": auc,
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "FPR": fpr,
        "n_atoms": n_atoms,
        "n_pos": n_pos,
        "score_list": importance_scores, # 保存原始分数供全局 AUC 用
        "true_list": true_mask
    }




# ==================== 数据加载 ====================
def load_fold_data(csv_path, smiles_col, label_col=None, label_cols=None,
                   fold_col=None, task_type="regression"):
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    splits = df[fold_col].tolist()
    if task_type == 'multitask':
        if label_cols is None:
            label_cols = [c for c in df.columns if c != smiles_col and 'fold' not in c.lower() and c.lower() != 'mol_id']
        labels_raw = df[label_cols].values.tolist()
        task_type = "multitask"
        num_classes = len(label_cols)
    else:
        labels_raw = df[label_col].tolist()
        if task_type in ["classification", "multiclass"]:
            unique = sorted(set(labels_raw))
            mapping = {v: i for i, v in enumerate(unique)}
            labels = [mapping[v] for v in labels_raw]
            num_classes = len(unique)
        else:
            labels = [float(v) for v in labels_raw]
            num_classes = None
        labels_raw = labels
    train_smi, train_y, val_smi, val_y, test_smi, test_y = [], [], [], [], [], []
    for i, (s, y, sp) in enumerate(zip(smiles, labels_raw, splits)):
        if Chem.MolFromSmiles(s) is None:
            continue
        if sp == 0:
            train_smi.append(s)
            train_y.append(y)
        elif sp == 1:
            val_smi.append(s)
            val_y.append(y)
        else:
            test_smi.append(s)
            test_y.append(y)
    return train_smi, train_y, val_smi, val_y, test_smi, test_y, task_type, num_classes


# ==================== 单折实验 ====================
def run_one_fold(test_smi, test_y,
                 task_type, hidden_dim, num_layers,
                 dropout, use_layernorm, num_classes, epochs, seed,
                 batch_size=1, save_path=None, dataset_name="dataset",
                 fold_idx=0, backbone="gcn",pool_type=None,load_model_path=None,explain_type=None,top_k_ratio=None,true_mask_list=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # 先 warn_only 试试        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    test_env = generate_env_labels(test_smi)

    cache_dir = "./feature_cache_ogb_explain"#sum,mean,max:./feature_cache_central
    test_cache = os.path.join(cache_dir, f"{dataset_name}_fold{fold_idx}_explain.pt")

    test_set = MoleculeDataset(test_smi, test_y, task_type, "explain", env_labels=test_env, cache_path=test_cache, recal=True)

    from torch.utils.data import DataLoader as TorchDataLoader
    test_loader = TorchDataLoader(test_set, batch_size=batch_size, shuffle=False,
                                  collate_fn=my_collate, num_workers=0)

    out_ch = 2 if task_type == "classification" else (num_classes if task_type in ["multiclass", "multitask"] else 1)
    convs = [hidden_dim] * num_layers
    norm = LayerNorm if use_layernorm else None
    in_dim = test_set[0].x.size(1)
    if pool_type=="sum":
        pooling = SumPooling(hidden_dim)
    elif pool_type=="max":
        pooling = MaxPooling(hidden_dim)
    elif pool_type=="mean":
        pooling = MeanPooling(hidden_dim)
    elif pool_type == "brics_atom_cross_attn_fusion_v2_align_nodecorr":
        pooling = BRICSAtomCrossAttnFusionV2AlignPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            align_weight=0.05,   # 0.05
        )
    elif pool_type == "ablation_v2align_frag_only":
        pooling = BRICSAtomCrossAttnFusionV2AblationPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            branch_mode='frag_only',
        )
    elif pool_type == "ablation_v2align_atom_only":
        pooling = BRICSAtomCrossAttnFusionV2AblationPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            branch_mode='atom_only',
        )
        
    elif pool_type == "ablation_v2align_concat":
        pooling = BRICSAtomCrossAttnFusionV2AblationPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            branch_mode='concat',
        )              
    elif pool_type == "ablation_v2align_fused_no_align":
        pooling = BRICSAtomCrossAttnFusionV2AlignPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            align_weight=0,   # ★ 对齐损失权重，建议 0.01~0.05
        )
    elif pool_type == "ablation_v2align_fused_no_decorr":
        pooling = BRICSAtomCrossAttnFusionV2AlignPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
            de_corr_weight=0,
            align_weight=0.05,   # ★ 对齐损失权重，建议 0.01~0.05
        )                     
    elif pool_type == "set2set":
        pooling = Set2SetPooling(
            in_channels=hidden_dim,
            processing_steps=2,
            num_layers=1,
        )
    elif pool_type == "global_attn_weight":
        pooling = GlobalAttentionPooling(
            in_channels=hidden_dim,
            gate_hidden=hidden_dim // 2,
        )
    elif pool_type == "gmt":
        pooling = GMTPooling(
            in_channels=hidden_dim,
            k=10,                  # 代表性节点数，可调
            num_encoder_blocks=1,
            heads=4,
            layer_norm=use_layernorm,
            dropout=dropout,
        )        

    model = build_model(
        backbone_name=backbone,
        in_dim=in_dim,
        out_ch=out_ch,
        edge_dim=10,
        pooling=pooling,
        convs=convs,
        dense_layers=[128, 64],
        dropout=dropout,
        norm=norm
    ).to(device)   

    if load_model_path is not None:
        print(f"Loading model from {load_model_path} for evaluation...")
        checkpoint = torch.load(load_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if explain_type=="ig":
            atom_importances = evaluate_ig(
                                model, test_loader, device,
                                n_steps=50, target_class=1,
                            )
            results = [calculate_metrics(m,n,explain_type) for m,n in zip(atom_importances,true_mask_list)]

        #val_score  = evaluate(model, val_loader,  task_type, device)   # ★ 新增
        elif explain_type in ["moepool","global_attention"]:
            atom_importances = evaluate_attention(model, test_loader, device)
            results = [calculate_metrics(m,n,explain_type,top_k_ratio) for m,n in zip(atom_importances,true_mask_list)]
        elif explain_type == "pgexplainer":
            atom_importances = evaluate_pgexplainer(
                model, test_loader, device,
                epochs=60, lr=0.003,
                target_class=1,
                in_dim=2 * hidden_dim,   # 动态取（39）
                hidden_dim=64,
            )
            results = [calculate_metrics(m, n, explain_type, top_k_ratio)
                       for m, n in zip(atom_importances, true_mask_list)]
        print("len(results)")
        print(len(results))
        return {
            "ACC": np.mean([m["ACC"] for m in results]),
            "Precision": np.mean([m["Precision"] for m in results]),
            "Recall": np.mean([m["Recall"] for m in results]),
            "FPR": np.mean([m["FPR"] for m in results]),
            "AUC_macro": np.mean([m["AUC"] for m in results]),
        }        
    else:
        raise ValueError(f"load_model_path is None, 无法进行评估。"
                         f"请检查传入的模型路径: {load_model_path}")


def load_best_hyperparams(dataset, backbone, pool, batch_size, ref_seed,
                          model_dir="./models_best_ogb"):
    """
    从参考 seed (516) 的已保存模型中读取最佳超参数。
    保存路径：{model_dir}_{ref_seed}/{dataset}_{backbone}_{pool}_batch{batch_size}_best_fold1.pt
    Returns:
        (hidden_dim, num_layers, dropout, use_layernorm) 或 None
    """
    ref_model_dir = f"{model_dir}_{ref_seed}"
    model_filename = f"{dataset}_{backbone}_{pool}_batch{batch_size}_best_fold1.pt"
    save_path = os.path.join(ref_model_dir, model_filename)

    if not os.path.isfile(save_path):
        print(f"  [load_best_hyperparams] Not found: {save_path}")
        return None
    try:
        info = torch.load(save_path, map_location='cpu')
        h  = info.get('hidden_dim')
        l  = info.get('num_layers')
        d  = info.get('dropout')
        ln = info.get('use_layernorm')
        if None in (h, l, d, ln):
            print(f"  [load_best_hyperparams] Missing fields in {save_path}")
            return None
        return (int(h), int(l), float(d), bool(ln))
    except Exception as e:
        print(f"  [load_best_hyperparams] Failed to load {save_path}: {e}")
        return None

def parse_mask(s):
    """'0,0,1,0' -> np.array([0,0,1,0], dtype=int)"""
    return np.array([int(x) for x in str(s).split(",")], dtype=int)


    
# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pooltype", type=str, default="sum",
                    choices=["sum","global_attn","brics_atom_cross_attn_fusion_v2_align_nodecorr","gmt","asap",\
                             "ablation_v2align_frag_only","ablation_v2align_atom_only","ablation_v2align_fused_no_align","ablation_v2align_concat","global_attn_weight"])
    parser.add_argument("--backbone", type=str, default="gcn",
                        choices=["gine", "gin", "gcn", "gat","gatv2"],
                        help="GNN backbone: gine / gin / gcn / gat")
    parser.add_argument("--explain_type", type=str,default="moepool",choices=['moepool','ig','global_attention',"pgexplainer"])
    args = parser.parse_args()
    BATCH_SIZE =32
    pool = args.pooltype  # 固定 Sum 池化
    explain_type=args.explain_type
    datasets_cfg = {
        'mutagenicity': {"csv": "/data1/nibingwei/fieldgnn/data/Mutagenicity_interpretable/mutagenicity_test_alert_final.csv", "smiles": "smiles", "label": "mutagenicity", "task": "classification"},
        "Aoral":{"csv": "./data/Aoral_with_split.csv", "smiles": "SMILES", "label": "Labels", "task": "classification"},
    }
    seed_list = [0,8,42, 67,123,322,999,2024,1001,4239]

    cfg = datasets_cfg[args.dataset]
    df_alert = pd.read_csv(cfg["csv"])
    true_mask_list=[parse_mask(s) for s in df_alert["alert_mask_str"]] 
    task_type = cfg["task"]
    csv_path = cfg["csv"]
    smiles_col = cfg["smiles"]
    label_col = cfg.get("label", None)
    label_cols = cfg.get("label_cols", None)
    fold_col = "fold1_split"
    # ==================== 评估所有种子的模型，计算测试集均值和方差 ====================
    print("\n" + "="*60)
    print("Evaluating all seeds on test set...")
    print("="*60)
    
    seed_results={}
    for seed_id in seed_list:
        model_dir = f"./models_best_ogb_layernorm_{seed_id}"#./models_best_ogb_new_ablation_nodecorr_{seed_id}
        model_filename = f"{args.dataset}_{args.backbone}_{pool}_batch32_best_fold1.pt"
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.isfile(model_path):
            print(f"[Seed {seed_id}] Model not found: {model_path}")
            raise FileNotFoundError(f"[Seed {seed_id}] Model not found: {model_path}")
            #continue

        # 读取模型保存的超参数

        model_info = torch.load(model_path, map_location='cpu')
        h  = model_info['hidden_dim']
        l  = model_info['num_layers']
        d  = model_info['dropout']
        ln = model_info['use_layernorm']
        bs = model_info.get('batch_size', BATCH_SIZE)
        task_type_model = model_info['task_type']
        out_ch = model_info['out_ch']

        
        
        if task_type_model == 'classification':
            num_classes_arg = 2
        elif task_type_model in ['multiclass', 'multitask']:
            num_classes_arg = out_ch
        else:
            num_classes_arg = None

        # 加载 fold1 数据
        train_smi, train_y, val_smi, val_y, test_smi, test_y, task_type_out, _ = load_fold_data(
            csv_path, smiles_col, label_col=label_col, label_cols=label_cols,
            fold_col=fold_col, task_type=task_type
        )
        print("len test smi")
        print(len(test_smi))
        print("len train smi")
        print(len(train_smi))
        print("len val smi")
        print(len(val_smi))        
        # 纯评估
        results= run_one_fold(     # ★ 改名接收
            test_smi, test_y,
            task_type_out, h, l, d, ln, num_classes_arg,
            0, seed_id,
            batch_size=BATCH_SIZE,
            save_path=None,
            dataset_name=args.dataset,
            fold_idx=1,
            backbone=args.backbone,
            pool_type=pool,
            load_model_path=model_path,explain_type=explain_type,true_mask_list=true_mask_list,top_k_ratio=0.3
        )
        seed_results[seed_id] = results

        # ============================================================
        # ① per-seed CSV：加两列 ckpt_val_score / val_score_eval
        # ============================================================
        per_seed_csv = f"{args.backbone}_{pool}_seed_eval_{BATCH_SIZE}_{explain_type}.csv"
        per_seed_fields = [
            "dataset", "backbone", "pool", "seed",
            "hidden_dim", "num_layers", "dropout", "use_layernorm", "batch_size","ACC", "Precision", "Recall", "FPR", "AUC_macro",

        ]
        file_exists = os.path.isfile(per_seed_csv) and os.path.getsize(per_seed_csv) > 0
        with open(per_seed_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_seed_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset":         args.dataset,
                "backbone":        args.backbone,
                "pool":            pool,
                "seed":            seed_id,
                "num_layers":      l,
                "dropout":         d,
                "use_layernorm":   ln,
                "batch_size":      bs,
                "hidden_dim":h,
                **results

            })
        print(f"Per-seed results appended to {per_seed_csv}")

        # ============================================================
        # ② test mean 矩阵（原有）
        # ============================================================
    METRIC_KEYS = ["ACC", "Precision", "Recall", "FPR", "AUC_macro"]
    mean_metrics = {
        k: float(np.mean([seed_results[s][k] for s in seed_results]))
        for k in METRIC_KEYS
    }
    print(f"[Mean over {len(seed_results)} seeds] {mean_metrics}")

    mean_csv = f"{args.backbone}_{explain_type}_mean.csv"
    df = pd.read_csv(mean_csv, index_col=0) if os.path.isfile(mean_csv) and os.path.getsize(mean_csv) > 0 else pd.DataFrame()
    if args.dataset not in df.columns:
        df[args.dataset] = np.nan
    for metric, val in mean_metrics.items():
        df.loc[f"{pool}__{metric}", args.dataset] = val
    df.to_csv(mean_csv, float_format="%.6f")

if __name__ == "__main__":
    main()