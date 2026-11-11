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
from sklearn.metrics import roc_auc_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import BondType, BondStereo
from collections import Counter
from itertools import product
from torch_geometric.utils import softmax, scatter

from torch_geometric.utils import softmax, scatter, to_dense_batch, to_dense_adj
from torch_geometric.explain import Explainer, PGExplainer
from decimal import Decimal, ROUND_DOWN
# ==================== 常量与特征生成 ====================
import networkx as nx
# ==================== 从训练脚本导入公共组件 ====================
from model_gnns_seed import (
    # 常量
    ATOM_SYMBOLS, DEGREE_LIST, HYBRID_LIST, HYDROGEN_LIST,
    # 特征与图构建
    compute_atom_centrality, get_atom_features_39, get_edge_features,
    generate_env_labels, get_brics_frag_ids, get_rotatable_bond_frag_ids,
    # 数据集与 collate
    MoleculeDataset, my_collate,
    # 基础池化
    SumPooling, MaxPooling, MeanPooling, MeanMaxPooling,
    Set2SetPooling, GMTPooling, GlobalAttentionPooling,
    # DiffPool / MinCutPool
    _build_dense_gin_mlp, AssignModule, DiffPoolModel, MinCutPoolModel,
    # 层次化（★ 必须用文件2 版本，见下文说明）
    GNN_Base_HierarchicalSAGTopK, GNN_Base_HierarchicalEdgePool,
    # 片段间 GNN 构建
    _build_frag_conv, _apply_frag_conv,
    # BRICS 片段池化
    BRICSFragmentPoolOriginal,
    BRICSAtomFusionPool, BRICSAtomConcatFusionPool,
    ManualMultiheadAttention,
    BRICSAtomCrossAttnFusionV2Pool,
    SignedAttentionScorer, BRICSAtomCrossAttnFusionSignedPool,
    BRICSAtomCrossAttnFusionV2AlignPool,
    BRICSAtomCrossAttnFusionV2AblationPool,
    BRICSAtomCrossAttnFusionV2AlignNoFragGNNPool,
    # 骨架
    GNN_Base_GINE, GNN_Base_GIN, GNN_Base_GCN,
    GNN_Base_GAT, GNN_Base_GATv2,
    build_model,
    # 数据加载
    load_fold_data,
)
# ================================================================
def trunc_fmt(x, n=4):
    """
    截断到 n 位小数（不四舍五入）
    - 0.72345 → '0.7234'
    - 0.72399 → '0.7239'
    - 用 Decimal 避免浮点乘法误差
    """
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    d = Decimal(float(x)).quantize(Decimal(10) ** -n, rounding=ROUND_DOWN)
    return f"{d:.{n}f}"

   
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

def _patch_gatv2_forward_for_edge_weight(model):
    """GATv2 没有 edge_weight，把 edge_weight 乘到 edge_attr 上"""
    def new_forward(self, x, edge_index, edge_attr, batch,
                    data_batch=None, edge_weight=None):
        for i, conv in enumerate(self.convs):
            if edge_weight is not None:
                ea = edge_attr * edge_weight.unsqueeze(-1)
            else:
                ea = edge_attr
            x = conv(x, edge_index, ea)

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
        print("logits")
        #print(logits.shape)
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
'''
def _manual_ig_batch(model, x, edge_index, edge_attr, node_batch, data_batch,
                     num_graphs, target_class=1, n_steps=50,
                     verify_each_graph=False, verify_step=1):
    """
    单 batch IG（一次前向处理 batch 中所有图）：
      - baseline = edge_weight 全 0（所有边断开）
      - input    = edge_weight 全 1（原图）
      - 沿 α ∈ (0,1] 采样 n_steps 点，累积 ∂(Σ_g logits[g, tc]) / ∂edge_weight
      - 每条边的梯度天然来自其所属图（不同图 logits 相互独立）

    新增参数：
      verify_each_graph : bool
          True 时，在 verify_step 这一步额外做"逐图求导累加"，
          与 sum 版本对比，打印 max|sum - each|。
      verify_step : int
          在第几步（1-based）做验证，默认第 1 步。

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
        # ==================== 验证路径：sum vs 逐图累加 ====================
        # 1) sum 版本（保留图，后面还要用）
        S = logits[:, target_class].sum()
        print("torch.autograd.grad")
        print(torch.autograd.grad(S, interp, retain_graph=True))
        grad_sum = torch.autograd.grad(S, interp, retain_graph=True)[0]

        # 2) 逐图求导并累加
        grad_each = torch.zeros_like(interp)
        edge_graph_id = node_batch[edge_index[0]]          # (E_total,) 每条边的图 id

        for g in range(num_graphs):
            is_last = (g == num_graphs - 1)
            grad_g = torch.autograd.grad(
                logits[g, target_class], interp, retain_graph=not is_last
            )[0]
            grad_each += grad_g

            # 当前图的边 id 列表（应为连续区间）
            ids = (edge_graph_id == g).nonzero(as_tuple=True)[0].tolist()
            rng = f"[{ids[0]}, {ids[-1]}]" if ids else "空"

            # 当前图 grad_g 里非 0 的边 id
            nz_ids = (grad_g != 0).nonzero(as_tuple=True)[0].tolist()

            print(f"[g={g:2d}] 本图边范围 {rng}, 共 {len(ids)} 条")
            print(f"[g={g:2d}] 非0 边id ({len(nz_ids)}): {nz_ids}")
        # 3) 对比
        diff = (grad_sum - grad_each).abs().max().item()
        print(f"[IG check] step={step}, num_graphs={num_graphs}, "
              f"num_edges={num_edges}, max|sum-each|={diff:.3e}")
        print(f"  grad_sum [:8]  = {grad_sum[:8].tolist()}")
        print(f"  grad_each[:8]  = {grad_each[:8].tolist()}")

        # 4) 正式累加仍用 sum 版本（两者等价）
        total_grad += grad_sum

    edge_imp = total_grad / n_steps                         # (E,)
    node_imp = _edge_to_node_importance(
        edge_imp, edge_index, x.size(0)
    ).cpu().numpy()                                         # (N,)

    # 按图拆分回每个分子
    node_batch_np = node_batch.cpu().numpy()
    return [node_imp[node_batch_np == g] for g in range(num_graphs)]
'''
def evaluate_ig(model, test_loader, device, n_steps=50, target_class=1,backbone='gatv2'):
    """
    遍历 test_loader（batch_size 任意），用 IG 计算每个分子的原子重要性。

    返回：
        atom_importances: list[np.ndarray]
            长度 = 分子数，顺序与 test_loader 遍历顺序一致
            （即 test_set.data_list 的顺序）
            每个元素形状 (n_atoms,)，带符号（正=支持预测，负=反对）
    """
    model.eval()
    if backbone == "gcn":
        model = _patch_gcn_forward_for_edge_weight(model)
        print("[IG] GCN forward patched for edge_weight")
    elif backbone in ("gat", "gatv2", "gine"):
        model = _patch_gatv2_forward_for_edge_weight(model)
        print(f"[IG] {backbone} forward patched (edge_weight -> edge_attr)")
    else:  # gin 以及任何未知 backbone
        raise NotImplementedError(
            f"IG with backbone='{backbone}' is not supported. "
            f"GINConv 的 forward(x, edge_index) 不接受 edge_attr，"
            f"也不接受 edge_weight，无法构造'边从无到有'的可微连续路径，"
            f"IG 的完备性公理无法定义。"
            f"请改用 gcn / gine / gat / gatv2：\n"
            f"  - gcn    : edge_weight 原生支持，baseline = 图断开\n"
            f"  - gine   : edge_weight 乘到 edge_attr 上，baseline = 边特征清零\n"
            f"  - gat    : 同上\n"
            f"  - gatv2  : 同上（推荐，和你的训练配置一致）"
        )
    print("[IG] GCN forward patched for edge_weight")

    atom_importances = []
    skipped = {"err": 0}

    for batch_idx_outer, batch in enumerate(test_loader):
        batch = batch.to(device)

        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        '''
        print("x.shape")
        print(x.shape)
        print("ei.shape")
        print(ei.shape)
        print("ea.shape")
        print(ea.shape)

        print("node_batch")# ★ 正确取 batch.batch
        print(node_batch)

        print("num_graphs")
        print(num_graphs)
        
        '''
        node_batch = batch.batch
        num_graphs = int(node_batch.max().item()) + 1        
        try:
            with torch.enable_grad():
                per_mol = _manual_ig_batch(
                    model, x, ei, ea, node_batch, batch,
                    num_graphs,
                    target_class=target_class,
                    n_steps=n_steps,
                )
                #print("permol len")
                #print(len(per_mol))
                #print(per_mol)
            atom_importances.extend(per_mol)
            #print("atom_importances")
            #print(len(atom_importances))
        except Exception as e:
            raise RuntimeError(f"[IG batch={batch_idx_outer}] failed: {type(e).__name__}: {e}") from e
            #print(f"  [IG batch={batch_idx_outer}] {type(e).__name__}: {e}")
            # 回退：每个分子填 0，保证长度对齐
            node_batch_np = node_batch.cpu().numpy()
            for g in range(num_graphs):
                n_atoms_g = int((node_batch_np == g).sum())
                atom_importances.append(np.zeros(n_atoms_g, dtype=np.float32))
            skipped["err"] += 1

    #print(f"[IG] 完成，共 {len(atom_importances)} 个分子，跳过: {skipped}")
    return atom_importances


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
    cnt=0
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
                print("have frag id")
            else:
                print("none frag id")
            # 3) 逐分子计算
            for b in range(num_mols):
                mask_bool = (batch_idx == b)
                atom_w    = atom_weights_all[mask_bool]
                cnt+=1
                #print("count smiles")
                #print(cnt)
                if frag_weights is not None:
                    frag_ids_b = frag_idx_all[mask_bool]
                    n_frags_b  = len(np.unique(frag_ids_b))
                    if n_frags_b > frag_count_threshold:
                        frag_w_for_atoms = frag_weights[frag_ids_b]
                        atom_w = atom_w * frag_w_for_atoms
                #print("atom_w")
                #print(atom_w.shape)
                #print(atom_w)
                atom_importances.append(atom_w)
                #print(len(atom_importances))
    #print("len atom_importances")
    #print(len(atom_importances))
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
'''    
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
    #print("true_mask")
    #print(len(true_mask))
    #print(true_mask)
    #print("importance_scores")
    #print(len(importance_scores))
    #print(importance_scores)
    # 如果全阳性或全阴性，指标无意义
    if n_pos == 0 or n_pos == n_atoms:
        raise ValueError(
            f"无法计算指标: n_pos={n_pos}, n_atoms={n_atoms}. ")

    # 1. 计算 AUC（用连续分数，直接算）
    try:
        auc = roc_auc_score(true_mask, importance_scores)
        print("auc")
        print(auc)
    except Exception:
        raise ValueError(
            f"全阴性(n_pos=0)或全阳性(n_pos=n_atoms)分子无法计算 AUC/ACC。")

    # 2. 基于 Top-K 截断，生成预测掩码 (0/1)
    if explain_type in ['BACPOOL','global_attention','pgexplainer']:
        k = max(1, int(round(n_atoms * top_k_ratio)))
        print("k")
        print(k)
        #k = min(k, n_atoms - 1) # 防止 k 等于 n_atoms
        topk_idx = np.argsort(-importance_scores)[:k]
        pred_mask = np.zeros(n_atoms, dtype=int)
        pred_mask[topk_idx] = 1
    elif explain_type=="ig":
        # IG：分数 > 0 的原子判为阳性
        #pred_mask = (importance_scores > 0).astype(int)
        scores = np.asarray(importance_scores, dtype=float)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        k = max(1, int(round(n_atoms * top_k_ratio)))
        print("k")
        print(k)
        topk_idx = np.argsort(-np.abs(scores))[:k]
        pred_mask = np.zeros(n_atoms, dtype=int)
        pred_mask[topk_idx] = 1
    else:
        raise ValueError(
            f"explain type not exist:{explain_type}")

    # 3. 计算混淆矩阵指标
    print("(pred_mask == 1) & (true_mask == 1))")
    #print((pred_mask == 1) & (true_mask == 1))
    tp = int(((pred_mask == 1) & (true_mask == 1)).sum())
    fp = int(((pred_mask == 1) & (true_mask == 0)).sum())
    fn = int(((pred_mask == 0) & (true_mask == 1)).sum())
    tn = int(((pred_mask == 0) & (true_mask == 0)).sum())

    acc  = (tp + tn) / n_atoms
    prec = tp / (tp + fp)# if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn)# if (tp + fn) > 0 else np.nan
    fpr  = fp / (fp + tn)# if (fp + tn) > 0 else np.nan

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
    norm = nn.LayerNorm if use_layernorm else None
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
        print("true_mask_list")
        print(true_mask_list)
        print(f"Loading model from {load_model_path} for evaluation...")
        checkpoint = torch.load(load_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if explain_type=="ig":
            
            atom_importances = evaluate_ig(
                                model, test_loader, device,
                                n_steps=50, target_class=1,backbone=backbone
                            )
                            
            #results = [calculate_metrics(m,n,explain_type) for m,n in zip(atom_importances,true_mask_list)]
            results = [calculate_metrics(m, n, explain_type, top_k_ratio)
                       for m, n in zip(atom_importances, true_mask_list)]
        #val_score  = evaluate(model, val_loader,  task_type, device)   # ★ 新增
        elif explain_type in ["BACPOOL","global_attention"]:
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
        #print(results)
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
    parser.add_argument("--explain_type", type=str,default="BACPOOL",choices=['BACPOOL','ig','global_attention',"pgexplainer"])
    args = parser.parse_args()
    BATCH_SIZE =32
    pool = args.pooltype  # 固定 Sum 池化
    explain_type=args.explain_type
    datasets_cfg = {
        'mutagenicity': {"csv": "/data1/nibingwei/fieldgnn/data/Mutagenicity_interpretable/mutagenicity_test_alert_final.csv", "smiles": "smiles", "label": "mutagenicity", "task": "classification"},#
        "Aoral":{"csv": "./data/Aoral_interpretable/matched_alerts_Aoral.csv", "smiles": "SMILES", "label": "Labels", "task": "classification"},
    }
    seed_list = [0,8,42, 67,123,322,999,2024,1001,4239]#,42, 67,123,322,999,2024,1001,4239
    time="111001"
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
        print("model path")
        print(model_path)
        print(f"model_path:{model_path}")
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
        per_seed_csv = f"{args.backbone}_{pool}_seed_eval_{BATCH_SIZE}_{explain_type}_{time}.csv"
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
    '''    
    METRIC_KEYS = ["ACC", "Precision", "Recall", "FPR", "AUC_macro"]
    mean_metrics = {
        k: float(np.mean([seed_results[s][k] for s in seed_results]))
        for k in METRIC_KEYS
    }
    std_metrics = {
        k: float(np.std([seed_results[s][k] for s in seed_results], ddof=ddof))
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
    '''
    METRIC_KEYS = ["ACC", "Precision", "Recall", "FPR", "AUC_macro"]
    n_seeds = len(seed_results)
    ddof = 1 if n_seeds > 1 else 0

    mean_metrics = {
        k: float(np.mean([seed_results[s][k] for s in seed_results]))
        for k in METRIC_KEYS
    }
    std_metrics = {
        k: float(np.std([seed_results[s][k] for s in seed_results], ddof=ddof))
        for k in METRIC_KEYS
    }

    # ---- 只存这一个 CSV：每个指标一列，格式 "mean±std" ----
    summary_csv = f"{args.backbone}_explain_summary_{time}.csv"#{args.backbone}_{pool}_{explain_type}_summary.csv
    csv_fieldnames = [
        "dataset", "backbone", "pool", "explain_type", "n_seeds",
        "ACC", "Precision", "Recall", "FPR", "AUC_macro",
    ]

    row = {
        "dataset":      args.dataset,
        "backbone":     args.backbone,
        "pool":         pool,
        "explain_type": explain_type,
        "n_seeds":      n_seeds,
    }
    for k in METRIC_KEYS:
        row[k] = f"{trunc_fmt(mean_metrics[k], 4)}\u00b1{trunc_fmt(std_metrics[k], 4)}"

    file_exists = os.path.isfile(summary_csv) and os.path.getsize(summary_csv) > 0
    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()