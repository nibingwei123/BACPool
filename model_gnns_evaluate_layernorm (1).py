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
def trunc_fmt(x, n=4):
    """截断到 n 位小数（不四舍五入）；nan/None 返回 'nan'"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    d = Decimal(float(x)).quantize(Decimal(10) ** -n, rounding=ROUND_DOWN)
    return f"{d:.{n}f}"
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

def run_one_fold(train_smi, train_y, val_smi, val_y, test_smi, test_y,
                 task_type, hidden_dim, num_layers,
                 dropout, use_layernorm, num_classes, epochs, seed,
                 batch_size=128, save_path=None, dataset_name="dataset",
                 fold_idx=0, backbone="gin",pool_type=None,load_model_path=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # 先 warn_only 试试    
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    train_env = generate_env_labels(train_smi)
    val_env = generate_env_labels(val_smi)
    test_env = generate_env_labels(test_smi)

    cache_dir = "./feature_cache_ogb"#sum,mean,max:./feature_cache_central
    train_cache = os.path.join(cache_dir, f"{dataset_name}_fold{fold_idx}_train.pt")
    val_cache = os.path.join(cache_dir, f"{dataset_name}_fold{fold_idx}_val.pt")
    test_cache = os.path.join(cache_dir, f"{dataset_name}_fold{fold_idx}_test.pt")

    train_set = MoleculeDataset(train_smi, train_y, task_type, "train", env_labels=train_env, cache_path=train_cache, recal=False)
    val_set = MoleculeDataset(val_smi, val_y, task_type, "val", env_labels=val_env, cache_path=val_cache, recal=False)
    test_set = MoleculeDataset(test_smi, test_y, task_type, "test", env_labels=test_env, cache_path=test_cache, recal=False)

    from torch.utils.data import DataLoader as TorchDataLoader
    train_loader = TorchDataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   collate_fn=my_collate, drop_last=False, num_workers=0)
    val_loader = TorchDataLoader(val_set, batch_size=batch_size, shuffle=False,
                                 collate_fn=my_collate, num_workers=0)
    test_loader = TorchDataLoader(test_set, batch_size=batch_size, shuffle=False,
                                  collate_fn=my_collate, num_workers=0)

    out_ch = 2 if task_type == "classification" else (num_classes if task_type in ["multiclass", "multitask"] else 1)
    convs = [hidden_dim] * num_layers
    norm = torch.nn.LayerNorm if use_layernorm else None
    in_dim = train_set[0].x.size(1)
    if pool_type=="sum":
        pooling = SumPooling(hidden_dim)
    elif pool_type=="max":
        pooling = MaxPooling(hidden_dim)
    elif pool_type=="mean":
        pooling = MeanPooling(hidden_dim)
    elif pool_type=="meanmax":
        pooling = MeanMaxPooling(hidden_dim)
    elif pool_type == "global_atom":
        pooling = GlobalAtomPool(
            in_channels=hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
            model_type=backbone,
        )
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
            de_corr_weight=0,#0.05
            align_weight=0.05,   # 0.05
        )
        
    elif pool_type == "brics_atom_cross_attn_fusion_v2_align":
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
            de_corr_weight=0.05,#0.05
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
    if pool_type in ["sag", "topk","asap"]:
        model = GNN_Base_HierarchicalSAGTopK(
            in_channels=in_dim,
            out_channels=out_ch,
            edge_dim=10,
            pool_type=pool_type,
            ratio=0.7,
            convs_layers=convs,
            dense_layers=[128, 64],
            batch_norms=norm,
            dropout_p=dropout,
            backbone=backbone,
            heads=4,
        ).to(device)
    elif pool_type == "edgepool":
        model = GNN_Base_HierarchicalEdgePool(
            in_channels=in_dim,
            out_channels=out_ch,
            edge_dim=10,
            ratio=0.5,                # 每层保留 50% 的边/节点
            convs_layers=convs,
            dense_layers=[128, 64],
            batch_norms=norm,
            dropout_p=dropout,
            backbone=backbone,
            heads=4,
        ).to(device)
    elif pool_type == "diffpool_sum":
        model = DiffPoolModel(
            in_channels=in_dim,
            out_channels=out_ch,
            edge_dim=10,
            hidden_channels=hidden_dim,
            max_node=150,
            ratio=0.3,
            num_layers=num_layers,      # ← 用你的超参数
            min_clusters=2,
            dense_layers=[128, 64],
            dropout=dropout,
            norm=norm,
            link_weight=1.0,
            entropy_weight=0.1,
            readout='sum',
            backbone=backbone,             # 兼容传参，内部未用
        ).to(device)
    elif pool_type == "mincutpool_sum":
        model = MinCutPoolModel(
            in_channels=in_dim,
            out_channels=out_ch,
            edge_dim=10,
            hidden_channels=hidden_dim,
            max_node=150,
            ratio=0.5,
            num_layers=num_layers,     # ★ 用 num_layers，池化层数 = num_layers - 1
            min_clusters=2,
            dense_layers=[128, 64],
            dropout=dropout,
            norm=norm,                 # ★ 传入 LayerNorm 或 None（只作用于 embed）
            mincut_weight=0.1,
            ortho_weight=0.1,
            temp=1.0,
            backbone=backbone,
        ).to(device)  
    else:
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
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        model.to(device)
        # ==================== ★ check_bn0 结束 ★ ====================
        # ==================== ★ batch 依赖检测 ★ ====================
        # ① 快速检测：5 个样本，单独 vs 混 batch
        
        max_diff = check_batch_dependency(
            model, test_set, device,
            n_candidates=5, n_others=31,
            tag=f"fold={fold_idx}, seed={seed}",
        )

        # ② 如果差异大，逐层定位（第 0 个样本）
        # ② 如果差异大，逐层定位（第 0 个样本）
        if max_diff > 1e-3:                              # ← 保持 1e-3
            print("same sample, layer-wise diff")
            b1 = my_collate([test_set[0]]).to(device)    # ← "cpu" 改 device
            b2 = my_collate([test_set[0]]).to(device)    # ← "cpu" 改 device
            n0_atoms = b1.x.size(0)
            n0_frags = int(torch.unique(b1.true_frag_ids).numel())   # ← 真实片段数
            compare_batch_layers(model, b1, b2,
                                 n0_atoms=n0_atoms, n0_frags=n0_frags,
                                 tag=f"fold={fold_idx}, seed={seed}")
        # ==================== ★ batch 依赖检测结束 ★ ====================
        
        test_score = evaluate(model, test_loader, task_type, device)
        val_score  = evaluate(model, val_loader,  task_type, device)
        return test_score, val_score, None      
    else:
        raise ValueError(f"load_model_path is None, 无法进行评估。"
                         f"请检查传入的模型路径: {load_model_path}")


def check_batch_dependency(model, test_set, device, n_candidates=5, n_others=31, tag=""):
    """
    检测单个样本的预测是否受 batch 组成影响。
    对 n_candidates 个样本，分别：
      ① 单独 forward（batch_size=1）
      ② 和另外 n_others 个样本一起 forward（batch_size=n_others+1）
    对比同一位置的预测值。
    """
    print(f"\n{'='*72}")
    print(f"[BatchDep] {tag}  检测 batch 组成对预测的影响")
    print(f"{'='*72}")

    model.eval()
    model.to(device)

    # 取候选样本下标（避免重复）
    idx_list = list(range(min(n_candidates, len(test_set))))
    max_diff_all = 0.0

    for idx in idx_list:
        # ① 单独
        b1 = my_collate([test_set[idx]]).to(device)
        with torch.no_grad():
            p1, _ = model(b1.x, b1.edge_index, b1.edge_attr, b1.batch, data_batch=b1)
        pred_single = p1[0].detach().clone()

        # ② 混 batch：目标样本 + 其他样本
        others = [test_set[i] for i in range(len(test_set)) if i != idx][:n_others]
        data_list = [test_set[idx]] + others
        b2 = my_collate(data_list).to(device)

        # 确认目标样本在 b2 的第 0 位
        # 确认目标样本在 b2 的第 0 位
        # 兼容：单任务标量 y / 多任务向量 y / 含 NaN 的标签
        _y_b2 = b2.y[0].float().cpu()
        _y_ts = test_set[idx].y.float().cpu()
        assert torch.allclose(_y_b2, _y_ts, equal_nan=True), \
            f"b2.y[0] != test_set[{idx}].y  " \
            f"(shape={tuple(_y_b2.shape)} vs {tuple(_y_ts.shape)})"
        n0 = b1.x.size(0)
        assert b2.x[:n0].equal(b1.x), f"b2.x[:{n0}] != b1.x"

        with torch.no_grad():
            p2, _ = model(b2.x, b2.edge_index, b2.edge_attr, b2.batch, data_batch=b2)
        pred_mixed = p2[0].detach().clone()

        diff = (pred_single - pred_mixed).abs().max().item()
        max_diff_all = max(max_diff_all, diff)

        print(f"  sample {idx:3d}: batch=1 → {pred_single.tolist()}")
        print(f"              batch={len(data_list):3d} → {pred_mixed.tolist()}")
        print(f"              diff = {diff:.10f}")

    print(f"\n  max diff over {len(idx_list)} samples = {max_diff_all:.10f}")
    if max_diff_all < 1e-6:
        print(f"  ✅ 预测与 batch 组成无关（阈值 1e-6）")
    elif max_diff_all < 1e-3:
        print(f"  ⚠️ 差异 {max_diff_all:.2e}，可能是浮点噪声，建议再测几个样本")
    else:
        print(f"  ❌ 差异 {max_diff_all:.2e}，预测受 batch 组成影响，需排查")
    print(f"{'='*72}\n")

    model.to(device)
    return max_diff_all


def compare_batch_layers(model, b1, b2, n0_atoms, n0_frags=2, tag=""):
    """
    逐层对比两个 batch 的中间输出，找第一层出现差异的位置。
    只在 check_batch_dependency 发现差异时调用。
    """
    print(f"\n{'='*110}")
    print(f"[LayerDiff] {tag}  逐层对比")
    print(f"{'='*110}")

    outs_1, outs_2 = {}, {}
    hooks = []

    def make_hook(store, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                store[name] = out.detach().clone()
            elif isinstance(out, tuple):
                store[name] = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in out)
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(make_hook(outs_1, name)))
    with torch.no_grad():
        model(b1.x, b1.edge_index, b1.edge_attr, b1.batch, data_batch=b1)
    for h in hooks: h.remove()

    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(make_hook(outs_2, name)))
    with torch.no_grad():
        model(b2.x, b2.edge_index, b2.edge_attr, b2.batch, data_batch=b2)
    for h in hooks: h.remove()

    print(f"{'状态':<4s} {'层名':<50s} {'shape_1':<20s} {'shape_2':<20s} {'max_diff':<14s}")
    print("-" * 110)
    first_bad = None
    for name in outs_1:
        if name not in outs_2:
            continue
        o1, o2 = outs_1[name], outs_2[name]
        if not (isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor)):
            continue
        try:
            if o1.dim() >= 1 and o2.dim() >= 1 and o1.size(0) >= n0_atoms and o2.size(0) >= n0_atoms and o1.size(-1) == o2.size(-1):
                d = (o1[:n0_atoms] - o2[:n0_atoms]).abs().max().item()
            elif o1.dim() >= 1 and o2.dim() >= 1 and o1.size(0) >= n0_frags and o2.size(0) >= n0_frags and o1.size(-1) == o2.size(-1):
                d = (o1[:n0_frags] - o2[:n0_frags]).abs().max().item()
            elif o1.shape == o2.shape and o1.size(0) >= 1:
                d = (o1[0] - o2[0]).abs().max().item()
            else:
                print(f"⚠️  {name:<50s} {str(tuple(o1.shape)):<20s} {str(tuple(o2.shape)):<20s} shape_mismatch")
                continue
        except Exception as e:
            print(f"⚠️  {name:<50s} compare_error: {e}")
            continue
        flag = "❌" if d > 1e-6 else "✅"
        print(f"{flag} {name:<50s} {str(tuple(o1.shape)):<20s} {str(tuple(o2.shape)):<20s} {d:.10f}")
        if d > 1e-6 and first_bad is None:
            first_bad = name

    print("-" * 110)
    if first_bad:
        print(f"❌ 第一层出现差异: {first_bad}")
    else:
        print("✅ 所有层都一致")
    print(f"{'='*110}\n")        
        
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

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pooltype", type=str, default="sum",
                    choices=["sum","max","mean","topk","sag","set2set","global_attn_weight","diffpool_sum","mincutpool_sum","edgepool",\
                             "brics_atom_cross_attn_fusion_signed","brics_atom_cross_attn_fusion_v2_align_nodecorr","gmt","asap",\
                             "ablation_v2align_frag_only","ablation_v2align_atom_only","ablation_v2align_fused_no_align","ablation_v2align_concat"])
    parser.add_argument("--backbone", type=str, default="gcn",
                        choices=["gine", "gin", "gcn", "gat","gatv2"],
                        help="GNN backbone: gine / gin / gcn / gat")
    args = parser.parse_args()
    args = parser.parse_args()
    if args.dataset=='hiv':
        BATCH_SIZE = 128
    else:
        BATCH_SIZE = 32
    pool = args.pooltype  # 固定 Sum 池化
    test_time="111001"
    datasets_cfg = {
        "bbbp": {"csv": "./data/bbbp_with_split.csv", "smiles": "smiles", "label": "p_np", "task": "classification"},
        "bace": {"csv": "./data/bace_with_split.csv", "smiles": "smiles", "label": "Class", "task": "classification"},
        "hiv": {"csv": "./data/hiv_with_split.csv", "smiles": "smiles", "label": "HIV_active", "task": "classification"},
        "clintox": {"csv": "./data/clintox_with_split.csv", "smiles": "smiles", "label": "CT_TOX", "task": "classification"},
        "esol": {"csv": "./data/esol_with_split.csv", "smiles": "smiles", "label": "measured log solubility in mols per litre", "task": "regression"},
        "freesolv": {"csv": "./data/freesolv_with_split.csv", "smiles": "smiles", "label": "expt", "task": "regression"},
        "lipophilicity": {"csv": "./data/lipophilicity_with_split.csv", "smiles": "smiles", "label": "exp", "task": "regression"},
        "qm7": {"csv": "./data/qm7_with_split.csv", "smiles": "smiles", "label": "u0_atom", "task": "regression"},
        "tox21": {"csv": "./data/tox21_with_split.csv", "smiles": "smiles", "label_cols": ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], "task": "multitask"},
        "toxcast": {"csv": "./data/toxcast_with_split.csv", "smiles": "smiles", "label_cols": None, "task": "multitask"},
        "muv": {"csv": "./data/muv_with_split.csv", "smiles": "smiles", "label_cols": ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'], "task": "multitask"},
        "clintox_multi": {"csv": "./data/clintox_with_split.csv", "smiles": "smiles", "label_cols": ["FDA_APPROVED", "CT_TOX"], "task": "multitask"},
        "qm8": {"csv": "./data/qm8_with_split.csv", "smiles": "smiles", "label_cols": ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"], "task": "multitask"},
        "Caco2_Wang": {"csv": "./adme_splits_under5k/Caco2_Wang.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "PAMPA_NCATS": {"csv": "./adme_splits_under5k/PAMPA_NCATS.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "HIA_Hou": {"csv": "./adme_splits_under5k/HIA_Hou.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "Pgp_Broccatelli": {"csv": "./adme_splits_under5k/Pgp_Broccatelli.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "Bioavailability_Ma": {"csv": "./adme_splits_under5k/Bioavailability_Ma.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "Lipophilicity_AstraZeneca": {"csv": "./adme_splits_under5k/Lipophilicity_AstraZeneca.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "HydrationFreeEnergy_FreeSolv": {"csv": "./adme_splits_under5k/HydrationFreeEnergy_FreeSolv.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "BBB_Martins": {"csv": "./adme_splits_under5k/BBB_Martins.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "PPBR_AZ": {"csv": "./adme_splits_under5k/PPBR_AZ.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "VDss_Lombardo": {"csv": "./adme_splits_under5k/VDss_Lombardo.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "CYP2C9_Substrate_CarbonMangels": {"csv": "./adme_splits_under5k/CYP2C9_Substrate_CarbonMangels.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "CYP2D6_Substrate_CarbonMangels": {"csv": "./adme_splits_under5k/CYP2D6_Substrate_CarbonMangels.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "CYP3A4_Substrate_CarbonMangels": {"csv": "./adme_splits_under5k/CYP3A4_Substrate_CarbonMangels.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "Half_Life_Obach": {"csv": "./adme_splits_under5k/Half_Life_Obach.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "Clearance_Hepatocyte_AZ": {"csv": "./adme_splits_under5k/Clearance_Hepatocyte_AZ.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "Clearance_Microsome_AZ": {"csv": "./adme_splits_under5k/Clearance_Microsome_AZ.csv", "smiles": "Drug", "label": "Y", "task": "regression"},
        "sider": {"csv": "./data/sider_with_split.csv", "smiles": "smiles", "label_cols": None, "task": "multitask"},
        "ames": {"csv": "./data/ames_with_split.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        "herg": {"csv": "./data/herg_with_split.csv", "smiles": "Drug", "label": "Y", "task": "classification"},
        'mutagenicity': {"csv": "./data/mutagenicity_with_split.csv", "smiles": "smiles", "label": "mutagenicity", "task": "classification"},
        "Aoral":{"csv": "./data/Aoral_with_split.csv", "smiles": "SMILES", "label": "Labels", "task": "classification"},
    }
    seed_list = [0,42, 123, 2024,8,67,322,999,1001,4239]#0,42, 123, 2024,8,67,322,999,1001,4239

    cfg = datasets_cfg[args.dataset]
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
    
    seed_score_map = {}
    seed_meta_map  = {}
    seed_val_map   = {}      # ★ 新增：评估得到的 val_score
    for seed_id in seed_list:
        model_dir = f"./models_best_ogb_layernorm_{seed_id}"#./models_best_ogb_new_ablation_nodecorr_{seed_id}
        model_filename = f"{args.dataset}_{args.backbone}_{pool}_batch{BATCH_SIZE}_best_fold1.pt"
        
        model_path = os.path.join(model_dir, model_filename)
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

        # ★ 从 checkpoint 读取训练时保存的 best val score
        #   兼容多种 key 命名，读不到就 None
        ckpt_val_score = model_info.get('val_score')

        seed_meta_map[seed_id] = {
            'hidden_dim':     h,
            'num_layers':     l,
            'dropout':        d,
            'use_layernorm':  ln,
            'batch_size':     bs,
            'ckpt_val_score': ckpt_val_score,   # ★ 新增
        }
        
        
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
        
        #print("train_smi")
        #print(train_smi)
        #print(train_smi[0].shape)
        #print(train_smi[0:10])
        #print("train_y")
        #print(train_y)
        #print(train_y[0].shape)
        #print(train_y[0:10])                    
        #print("valid smi")
        #print(valid_smi)
        #print(valid_smi[0].shape)
        #print(val_smi[0:10])                    
        #print("valid y")
        #print(valid_y)
        #print(valid_y[0].shape)
        #print(val_y[0:10])                    
        #print("test smi")
        #print(test_smi)                 
        #print(test_smi[0].shape)
        #print(test_smi[0:10])  
        #print("test_y")
        #print(test_y)                       
        #print(test_y[0].shape)
        #print(test_y[0:10])
        
        # 纯评估
        test_score, val_score_eval, _ = run_one_fold(     # ★ 改名接收
            train_smi, train_y, val_smi, val_y, test_smi, test_y,
            task_type_out, h, l, d, ln, num_classes_arg,
            0, seed_id,
            batch_size=BATCH_SIZE,
            save_path=None,
            dataset_name=args.dataset,
            fold_idx=1,
            backbone=args.backbone,
            pool_type=pool,
            load_model_path=model_path
        )

        seed_score_map[seed_id] = test_score        # ★ 用 dict 记录
        seed_val_map[seed_id]   = val_score_eval          
        print(f"[Seed {seed_id}] Test score = {test_score:.4f}")
    '''
    # ==================== 汇总 + 写 CSV ====================
    if seed_score_map:
        scores     = list(seed_score_map.values())
        assert len(scores) == 10, f"期望 scores 长度为 10，但实际为 {len(scores)}"
        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores))

        # val 的均值（缺失的 seed 跳过）
        val_scores = [v for v in seed_val_map.values() if v is not None]
        assert len(val_scores) == 10, f"期望 scores 长度为 10，但实际为 {len(val_scores)}"
        val_mean   = float(np.mean(val_scores)) if val_scores else float('nan')
        val_std    = float(np.std(val_scores))  if val_scores else float('nan')

        # 控制台打印
        print("\n" + "="*60)
        print(f"Results across {len(scores)} seeds:")
        print(f"Test scores: {[f'{s:.4f}' for s in scores]}")
        print(f"Test Mean = {mean_score:.4f}")
        print(f"Test Std  = {std_score:.4f}")
        print(f"Val  Mean = {val_mean:.4f}")
        print(f"Val  Std  = {val_std:.4f}")
        print("="*60)

        # ============================================================
        # ① per-seed CSV：加两列 ckpt_val_score / val_score_eval
        # ============================================================
        per_seed_csv = f"{args.backbone}_{pool}_seed_eval_{BATCH_SIZE}_{test_time}_layernorm.csv"
        per_seed_fields = [
            "dataset", "backbone", "pool", "seed",
            "hidden_dim", "num_layers", "dropout", "use_layernorm", "batch_size",
            "ckpt_val_score",    # ★ checkpoint 里存的 val
            "val_score_eval",    # ★ 重新评估得到的 val
            "test_score",
            "mean_score",
        ]
        file_exists = os.path.isfile(per_seed_csv) and os.path.getsize(per_seed_csv) > 0
        with open(per_seed_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_seed_fields)
            if not file_exists:
                writer.writeheader()
            for seed_id in seed_list:
                if seed_id not in seed_score_map:
                    raise ValueError("seed_id not in seed_score_map")
                    continue
                meta = seed_meta_map[seed_id]
                ckpt_val = meta['ckpt_val_score']
                val_eval = seed_val_map.get(seed_id, None)
                writer.writerow({
                    "dataset":         args.dataset,
                    "backbone":        args.backbone,
                    "pool":            pool,
                    "seed":            seed_id,
                    "hidden_dim":      meta['hidden_dim'],
                    "num_layers":      meta['num_layers'],
                    "dropout":         meta['dropout'],
                    "use_layernorm":   meta['use_layernorm'],
                    "batch_size":      meta['batch_size'],
                    "ckpt_val_score":  "" if ckpt_val is None else f"{ckpt_val:.6f}",
                    "val_score_eval":  "" if val_eval is None else f"{val_eval:.6f}",
                    "test_score":      f"{seed_score_map[seed_id]:.6f}",
                    "mean_score":      f"{mean_score:.6f}",
                })
        print(f"Per-seed results appended to {per_seed_csv}")
        
        # ============================================================
        # ② test mean 矩阵（原有）
        # ============================================================
        mean_csv = f"{args.backbone}_seed_eval_test_mean_new_{test_time}_layernorm.csv"
        if os.path.isfile(mean_csv) and os.path.getsize(mean_csv) > 0:
            df = pd.read_csv(mean_csv, index_col=0)
        else:
            df = pd.DataFrame()
        if args.dataset not in df.columns:
            df[args.dataset] = np.nan
        if pool not in df.index:
            df.loc[pool] = np.nan
        df.loc[pool, args.dataset] = mean_score
        df.to_csv(mean_csv, float_format="%.6f")
        print(f"Test mean matrix updated in {mean_csv}")
        
        # ============================================================
        # ③ val mean 矩阵（新增）：行 = pool，列 = dataset
        # ============================================================
        val_mean_csv = f"{args.backbone}_seed_eval_val_mean_new_{test_time}_layernorm.csv"
        if os.path.isfile(val_mean_csv) and os.path.getsize(val_mean_csv) > 0:
            df_val = pd.read_csv(val_mean_csv, index_col=0)
        else:
            df_val = pd.DataFrame()
        if args.dataset not in df_val.columns:
            df_val[args.dataset] = np.nan
        if pool not in df_val.index:
            df_val.loc[pool] = np.nan
        df_val.loc[pool, args.dataset] = val_mean
        df_val.to_csv(val_mean_csv, float_format="%.6f")
        print(f"Val mean matrix updated in {val_mean_csv}")
                # ============================================================
        # ④ test std 矩阵（新增）
        # ============================================================
        test_std_csv = f"{args.backbone}_seed_eval_test_std_new_{test_time}_laynorm.csv"
        if os.path.isfile(test_std_csv) and os.path.getsize(test_std_csv) > 0:
            df_test_std = pd.read_csv(test_std_csv, index_col=0)
        else:
            df_test_std = pd.DataFrame()
        if args.dataset not in df_test_std.columns:
            df_test_std[args.dataset] = np.nan
        if pool not in df_test_std.index:
            df_test_std.loc[pool] = np.nan
        df_test_std.loc[pool, args.dataset] = std_score
        df_test_std.to_csv(test_std_csv, float_format="%.6f")
        print(f"Test std matrix updated in {test_std_csv}")

        # ============================================================
        # ⑤ val std 矩阵（新增）
        # ============================================================
        val_std_csv = f"{args.backbone}_seed_eval_val_std_new_{test_time}_layernorm.csv"
        if os.path.isfile(val_std_csv) and os.path.getsize(val_std_csv) > 0:
            df_val_std = pd.read_csv(val_std_csv, index_col=0)
        else:
            df_val_std = pd.DataFrame()
        if args.dataset not in df_val_std.columns:
            df_val_std[args.dataset] = np.nan
        if pool not in df_val_std.index:
            df_val_std.loc[pool] = np.nan
        df_val_std.loc[pool, args.dataset] = val_std
        df_val_std.to_csv(val_std_csv, float_format="%.6f")
        print(f"Val std matrix updated in {val_std_csv}")

    else:
        print("No models were evaluated. Please check the model paths and seeds.")
        raise ValueError("no mean values")
    '''
    
    if seed_score_map:
        scores = list(seed_score_map.values())
        assert len(scores) == 10, f"期望 scores 长度为 10，但实际为 {len(scores)}"

        n_seeds = len(scores)
        ddof    = 1 if n_seeds > 1 else 0          # ★ 样本标准差（n=10 → ddof=1）

        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores, ddof=ddof))          # ★ ddof=1

        val_scores = [v for v in seed_val_map.values() if v is not None]
        assert len(val_scores) == 10, f"期望 val_scores 长度为 10，但实际为 {len(val_scores)}"

        val_mean = float(np.mean(val_scores))
        val_std  = float(np.std(val_scores, ddof=ddof))  # ★ ddof=1

        # ---- 截断 4 位的 mean±std ----
        test_meanstd = f"{trunc_fmt(mean_score, 4)}\u00b1{trunc_fmt(std_score, 4)}"
        val_meanstd  = f"{trunc_fmt(val_mean, 4)}\u00b1{trunc_fmt(val_std, 4)}"

        print("\n" + "=" * 60)
        print(f"Results across {n_seeds} seeds  (ddof={ddof}, 截断 4 位)")
        print(f"Test scores    : {[trunc_fmt(s, 4) for s in scores]}")
        print(f"Test mean±std  : {test_meanstd}")
        print(f"Val  mean±std  : {val_meanstd}")
        print("=" * 60)

        # ============================================================
        # ① per-seed CSV（保留全精度数值 + 新增 std/mean±std 列）
        # ============================================================
        per_seed_csv = f"{args.backbone}_{pool}_seed_eval_{BATCH_SIZE}_{test_time}_layernorm_nonetopk_noshuffle.csv"
        per_seed_fields = [
            "dataset", "backbone", "pool", "seed",
            "hidden_dim", "num_layers", "dropout", "use_layernorm", "batch_size",
            "ckpt_val_score", "val_score_eval", "test_score",
            "mean_score", "std_score", "mean±std",
        ]
        file_exists = os.path.isfile(per_seed_csv) and os.path.getsize(per_seed_csv) > 0
        with open(per_seed_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_seed_fields)
            if not file_exists:
                writer.writeheader()
            for seed_id in seed_list:
                if seed_id not in seed_score_map:
                    raise ValueError("seed_id not in seed_score_map")
                meta = seed_meta_map[seed_id]
                ckpt_val = meta['ckpt_val_score']
                val_eval = seed_val_map.get(seed_id, None)
                writer.writerow({
                    "dataset":         args.dataset,
                    "backbone":        args.backbone,
                    "pool":            pool,
                    "seed":            seed_id,
                    "hidden_dim":      meta['hidden_dim'],
                    "num_layers":      meta['num_layers'],
                    "dropout":         meta['dropout'],
                    "use_layernorm":   meta['use_layernorm'],
                    "batch_size":      meta['batch_size'],
                    "ckpt_val_score":  "" if ckpt_val is None else f"{ckpt_val:.6f}",
                    "val_score_eval":  "" if val_eval is None else f"{val_eval:.6f}",
                    "test_score":      f"{seed_score_map[seed_id]:.6f}",
                    "mean_score":      f"{mean_score:.6f}",
                    "std_score":       f"{std_score:.6f}",
                    "mean±std":        test_meanstd,
                })
        print(f"Per-seed results appended to {per_seed_csv}")
        # ============================================================
        # ③ 新增：mean±std 矩阵（截断 4 位字符串，直接贴论文）
        # ============================================================
        test_meanstd_csv = f"{args.backbone}_seed_eval_test_meanstd_new_{test_time}_layernorm_nonetopk_noshuffle.csv"
        if os.path.isfile(test_meanstd_csv) and os.path.getsize(test_meanstd_csv) > 0:
            df_test_mstd = pd.read_csv(test_meanstd_csv, index_col=0, dtype=str, keep_default_na=False)
        else:
            df_test_mstd = pd.DataFrame()
        if args.dataset not in df_test_mstd.columns:
            df_test_mstd[args.dataset] = ""
        if pool not in df_test_mstd.index:
            df_test_mstd.loc[pool] = ""
        df_test_mstd.loc[pool, args.dataset] = test_meanstd
        df_test_mstd.to_csv(test_meanstd_csv)

        val_meanstd_csv = f"{args.backbone}_seed_eval_val_meanstd_new_{test_time}_layernorm_nonetopk_noshuffle.csv"
        if os.path.isfile(val_meanstd_csv) and os.path.getsize(val_meanstd_csv) > 0:
            df_val_mstd = pd.read_csv(val_meanstd_csv, index_col=0, dtype=str, keep_default_na=False)
        else:
            df_val_mstd = pd.DataFrame()
        if args.dataset not in df_val_mstd.columns:
            df_val_mstd[args.dataset] = ""
        if pool not in df_val_mstd.index:
            df_val_mstd.loc[pool] = ""
        df_val_mstd.loc[pool, args.dataset] = val_meanstd
        df_val_mstd.to_csv(val_meanstd_csv)

        print(f"Mean±Std matrices updated:")
        print(f"  - {test_meanstd_csv}")
        print(f"  - {val_meanstd_csv}")

    else:
        print("No models were evaluated. Please check the model paths and seeds.")
        raise ValueError("no mean values")
if __name__ == "__main__":
    main()