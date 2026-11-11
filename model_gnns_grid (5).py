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
    load_fold_data,evaluate,train_epoch
)
# ================================================================






# ==================== 单折实验 ====================
def run_one_fold(train_smi, train_y, val_smi, val_y, test_smi, test_y,
                 task_type, hidden_dim, num_layers,
                 dropout, use_layernorm, num_classes, epochs, seed,
                 batch_size=128, save_path=None, dataset_name="dataset",
                 fold_idx=0, backbone="gin",pool_type=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)  # 先 warn_only 试试    
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    elif pool_type == "brics_frag":
        pooling = BRICSFragmentPool(
            in_channels=hidden_dim,
            hidden=hidden_dim // 2,
            dropout=dropout
        )
    elif pool_type == "brics_frag_original_23":
        pooling = BRICSFragmentPoolOriginal(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            aggr='attention',
            gate_hidden=hidden_dim // 2,model_type=backbone
        )
    elif pool_type == "global_atom":
        pooling = GlobalAtomPool(
            in_channels=hidden_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
            model_type=backbone,
        )
    elif pool_type == "brics_atom_fusion":
        pooling = BRICSAtomFusionPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
        )
    elif pool_type == "brics_atom_concat_fusion":
        pooling = BRICSAtomConcatFusionPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
        )
    elif pool_type == "brics_atom_cross_attn_fusion":
        pooling = BRICSAtomCrossAttnFusionPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=4,
            ffn_ratio=2,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
        )
    elif pool_type == "brics_atom_cross_attn_fusion_v2":
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
            align_weight=0,   # 0.05
        )    
    elif pool_type == "brics_atom_align_attn":
        pooling = BRICSAtomAlignAttnPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=1,
            ffn_ratio=1,
            align_weight=0.05,
            gate_init=-1.0,
        )
    elif pool_type == "brics_atom_cross_attn_fusion_signed":
        pooling = BRICSAtomCrossAttnFusionSignedPool(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_gnn_layers=1,
            dropout=dropout,
            use_layernorm=use_layernorm,
            gate_hidden=hidden_dim // 2,
            model_type=backbone,
            n_heads=4,
            ffn_ratio=2,
            alpha_init=-2.0,
            alpha_max=1.0,
            alpha_reg=0.01,
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
    elif pool_type == "brics_atom_cross_attn_fusion_signed_align":
        pooling = BRICSAtomCrossAttnFusionSignedAlignPool(
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
            de_corr_weight=0.05,
            align_weight=0.05,
            signed_attn_temperature_init=1.0,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    if task_type == "multitask":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif task_type in ["classification", "multiclass"]:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if task_type in ["regression"] else "max",
        factor=0.7, patience=20
    )

    best_val = 0.0 if task_type in ["classification", "multiclass", "multitask"] else float("inf")
    best_state, wait, patience = None, 0, 30
    metric_name = "AUC" if task_type in ["classification", "multiclass", "multitask"] else "RMSE"

    for epoch in range(1, epochs + 1):
        model.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, task_type, device)
        val_score = evaluate(model, val_loader, task_type, device)
        if math.isnan(val_score):
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val {metric_name}=nan")
        else:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val {metric_name}={val_score:.4f}")
        scheduler.step(val_score)
        is_better = (val_score > best_val) if task_type in ["classification", "multiclass", "multitask"] else (val_score < best_val)
        if is_better:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    test_score = evaluate(model, test_loader, task_type, device)
    print(f"Fold Test {metric_name}: {test_score:.4f} (Val Best: {best_val:.4f})")
    return test_score, best_val, best_state


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pooltype", type=str, default="sum",
                    choices=["sum","max","mean","meanmax","brics_frag_original_23",\
                             "brics_global_context","brics_atom_dual","brics_atom_align",\
                              "brics_contrastive","global_atom","brics_atom_fusion","brics_atom_concat_fusion","brics_atom_cross_attn_fusion",\
                              "brics_atom_cross_attn_fusion_simple","topk","sag","set2set","global_attn_weight","diffpool_sum","mincutpool_sum","edgepool",\
                              "brics_atom_adaptive_residual","causal_brics_atom_cross_attn_convex","brics_atom_cross_attn_fusion_v2","brics_atom_align_attn",\
                              "brics_atom_cross_attn_fusion_signed","gmt","asap","brics_atom_cross_attn_fusion_signed_align",\
                              "brics_atom_cross_attn_fusion_v2_align","brics_atom_cross_attn_fusion_v2_align_nodecorr"])
    parser.add_argument("--backbone", type=str, default="gcn",
                        choices=["gine", "gin", "gcn", "gat","gatv2"],
                        help="GNN backbone: gine / gin / gcn / gat")
    args = parser.parse_args()
    if args.dataset=='hiv':
        BATCH_SIZE = 128
    else:
        BATCH_SIZE = 32
    print(f"batchsize:{BATCH_SIZE}")
    pool = args.pooltype  # 固定 Sum 池化

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
    seed=[516]
    for seed_id in seed:
        model_save_path = f"./models_best_ogb_layernorm_{seed_id}_testnode1"
        if args.dataset not in datasets_cfg:
            raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(datasets_cfg.keys())}")

        cfg = datasets_cfg[args.dataset]
        task_type = cfg["task"]
        csv_path = cfg["csv"]
        smiles_col = cfg["smiles"]
        label_col = cfg.get("label", None)
        label_cols = cfg.get("label_cols", None)

        param_combinations = list(product([128], [2,3], [0.1, 0.3], [True,False]))#mincutpool：list(product([128,256], [2,3], [0.1, 0.3], [True])),mincut_toxcast:product([128,256], [2], [0.1, 0.3,0.5], [True]

        result_csv = f"{args.backbone}_{pool}_results_{BATCH_SIZE}_{seed_id}_node2_ogb_layernorm_testnode1.csv"
        os.makedirs(model_save_path, exist_ok=True)

        fieldnames = ["dataset", "backbone", "pool", "hidden_dim", "num_layers", "dropout",
                      "use_layernorm", "seed", "mean", "std", "mean_val"]
        for i in range(1,2):
            fieldnames.append(f"fold{i}_test")
        for i in range(1,2):
            fieldnames.append(f"fold{i}_val")

        write_header = not os.path.exists(result_csv)
        with open(result_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            best_avg_val = -float('inf') if task_type in ["classification", "multiclass", "multitask"] else float('inf')
            best_combo_states = None
            best_hyper = ""
            best_hidden_dim = None
            best_num_layers = None
            best_dropout = None
            best_use_layernorm = None
            best_fold_val_scores = []

            for h, l, d, ln in param_combinations:
                print(f"\n=== Backbone: {args.backbone}, Pool: {pool}, dim={h}, layers={l}, dropout={d}, LN={ln} ===")
                fold_test_scores = []
                fold_val_scores = []
                fold_best_states = []

                for fold in range(1,2):
                    fold_col = f"fold{fold}_split"
                    train_smi, train_y, val_smi, val_y, test_smi, test_y, task_type_out, num_classes = load_fold_data(
                        csv_path, smiles_col, label_col=label_col, label_cols=label_cols,
                        fold_col=fold_col, task_type=task_type
                    )
                    test_score, best_val, best_state = run_one_fold(
                        train_smi, train_y, val_smi, val_y, test_smi, test_y,
                        task_type_out, h, l, d, ln, num_classes,
                        args.epochs, seed_id,
                        batch_size=BATCH_SIZE,
                        save_path=None,
                        dataset_name=args.dataset,
                        fold_idx=fold,
                        backbone=args.backbone,pool_type=pool
                    )
                    fold_test_scores.append(test_score)
                    fold_val_scores.append(best_val)
                    fold_best_states.append(best_state)

                mean_test = np.mean(fold_test_scores)
                std_test = np.std(fold_test_scores)
                mean_val = np.mean(fold_val_scores)
                print(f"Mean Test: {mean_test:.4f} ± {std_test:.4f}, Mean Val: {mean_val:.4f}")

                is_better = False
                if task_type_out in ["classification", "multiclass", "multitask"]:
                    if mean_val > best_avg_val:
                        is_better = True
                else:
                    if mean_val < best_avg_val:
                        is_better = True

                if is_better:
                    best_avg_val = mean_val
                    best_combo_states = fold_best_states
                    best_hyper = f"h{h}_l{l}_d{d}_ln{ln}"
                    best_hidden_dim = h
                    best_num_layers = l
                    best_dropout = d
                    best_use_layernorm = ln
                    best_fold_val_scores = fold_val_scores[:]

                row = {
                    "dataset": args.dataset,
                    "backbone": args.backbone,
                    "pool": pool,
                    "hidden_dim": h,
                    "num_layers": l,
                    "dropout": d,
                    "use_layernorm": ln,
                    "seed": seed_id,
                    "mean": mean_test,
                    "std": std_test,
                    "mean_val": mean_val
                }
                for i in range(0,1):
                    row[f"fold{i+1}_test"] = fold_test_scores[i]
                    row[f"fold{i+1}_val"] = fold_val_scores[i]

                writer.writerow(row)
                f.flush()

            if best_combo_states is not None:
                for fold_idx, state in enumerate(best_combo_states):
                    model_filename = f"{args.dataset}_{args.backbone}_{pool}_batch{BATCH_SIZE}_best_fold{fold_idx+1}.pt"
                    save_path = os.path.join(model_save_path, model_filename)

                    model_info = {
                        'state_dict': state,
                        'backbone': args.backbone,
                        'pool_type': pool,
                        'hidden_dim': best_hidden_dim,
                        'num_layers': best_num_layers,
                        'dropout': best_dropout,
                        'use_layernorm': best_use_layernorm,
                        'batch_size': BATCH_SIZE,
                        'epochs': args.epochs,
                        'seed': seed_id,
                        'dataset': args.dataset,
                        'fold': fold_idx + 1,
                        'val_score': best_fold_val_scores[fold_idx] if fold_idx < len(best_fold_val_scores) else None,
                        'task_type': task_type_out,
                        'in_dim': None,
                        'out_ch': 2 if task_type_out == "classification" else (num_classes if task_type_out in ["multiclass", "multitask"] else 1)
                    }
                    torch.save(model_info, save_path)
                    print(f"Best model ({args.backbone}/{pool}) fold {fold_idx+1} saved to {save_path}")
                print(f"Best hyper-params for {args.backbone}/{pool}: {best_hyper}, avg_val={best_avg_val:.4f}")

        print(f"\nAll experiments completed. Best models saved in .{model_save_path}")


if __name__ == "__main__":
    main()