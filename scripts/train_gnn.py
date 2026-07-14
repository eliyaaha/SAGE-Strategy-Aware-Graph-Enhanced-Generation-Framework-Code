import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.hyperparams import *
from src.preprocessing import apply_lexicon_matching, load_and_merge_data, process_labels, create_expert_seeker_pairs, collapse_consecutive_turns,clean_neutral_label
from src.graph_utils import build_hetero_graph
from models.hgt_model import HGTMultiLabel
from src.metrics import tune_thresholds, generate_performance_report

def set_seeds(seed=SEED):
    """Ensure reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


def run_gnn_pipeline():
    set_seeds(SEED)
    
    # 1. Data Loading and Merging
    print(f"--- Loading and merging raw data ---")
    df = load_and_merge_data(RAW_METADATA_PATH, BLOCKED_CONV_PATH)

    print("--- Cleaning neutral label from 'processed' ---")
    df = clean_neutral_label(df)

    print("--- Applying Lexicon matching (GSR) ---")
    df, lexicon_cols = apply_lexicon_matching(df, LEXICON_PATH)

    print(f"Rows before collapse: {len(df)}")

    print("--- Collapsing consecutive turns (Speaker Turn Merging) ---")
    df = collapse_consecutive_turns(df)

    print(f"Rows after collapse: {len(df)}")
    
    print(f"--- Processing strategy labels for: {TARGET_LABELS} ---")
    df = process_labels(df, TARGET_LABELS)

    print(f"--- Saving processed data to {PROCESSED_DATA_PATH} ---")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    
    # Identify valid pairs for training/evaluation
    pairs = create_expert_seeker_pairs(df)

    convs_in_pairs = df.loc[pairs["df_idx"], "engagement_id"].to_numpy()
    uniq_convs = np.unique(convs_in_pairs)

    # Use the modern Generator for shuffling to match original RNG
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq_convs)
    
    # Calculate split points based on the number of conversations in pairs
    n = len(uniq_convs)
    train_cut = int(0.75 * n)
    val_cut = int(0.875 * n)
    
    # Define sets for filtering masks
    tr_ids = set(uniq_convs[:train_cut])
    va_ids = set(uniq_convs[train_cut:val_cut])
    te_ids = set(uniq_convs[val_cut:])
    
    print(f"Splits created: Train={len(tr_ids)}, Val={len(va_ids)}, Test={len(te_ids)} (Conversations)")

    # 3. Graph Construction
    print("--- Building Heterogeneous Graph (SAGE) ---")
    set_seeds(SEED)
    data = build_hetero_graph(df, tr_ids, TARGET_LABELS, lexicon_cols, DEVICE)

    data = data.to(DEVICE)
    
    # Identify labels for 'message' nodes
    # We need to map labels from 'pairs' back to the global 'df' structure for the graph
    label_names = sorted(list(TARGET_LABELS))
    num_labels = len(label_names)
    Y_message = np.zeros((len(df), num_labels), dtype=np.float32)
    
    for _, row in pairs.iterrows():
        idx = int(row["df_idx"])
        active_indices = [label_names.index(l) for l in row["labels_list"]]
        Y_message[idx, active_indices] = 1.0
        
    data["message"].y = torch.tensor(Y_message, dtype=torch.float32)

    # Masks for message-level classification
    dfidx_all = pairs["df_idx"].values
    full_train_mask = torch.zeros(len(df), dtype=torch.bool)
    full_train_mask[pairs[pairs["df_idx"].isin(df.index[df["engagement_id"].isin(tr_ids)])]["df_idx"].values] = True
    data["message"].train_mask = full_train_mask
    
    full_val_mask = torch.zeros(len(df), dtype=torch.bool)
    full_val_mask[pairs[pairs["df_idx"].isin(df.index[df["engagement_id"].isin(va_ids)])]["df_idx"].values] = True
    data["message"].val_mask = full_val_mask

    full_test_mask = torch.zeros(len(df), dtype=torch.bool)
    full_test_mask[pairs[pairs["df_idx"].isin(df.index[df["engagement_id"].isin(te_ids)])]["df_idx"].values] = True
    data["message"].test_mask = full_test_mask

    # 4. Model Initialization
    in_channels_dict = {
        "message": data["message"].x.size(1),
        "conversation": data["conversation"].x.size(1),
        "lexicon": 32, # Based on Embedding(num, 32)
        "distress": 32
    }

    set_seeds(SEED)

    model = HGTMultiLabel(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=num_labels,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        lexicon_x=data["lexicon"].x,
        distress_x=data["distress"].x
    ).to(DEVICE)
    
    # 5. Loss and Optimizer 
    y_train = data["message"].y[data["message"].train_mask].to(DEVICE)
    pos_counts = y_train.sum(dim=0).clamp(min=1)
    neg_counts = data["message"].train_mask.sum() - pos_counts
    pos_weight = (neg_counts / pos_counts).to(torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)

    # 6. Training Loop with Early Stopping
    best_val_f1 = -1.0
    wait = 0
    best_state = None
    
    print(f"--- Starting HGT Training (Device: {DEVICE}) ---")
    for epoch in range(1, GNN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(data.x_dict, data.edge_index_dict)
        loss = criterion(logits[data["message"].train_mask], y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(data.x_dict, data.edge_index_dict)
                v_probs = torch.sigmoid(val_logits[data["message"].val_mask]).cpu().numpy()
                v_true = data["message"].y[data["message"].val_mask].cpu().numpy()
                
                v_pred = (v_probs >= 0.5).astype(int)
                val_f1 = f1_score(v_true, v_pred, average="macro", zero_division=0)
                
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1 (Macro): {val_f1:.3f}")
                
                if val_f1 > best_val_f1 + 1e-4:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
            
            if wait >= GNN_PATIENCE:
                print("--- Early stopping triggered ---")
                break

    # 7. Final Evaluation and Threshold Tuning
    if best_state:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            full_logits, _ = model(data.x_dict, data.edge_index_dict)
            
            # Tune thresholds on Validation set
            v_probs = torch.sigmoid(full_logits[data["message"].val_mask]).cpu().numpy()
            v_true = data["message"].y[data["message"].val_mask].cpu().numpy()
            best_thresholds = tune_thresholds(v_true, v_probs)
            print(f"Tuned thresholds: {dict(zip(label_names, best_thresholds))}")

            # Evaluate on Test set
            t_probs = torch.sigmoid(full_logits[data["message"].test_mask]).cpu().numpy()
            t_true = data["message"].y[data["message"].test_mask].cpu().numpy()
            
            report_df, _ = generate_performance_report(t_true, t_probs, best_thresholds, label_names)
            
            print("\n--- Final Test Report ---")
            print(report_df.to_string(index=False))
            
            # Save the trained model
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': best_state,
                'thresholds': best_thresholds,
                'label_names': label_names,
                'lexicon_cols': lexicon_cols
            }, os.path.join(OUTPUT_DIR, "best_hgt_model.pt"))
            print(f"Model and metadata saved to {OUTPUT_DIR}")

            graph_save_path = os.path.join(OUTPUT_DIR, "sage_hetero_graph.pt")
            torch.save(data, graph_save_path)
            
            print(f"--- All components saved to {OUTPUT_DIR} ---")
            print(f"Model: best_hgt_model.pt | Graph: sage_hetero_graph.pt")

if __name__ == "__main__":
    run_gnn_pipeline()