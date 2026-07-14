import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support, matthews_corrcoef
import torch

def tune_thresholds(y_true, y_prob):
    """
    Find the best threshold for each label based on validation F1-score.
    Matches Block 13.1 logic.
    """
    num_labels = y_true.shape[1]
    best_thresholds = np.full(num_labels, 0.5, dtype=np.float32)

    for i in range(num_labels):
        y_i = y_true[:, i]
        if y_i.sum() == 0:
            continue
        
        p, r, thr = precision_recall_curve(y_i, y_prob[:, i])
        f1 = 2 * p * r / np.clip(p + r, 1e-8, None)
        
        idx = np.nanargmax(f1)
        best_thresholds[i] = thr[idx] if idx < len(thr) else 0.5
    
    return best_thresholds

def generate_performance_report(y_true, y_prob, thresholds, label_names):
    """
    Generate a full performance report including Micro/Macro P, R, F1, MCC and Support.
    Matches the logic from Blocks 13.2 and 13.3.
    """
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    num_labels = len(label_names)

    # 1. Calculate Per-Label Metrics
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_mcc = np.array([
        matthews_corrcoef(y_true[:, i], y_pred[:, i])
        for i in range(num_labels)
    ])

    # 2. Build Base Report DataFrame
    report_df = pd.DataFrame({
        "label": label_names,
        "precision": per_p,
        "recall": per_r,
        "f1": per_f1,
        "mcc": per_mcc,
        "support": per_sup.astype(int),
    })

    # 3. Calculate Micro/Macro Summary
    micro_metrics = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_metrics = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    mcc_micro = matthews_corrcoef(y_true.reshape(-1), y_pred.reshape(-1))
    mcc_macro = float(np.nanmean(per_mcc))

    # 4. Add Summary Rows
    summary_rows = pd.DataFrame([
        {
            "label": "micro_avg", 
            "precision": micro_metrics[0], "recall": micro_metrics[1], 
            "f1": micro_metrics[2], "mcc": mcc_micro, "support": int(y_true.sum())
        },
        {
            "label": "macro_avg", 
            "precision": macro_metrics[0], "recall": macro_metrics[1], 
            "f1": macro_metrics[2], "mcc": mcc_macro, "support": int(y_true.sum())
        },
    ])

    report_df = pd.concat([report_df, summary_rows], ignore_index=True)
    
    # Round numeric columns for readability
    cols_to_round = ["precision", "recall", "f1", "mcc"]
    report_df[cols_to_round] = report_df[cols_to_round].round(3)
    
    return report_df, y_pred