import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gen_params import *
from config.hyperparams import PROCESSED_DATA_PATH, TARGET_LABELS, SEED, OUTPUT_DIR,GNN_HIDDEN_DIM
from src.preprocessing import create_expert_seeker_pairs
from src.prompt_utils import build_history_text, build_prompt_ft, build_prompt_ft_with_strategy
from src.gen_metrics import compute_bert_score, compute_ppl
from models.hgt_model import HGTMultiLabel
from models.sage_generator import load_sage_generator
from scripts.train_gnn import set_seeds

def run_generation_pipeline():
    # Ensures reproducibility of GPU operations and randomized logic
    set_seeds(SEED)

    print("--- Starting SAGE Generation & Evaluation Pipeline ---")

    # 1. Setup Paths and Load Data (Consistent with hyperparams.py)
    # Define paths for all required artifacts
    hgt_path = os.path.join(OUTPUT_DIR, "best_hgt_model.pt")
    graph_path = os.path.join(OUTPUT_DIR, "sage_hetero_graph.pt")
    sage_path = "./outputs/sage_ft_checkpoints/best_sage_model.pt"

    # Load processing data
    df = pd.read_csv(PROCESSED_DATA_PATH, encoding="utf-8-sig")
    df['engagement_id'] = df['engagement_id'].astype(str).str.strip()

    pairs = create_expert_seeker_pairs(df)
    convs_in_pairs = df.loc[pairs["df_idx"], "engagement_id"].to_numpy()
    uniq_convs = np.unique(convs_in_pairs)

    rng = np.random.default_rng(SEED) 
    rng.shuffle(uniq_convs)
    te_ids = set(uniq_convs[int(0.875 * len(uniq_convs)):])

    # Build gen_df to map expert messages to their preceding seeker messages
    gen_rows = []
    for expert_idx in pairs["df_idx"]:
        conv_id = df.loc[expert_idx, "engagement_id"]
        # Find the last seeker message before this expert message in the same conversation
        seeker_idx = df[(df["engagement_id"] == conv_id) & (df.index < expert_idx) & (df["seeker"] == True)].index.max()
        
        if pd.isna(seeker_idx): continue
        
        gen_rows.append({
            "expert_msg_idx": int(expert_idx),
            "seeker_msg_idx": int(seeker_idx)
        })
    gen_df = pd.DataFrame(gen_rows)
    gen_key_to_row = {int(row["expert_msg_idx"]): int(row["seeker_msg_idx"]) for _, row in gen_df.iterrows()}

    print(f"Total conversations in pairs: {len(uniq_convs)}, Test conversations: {len(te_ids)}")

    # 2. Load HGT and Get Predictions 
    print("Loading HGT for strategy prediction...")
    
    # Load the full HeteroData object to extract metadata and dimensions
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file missing at {graph_path}")
    data = torch.load(graph_path, weights_only=False).to(DEVICE)
    hgt_checkpoint = torch.load(hgt_path, weights_only=False)

    label_names = hgt_checkpoint.get('label_names', sorted(list(TARGET_LABELS)))
    thresholds = hgt_checkpoint.get('thresholds', 0.5)

    # Extract dimensions directly from the graph features
    in_channels_dict = {
        "message":      data["message"].x.size(1),
        "conversation": data["conversation"].x.size(1),
        "lexicon":      data["lexicon"].x.size(1),
        "distress":     data["distress"].x.size(1)
    }
    # Initialize HGT with correct metadata
    hgt_model = HGTMultiLabel(
        metadata=data.metadata(), 
        in_channels_dict=in_channels_dict,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=len(label_names),
        lexicon_x=data["lexicon"].x, 
        distress_x=data["distress"].x
    ).to(DEVICE)
        
    # Load trained HGT weights
    hgt_model.load_state_dict(hgt_checkpoint['model_state_dict'])
    hgt_model.eval()
    
    print("Running HGT inference for all strategy predictions...")
    with torch.no_grad():
        # Get embeddings and predicted strategies for all messages
        logits, h_dict = hgt_model(data.x_dict, data.edge_index_dict)
        h_all = {ntype: h.detach() for ntype, h in h_dict.items()}
        
        # Apply thresholds (using saved thresholds from GNN training)
        thresholds = hgt_checkpoint.get('thresholds', 0.5)
        probs = torch.sigmoid(logits).cpu().numpy()

        # Apply thresholds and order labels
        all_predicted_strategies = []
        STRATEGY_ORDER = {"שיקוף": 0, "דיבוב": 1, "מתן נקודה למחשבה": 2}

        for i in range(len(probs)):
            # Use label_names[j] instead of sorted_target_labels[j]
            active = [label_names[j] for j, val in enumerate(probs[i] >= thresholds) if val]
            active.sort(key=lambda x: STRATEGY_ORDER.get(x, 99))
            all_predicted_strategies.append(active)

    del hgt_model # cleanup HGT model from memory
    h_all = {ntype: h.cpu() for ntype, h in h_all.items()} 
    import gc
    gc.collect() 
    torch.cuda.empty_cache()

    # 3. Load Gemma-3-12b Generator
    print(f"Loading SAGE Generator with adapters from: {sage_path}")

    dynamic_graph_dim = h_all["message"].shape[1]
    print(f"Detected dynamic graph_dim: {dynamic_graph_dim}")
    # Initialize Gemma with LoRA and Graph-Injection layers
    generator = load_sage_generator(graph_dim=dynamic_graph_dim, checkpoint_path=sage_path)

    # 4. Evaluation Loop 
    results = []
    test_mask = data["message"].test_mask.cpu().numpy()
    
    for conv_id in tqdm(te_ids, desc="Generating Responses"):
        conv_subset = df[df["engagement_id"] == conv_id].sort_values("message_id")

        for idx, row in conv_subset.iterrows():
            if row["seeker"]: continue # Only intervene on expert turns

            if not test_mask[idx]: continue

            if idx not in gen_key_to_row:
                continue
                
            seeker_idx = gen_key_to_row[idx] 
                
            seeker_text = df.loc[seeker_idx, "text"].strip()
            pred_strat = all_predicted_strategies[idx]
            
            # Identify intervention point and build prompts
            history = build_history_text(df, conv_id, seeker_idx)
            
            # Get strategy from HGT predictions
            pred_strat = all_predicted_strategies[idx]
            prompt_basic = build_prompt_ft(history, seeker_text)
            prompt_strat = build_prompt_ft_with_strategy(history, seeker_text, pred_strat)

            # --- Generation Logic ---
            with torch.no_grad():
                # Here you would call your generation functions for:
                # vanilla_basic, ft_vanilla, ft_ga_only, ft_ga_strategy
                # GA-Strategy:
                gen_text = generator.generate_dynamic(
                    prompt=prompt_strat,
                    current_seeker_idx=seeker_idx,
                    h_all=h_all,
                    data_obj=data
                )

                conv_edge = data["message", "in_conversation", "conversation"].edge_index
                conv_idx = conv_edge[1][seeker_idx]
                all_conv_msgs = (conv_edge[1] == conv_idx).nonzero(as_tuple=True)[0]
                history_indices = all_conv_msgs[all_conv_msgs < seeker_idx]
                
                lex_edge = data["message", "has_lexicon", "lexicon"].edge_index
                relevant_msgs = torch.cat([history_indices, torch.tensor([seeker_idx], device=DEVICE)])
                lex_mask = torch.isin(lex_edge[0], relevant_msgs)
                lexicon_indices = lex_edge[1][lex_mask].unique()

                ppl_val = compute_ppl(
                    text=gen_text,
                    model=generator,
                    tokenizer=generator.tokenizer,
                    s_idx=seeker_idx,
                    hist_idx=history_indices,
                    lex_idx=lexicon_indices,
                    h_all=h_all
                )
                
            results.append({
                "conv_id": conv_id,
                "reference_text": row["text"],
                "predicted_strategy": pred_strat,
                "generated_text": gen_text,
                "ppl": ppl_val
            })

    # 5. Metrics Calculation
    print("\n--- Running Evaluation Metrics ---")
    df_results = pd.DataFrame(results)
    
    # BERTScore
    df_results["bert_score"] = compute_bert_score(
        df_results["generated_text"].tolist(), 
        df_results["reference_text"].tolist()
    )

    # 6. Final Summary
    print(f"Mean BERTScore: {df_results['bert_score'].mean():.4f}")
    print(f"Mean PPL: {df_results['ppl'].mean():.4f}")
    output_save_path = os.path.join("./outputs", "generation_results.csv")
    df_results.to_csv(output_save_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_save_path}")

if __name__ == "__main__":
    run_generation_pipeline()