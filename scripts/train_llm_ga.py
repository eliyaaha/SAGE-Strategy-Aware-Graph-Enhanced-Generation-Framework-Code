import sys
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gen_params import *
from config.hyperparams import GNN_HIDDEN_DIM, SEED, PROCESSED_DATA_PATH, GNN_EPOCHS, OUTPUT_DIR, TARGET_LABELS
from src.preprocessing import load_and_merge_data, process_labels, create_expert_seeker_pairs
from src.prompt_utils import build_history_text, build_prompt_ft, build_prompt_ft_with_strategy
from models.hgt_model import HGTMultiLabel
from models.sage_generator import load_sage_generator

# --- Reproduction ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Dataset & Collate ---
class GraphAwareFTDataset(torch.utils.data.Dataset):
    def __init__(self, gen_df, data_obj, global_df):
        self.df = gen_df
        self.data = data_obj # Full HeteroData object
        self.global_df = global_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        conv_id = row["engagement_id"]
        current_seeker_idx = int(row["seeker_msg_idx"])
        
        # 1. Identify History Messages (causal boundary)
        history_mask = (self.global_df["engagement_id"] == conv_id) & (self.global_df.index < current_seeker_idx)
        history_indices = torch.tensor(self.global_df.index[history_mask].tolist(), dtype=torch.long)

        # 2. Identify Causal Lexicon Nodes
        lex_edge_index = self.data["message", "has_lexicon", "lexicon"].edge_index
        all_relevant_msgs = history_indices.tolist() + [current_seeker_idx]

        lex_mask = torch.isin(
        lex_edge_index[0], 
        torch.tensor(all_relevant_msgs).to(lex_edge_index.device)
    )
        active_lexicon_indices = lex_edge_index[1][lex_mask].unique()

        gold_strat = row["strategy_labels"]
        if not (isinstance(gold_strat, list) and len(gold_strat) > 0):
            gold_strat = []

        return {
            "history_text": row.get("history_text", ""),
            "seeker_text": row["seeker_text"],
            "strategy_to_use": gold_strat, 
            "expert_text": row["expert_text"],
            "current_seeker_idx": current_seeker_idx,
            "history_indices": history_indices,
            "lexicon_indices": active_lexicon_indices
        }

def unified_collate_fn(examples, tokenizer):
    texts = []
    current_seeker_indices = []
    history_indices_list = []
    lexicon_indices_list = []
    
    for ex in examples:
        # Strategy Masking
        is_masked = random.random() < 0.0 
        if is_masked:
            prompt = build_prompt_ft(ex["history_text"], ex["seeker_text"])
        else:
            prompt = build_prompt_ft_with_strategy(ex["history_text"], ex["seeker_text"], ex["strategy_to_use"])
            
        full_text = f"{prompt} {ex['expert_text']}{tokenizer.eos_token}"
        texts.append(full_text)
        
        current_seeker_indices.append(ex["current_seeker_idx"])
        history_indices_list.append(ex["history_indices"])
        lexicon_indices_list.append(ex["lexicon_indices"])

    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt")
    
    # Label masking for therapist-only loss
    labels = enc["input_ids"].clone()
    marker = tokenizer("מטפל:", add_special_tokens=False)["input_ids"]
    for i in range(labels.size(0)):
        ids = labels[i].tolist()
        for j in range(len(ids) - len(marker)):
            if ids[j:j+len(marker)] == marker:
                labels[i, :j+len(marker)] = -100
                break
                
    return {
        "input_ids": enc["input_ids"].to(DEVICE),
        "attention_mask": enc["attention_mask"].to(DEVICE),
        "labels": labels.to(DEVICE),
        "current_seeker_idx": torch.tensor(current_seeker_indices, dtype=torch.long),
        "history_indices": history_indices_list,
        "lexicon_indices": lexicon_indices_list
    }

# --- Train Cycle ---
def train_cycle(model, hgt, loader, optimizer, data_obj, is_train=True, accum_steps=4):
    model.train() if is_train else model.eval()
    hgt.eval() 
    
    total_loss = 0
    pbar = tqdm(loader, desc="Training" if is_train else "Validation", leave=False)
    
    if is_train:
        optimizer.zero_grad()

    with torch.no_grad():
        _, h_all = hgt(
            {ntype: x.to(DEVICE) for ntype, x in data_obj.x_dict.items()},
            {rel: ei.to(DEVICE) for rel, ei in data_obj.edge_index_dict.items()}
        )

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(pbar):
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
                current_seeker_idx=batch["current_seeker_idx"].to(DEVICE),
                history_indices=[t.to(DEVICE) for t in batch["history_indices"]],
                lexicon_indices=[t.to(DEVICE) for t in batch["lexicon_indices"]],
                h_all=h_all
            )
            
            loss = outputs.loss / accum_steps
            
            if is_train:
                loss.backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accum_steps
            
    return total_loss / len(loader)

def run_training():
    """
    Full training pipeline: Data preparation, Model initialization, 
    and Fine-tuning with Early Stopping.
    """
    print("--- Starting Graph-Aware LLM Training ---")

    # 1. Load Data & Create Base Pairs
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Load the pre-built graph object at the beginning of run_training()
    graph_path = os.path.join(OUTPUT_DIR, "sage_hetero_graph.pt")
    if not os.path.exists(graph_path):
        # Ensure train_gnn.py has been executed to generate this file
        raise FileNotFoundError(f"Graph file not found at {graph_path}. Please run train_gnn.py first.")

    # Load the HeteroData object and move it to the active device 
    data = torch.load(graph_path, weights_only=False).to(DEVICE)
    print(f"--- Graph loaded and moved to {DEVICE} ---")
    
    # Identify valid expert messages for the generation task
    is_expert = ~df["seeker"].values
    # Ensure message has labels and a preceding seeker message
    df["seeker_text_candidate"] = np.where(df["seeker"], df["text"], np.nan)
    df["last_seeker_text"] = df.groupby("engagement_id")["seeker_text_candidate"].ffill()
    valid_expert = is_expert & (df["processed"].notna()) & (df["last_seeker_text"].notna())
    
    # 2. Consistent Splitting by Conversation ID 
    conv_ids = df.loc[valid_expert, "engagement_id"].to_numpy()
    uniq_convs = np.unique(df["engagement_id"].unique())
    rng = np.random.default_rng(SEED) 
    rng.shuffle(uniq_convs)
    
    n = len(uniq_convs)
    cut1 = int(0.75 * n)
    cut2 = int(0.875 * n)
    
    tr_ids = set(uniq_convs[:cut1])
    va_ids = set(uniq_convs[cut1:cut2])
    te_ids = set(uniq_convs[cut2:])
    
    # 3. Build Generation Rows with History Context
    gen_rows = []
    for expert_idx in np.where(valid_expert)[0]:
        conv_id = df.loc[expert_idx, "engagement_id"]
        
        if conv_id in tr_ids:
            split_name = "train"
        elif conv_id in va_ids:
            split_name = "val"
        else:
            continue 

        # Find the seeker message immediately preceding this expert message
        seeker_idx = None
        for i in range(expert_idx - 1, -1, -1):
            if df.loc[i, "engagement_id"] != conv_id: break
            if df.loc[i, "seeker"]:
                seeker_idx = i
                break
        
        if seeker_idx is not None:
            history = build_history_text(df, conv_id, seeker_idx)
            gen_rows.append({
                "engagement_id": conv_id,
                "split": split_name,
                "seeker_msg_idx": seeker_idx,
                "expert_msg_idx": expert_idx,
                "history_text": history,
                "seeker_text": str(df.loc[seeker_idx, "text"]).strip(),
                "expert_text": str(df.loc[expert_idx, "text"]).strip(),
                "strategy_labels": str(df.loc[expert_idx, "processed"]).split("+")
            })

    gen_df = pd.DataFrame(gen_rows)
    train_df = gen_df[gen_df["split"] == "train"]
    val_df = gen_df[gen_df["split"] == "val"]

    # 4. Initialize Datasets & Loaders 
    # This allows the Dataset to perform causal subgraph retrieval during training
    train_ds = GraphAwareFTDataset(train_df, data, df)
    val_ds = GraphAwareFTDataset(val_df, data, df)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_NAME)
    
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, 
        collate_fn=lambda x: unified_collate_fn(x, tokenizer)
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        collate_fn=lambda x: unified_collate_fn(x, tokenizer)
    )

    # 5. Initialize HGT model with dimensions derived directly from the graph
    in_channels_dict = {
        "message":      data["message"].x.size(1),
        "conversation": data["conversation"].x.size(1),
        "lexicon":      data["lexicon"].x.size(1),
        "distress":     data["distress"].x.size(1)
    }

    # Load pre-trained HGT checkpoint first, so the output dimension matches the saved labels
    hgt_checkpoint = torch.load(
        os.path.join(OUTPUT_DIR, "best_hgt_model.pt"),
        weights_only=False
    )

    label_names = hgt_checkpoint.get("label_names", sorted(list(TARGET_LABELS)))

    # Reconstruct HGT architecture using graph metadata
    hgt_model = HGTMultiLabel(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=len(label_names),
        lexicon_x=data["lexicon"].x,
        distress_x=data["distress"].x
    ).to(DEVICE)

    # Load pre-trained HGT weights from the classifier training stage
    hgt_model.load_state_dict(hgt_checkpoint["model_state_dict"])


    hgt_model.eval() # Set to evaluation mode as HGT provides fixed embeddings for the LLM

    with torch.no_grad():
        _, h_all_init = hgt_model(
            {ntype: x.to(DEVICE) for ntype, x in data.x_dict.items()},
            {rel: ei.to(DEVICE) for rel, ei in data.edge_index_dict.items()}
        )
        dynamic_graph_dim = h_all_init["message"].shape[1]
    
    print(f"--- Detected dynamic graph_dim: {dynamic_graph_dim} ---")
    
    # Load and wrap Gemma
    graph_aware_model = load_sage_generator(graph_dim=dynamic_graph_dim)

    # 6. Setup Optimizer with Dual Learning Rates
    ga_params = [p for n, p in graph_aware_model.named_parameters() if any(k in n for k in ["graph_proj", "gating", "cross_attn", "graph_to_llm_proj"])]
    lora_params = [p for n, p in graph_aware_model.named_parameters() if "base_lm" in n]
    
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': GA_LR_LORA}, 
        {'params': ga_params, 'lr': GA_LR_GRAPH}
    ], weight_decay=0.01)

    # 7. Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience = 2
    wait = 0
    checkpoint_dir = "./outputs/sage_ft_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting Training on {len(train_df)} interventions...")
    for epoch in range(EPOCHS_GRAPHAWARE): # As per epochs_graphaware
        train_loss = train_cycle(graph_aware_model, hgt_model, train_loader, optimizer, data, is_train=True)
        val_loss = train_cycle(graph_aware_model, hgt_model, val_loader, optimizer, data, is_train=False)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            # Save the best model components
            save_path = os.path.join(checkpoint_dir, "best_sage_model.pt")
            torch.save(graph_aware_model.state_dict(), save_path)
            print(f"  --> Saved new best weights to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    run_training()