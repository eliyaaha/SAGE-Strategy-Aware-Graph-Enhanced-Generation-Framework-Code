import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from config.hyperparams import SEED

def get_alephbert_embeddings(texts, model_name, device, max_length=128):
    """Compute AlephBERT CLS embeddings for message nodes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for txt in texts:
        encoded = tokenizer(txt, truncation=True, padding="max_length", 
                          max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze(0).cpu())
    
    del model
    torch.cuda.empty_cache()
    return torch.stack(all_embeddings)

def reset_torch_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def build_hetero_graph(df, tr_ids, target_labels, lexicon_cols, device, include_lexicon=True, include_distress=True):
    """
    Construct the full SAGE heterogeneous graph.
    Includes demographics, lexicon, and distress nodes with leakage prevention.
    Ablation options to exclude lexicon/distress components for simplified baselines.
    """
    data = HeteroData()
    num_messages = len(df)
    conv_ids = df["engagement_id"].to_numpy()
    uniq_convs, inv_conv = np.unique(conv_ids, return_inverse=True)
    num_conversations = len(uniq_convs)

    # --- 1. Message Nodes & Positional Encoding ---
    from config.hyperparams import ENCODER_MODEL
    # Initial AlephBERT features
    msg_x = get_alephbert_embeddings(df["text"].astype(str).tolist(), ENCODER_MODEL, device)
    
    # Calculate normalized position (0..1) within conversation
    pos_list = np.zeros(num_messages, dtype=np.float32)
    for conv_idx in range(num_conversations):
        indices = np.where(inv_conv == conv_idx)[0]
        if len(indices) > 1:
            pos_list[indices] = np.arange(len(indices)) / float(len(indices) - 1)
    
    pos_feat = torch.tensor(pos_list).unsqueeze(1)
    data["message"].x = torch.cat([msg_x, pos_feat], dim=1) 

    # --- 2. Conversation Nodes (Age & Gender One-hot) ---
    def get_one_hot_vocab(series, unk_label):
        raw = series.astype(str).str.strip()
        raw = raw.replace(["", "nan", "NaN", "None"], np.nan)
        values = raw.dropna().unique().tolist()
        vocab = sorted([str(v).strip() for v in values if str(v).strip() != ""])
        vocab.append(unk_label)
        return {v: i for i, v in enumerate(vocab)}, len(vocab)

    # Conversation & Distress Nodes (Ablation Controlled)
    if include_distress:
        # Demographics One-hot
        age_map, num_age = get_one_hot_vocab(df["age"], "UNK_AGE")
        gen_map, num_gen = get_one_hot_vocab(df["gender"], "UNK_GENDER")

        conv_features = []
        for conv_idx in range(num_conversations):
            sample_idx = np.where(inv_conv == conv_idx)[0][0]
            
            # Age One-hot
            age_vec = np.zeros(num_age, dtype=np.float32)
            age_val = str(df["age"].iloc[sample_idx]).strip()
            age_idx = age_map.get(age_val, age_map["UNK_AGE"])
            age_vec[age_idx] = 1.0
            
            # Gender One-hot
            gender_vec = np.zeros(num_gen, dtype=np.float32)
            gender_val = str(df["gender"].iloc[sample_idx]).strip()
            gender_idx = gen_map.get(gender_val, gen_map["UNK_GENDER"])
            gender_vec[gender_idx] = 1.0
            
            conv_features.append(np.concatenate([age_vec, gender_vec]))
        
        data["conversation"].x = torch.tensor(np.vstack(conv_features), dtype=torch.float32) 

        # --- 3. Distress Nodes (Subject Vocab) ---
        distress_cols = [f"subject_{i}" for i in range(1, 7)]
        all_distress = []
        for col in distress_cols:
            all_distress.extend(df[col].dropna().astype(str).str.strip().tolist())
        
        distress_vocab = sorted(list(set([d for d in all_distress if d != ""])))
        num_distress = len(distress_vocab)
        dist_map = {d: i for i, d in enumerate(distress_vocab)}
        
        # Trainable distress embeddings
        reset_torch_seed(SEED)
        distress_embed = nn.Embedding(num_distress, 32)
        data["distress"].x = distress_embed.weight.clone().detach()

    else:
        # Ablation Mode: No Distress Nodes, Only Conversation with Demographics
        data["conversation"].x = torch.zeros((num_conversations, 1), dtype=torch.float32)
        data["distress"].x = torch.zeros((1, 32), dtype=torch.float32)

    # --- Lexicon Nodes ---
    if include_lexicon and len(lexicon_cols) > 0:
        # Trainable lexicon embeddings
        reset_torch_seed(SEED)
        lex_embed = nn.Embedding(len(lexicon_cols), 32)
        data["lexicon"].x = lex_embed.weight.clone().detach()
    else:
        data["lexicon"].x = torch.zeros((1, 32), dtype=torch.float32)

    # --- Temporal Edges (follows) ---
    src_mm, dst_mm = [], []
    for step in [1, 2, 3]:
        for i in range(num_messages - step):
            if conv_ids[i] == conv_ids[i + step]:
                src_mm.append(i)
                dst_mm.append(i + step)
    data["message", "follows", "message"].edge_index = torch.tensor([src_mm, dst_mm], dtype=torch.long)

    # --- Edges: Message <-> Lexicon (Seeker Only) ---
    if include_lexicon and len(lexicon_cols) > 0:
        cat_mat = df[lexicon_cols].fillna(0).clip(0, 1).to_numpy().astype(np.float32)
        m_idx, l_idx = np.nonzero(cat_mat)
        is_seeker = df["seeker"].astype(bool).to_numpy()
        seeker_mask = is_seeker[m_idx]
        
        data["message", "has_lexicon", "lexicon"].edge_index = torch.tensor([m_idx[seeker_mask], l_idx[seeker_mask]], dtype=torch.long)
        data["lexicon", "rev_has_lexicon", "message"].edge_index = torch.tensor([l_idx[seeker_mask], m_idx[seeker_mask]], dtype=torch.long)
    else:
        data["message", "has_lexicon", "lexicon"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["lexicon", "rev_has_lexicon", "message"].edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Edges: Message <-> Conversation ---
    if include_distress:
        msg_idx_t = torch.arange(num_messages, dtype=torch.long)
        conv_idx_t = torch.tensor(inv_conv, dtype=torch.long)
        data["message", "in_conversation", "conversation"].edge_index = torch.stack([msg_idx_t, conv_idx_t], dim=0)
    else:
        data["message", "in_conversation", "conversation"].edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Edges: Conversation <-> Distress (TRAIN ONLY - Leakage Prevention) ---
    if include_distress:
        c_to_d_src, c_to_d_dst = [], []
        for conv_idx, conv_id in enumerate(uniq_convs):
            if conv_id in tr_ids: # Only connect if conversation is in training set
                # Find all unique distress labels for this conversation
                sample_indices = np.where(inv_conv == conv_idx)[0]
                for s_idx in sample_indices:
                    for col in distress_cols:
                        val = str(df[col].iloc[s_idx]).strip()
                        if val in dist_map:
                            c_to_d_src.append(conv_idx)
                            c_to_d_dst.append(dist_map[val])
        
        if c_to_d_src:
            data["conversation", "has_distress", "distress"].edge_index = torch.tensor([c_to_d_src, c_to_d_dst], dtype=torch.long).unique(dim=1)
            data["distress", "rev_has_distress", "conversation"].edge_index = torch.tensor([c_to_d_dst, c_to_d_src], dtype=torch.long).unique(dim=1)
    else:
        data["conversation", "has_distress", "distress"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["distress", "rev_has_distress", "conversation"].edge_index = torch.empty((2, 0), dtype=torch.long)



    return data