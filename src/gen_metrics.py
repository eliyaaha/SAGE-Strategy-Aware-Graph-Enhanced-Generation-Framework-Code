import sys
import os
import math
import torch
from bert_score import score as bert_score_func

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gen_params import DEVICE

def compute_bert_score(predictions, references):
    """
    Computes BERTScore for Hebrew text with baseline rescaling.
    """
    if not predictions or not references:
        return []

    # Running BERTScore for Hebrew 
    P, R, F1 = bert_score_func(
        cands=predictions,
        refs=references,
        lang="he",
        rescale_with_baseline=True
    )
    return F1.tolist()

def compute_ppl(text, model, tokenizer, s_idx=None, hist_idx=None, lex_idx=None, h_all=None):
    """
    Calculates Perplexity (PPL).
    Supports both Vanilla LLMs and Dynamic Graph-Aware models.
    """
    if not text or str(text).strip() == "": 
        return None
        
    # Standardize to add_special_tokens=False to match training setup
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    if enc["input_ids"].size(1) <= 1: 
        return None 
    
    model.eval()
    with torch.no_grad():
        if h_all is not None:
            outputs = model(
                input_ids=enc["input_ids"], 
                attention_mask=torch.ones_like(enc["input_ids"]),
                current_seeker_idx=torch.tensor([s_idx], device=DEVICE),
                history_indices=[hist_idx],
                lexicon_indices=[lex_idx],
                labels=enc["input_ids"],
                h_all=h_all
            )
        else:
            # For Vanilla/FT models: Standard cross-entropy loss calculation
            if hasattr(model, 'base_lm'):
                outputs = model.base_lm(input_ids=enc["input_ids"], labels=enc["input_ids"])
            else:
                outputs = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
            
    return math.exp(outputs.loss.item())