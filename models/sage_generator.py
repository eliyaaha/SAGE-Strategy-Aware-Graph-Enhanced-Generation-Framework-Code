import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gen_params import GEMMA_MODEL_NAME, DEVICE, NUM_VIRTUAL_TOKENS, INITIAL_GRAPH_GATING


class GraphAwareGemma(nn.Module):
    """
    Implements Dynamic Cross-modality Pooling with Strategy Guidance.
    """
    def __init__(self, base_lm, graph_dim):
        super().__init__()
        self.base_lm = base_lm
        self.d_model = base_lm.get_input_embeddings().embedding_dim
        self.graph_dim = graph_dim
        
        # Projection layer to align Graph space with LLM space
        self.graph_to_llm_proj = nn.Linear(graph_dim, self.d_model).to(torch.bfloat16)
        
        # Multi-head Attention to perform the dynamic pooling (CMP)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=8, 
            batch_first=True
        ).to(torch.bfloat16)

        # Domain Projector (2-layer MLP with GELU as per original source)
        self.graph_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * NUM_VIRTUAL_TOKENS),
            nn.GELU(),
            nn.LayerNorm(self.d_model * NUM_VIRTUAL_TOKENS)
        ).to(torch.bfloat16)
        
        # Learnable gating parameter to control graph influence
        self.gating = nn.Parameter(torch.tensor([INITIAL_GRAPH_GATING], dtype=torch.bfloat16))

    def forward(self, input_ids, attention_mask, current_seeker_idx, history_indices, lexicon_indices, labels=None, h_all=None):
        """
        Forward pass with dynamic attention over causal subgraph nodes.
        """
        batch_size = input_ids.shape[0]
        inputs_embeds = self.base_lm.get_input_embeddings()(input_ids)

        pooled_graph_features = []
        
        device = self.gating.device 
        
        for i in range(batch_size):
            h_device = h_all["message"].device
            
            msg_h = h_all["message"][history_indices[i].to(h_device)].to(device) 
            lex_h = h_all["lexicon"][lexicon_indices[i].to(h_device)].to(device) 
            
            memory_bank = torch.cat([msg_h, lex_h], dim=0) 
            memory_bank_llm = self.graph_to_llm_proj(memory_bank.to(torch.bfloat16)) 

            current_seeker_h = h_all["message"][current_seeker_idx[i].to(h_device)].unsqueeze(0).to(device) 
            query_llm = self.graph_to_llm_proj(current_seeker_h.to(torch.bfloat16))    
            # Perform Cross-Attention
            attn_output, _ = self.cross_attn(
                query=query_llm.unsqueeze(0), 
                key=memory_bank_llm.unsqueeze(0), 
                value=memory_bank_llm.unsqueeze(0)
            )
            
            pooled_graph_features.append(attn_output.view(self.d_model)) 

        # Project context to virtual tokens
        graph_context = torch.stack(pooled_graph_features).to(self.gating.device)
        graph_tokens = (self.graph_proj(graph_context) * self.gating).view(
            batch_size, NUM_VIRTUAL_TOKENS, self.d_model
        )

        # Concatenate graph tokens with text embeddings
        full_embeds = torch.cat([graph_tokens, inputs_embeds], dim=1)
        
        # Expand attention mask
        prefix_mask = torch.ones((batch_size, NUM_VIRTUAL_TOKENS), device=DEVICE, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Adjust labels for training if provided
        full_labels = None
        if labels is not None:
            prefix_labels = torch.full((batch_size, NUM_VIRTUAL_TOKENS), -100, device=DEVICE, dtype=labels.dtype)
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        return self.base_lm(
            inputs_embeds=full_embeds, 
            attention_mask=full_attention_mask, 
            labels=full_labels
        )
    
    @torch.no_grad()
    def generate_dynamic(self, prompt, current_seeker_idx, h_all, data_obj, max_new_tokens=60):
        """
        Implements Causal Subgraph Retrieval exactly as per Notebook Block 23.
        """
        self.eval()
        device = self.gating.device
        
        # 1. Causal Subgraph Retrieval
        # Identify history: previous messages in the same conversation
        conv_edge = data_obj["message", "in_conversation", "conversation"].edge_index
        conv_idx = conv_edge[1][current_seeker_idx]
        all_conv_msgs = (conv_edge[1] == conv_idx).nonzero(as_tuple=True)[0]
        history_indices = all_conv_msgs[all_conv_msgs < current_seeker_idx]
        
        # Identify lexicon: nodes connected to history or current seeker msg
        lex_edge = data_obj["message", "has_lexicon", "lexicon"].edge_index
        relevant_msgs = torch.cat([history_indices, torch.tensor([current_seeker_idx], device=device)])
        lex_mask = torch.isin(lex_edge[0], relevant_msgs)
        lexicon_indices = lex_edge[1][lex_mask].unique()

        # 2. Tokenization and indexing
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        generated_ids = enc["input_ids"]
        
        # Wrapping in lists to match Batch expectations in forward()
        s_idx_tensor = torch.tensor([current_seeker_idx], dtype=torch.long).to(device)
        h_indices_list = [history_indices.to(device)]
        l_indices_list = [lexicon_indices.to(device)]

        # 3. Autoregressive Loop (Original Sampling: Temp 0.4, Top-P 0.9)
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                current_seeker_idx=s_idx_tensor,
                history_indices=h_indices_list,
                lexicon_indices=l_indices_list,
                h_all=h_all
            )
            
            logits = outputs.logits[:, -1, :]
            logits = logits[0] / 0.4 
            logits = self._top_p_filtering(logits, top_p=0.9)
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return self._extract_therapist_reply(self.tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    def _top_p_filtering(self, logits, top_p=0.9, min_tokens_to_keep=1):
        """Nucleus sampling logic from Notebook Block 22.5."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        remove = cumprobs > top_p
        if min_tokens_to_keep > 1:
            remove[:min_tokens_to_keep] = False
        remove[..., 0] = False

        filtered = logits.clone()
        filtered[sorted_indices[remove]] = -float("inf")
        return filtered

    def _extract_therapist_reply(self, full_text):
        """Original cleanup logic."""
        marker = "מטפל:"
        if marker in full_text:
            return full_text.split(marker)[-1].strip()
        return full_text.strip()

    def _block_repeat_ngrams(self, logits, generated_ids, no_repeat_ngram_size=3):
        if no_repeat_ngram_size <= 0:
            return logits
        ids = generated_ids[0].tolist()
        T = len(ids)
        if T < no_repeat_ngram_size:
            return logits
        prefix_to_next = {}
        for i in range(T - no_repeat_ngram_size + 1):
            prefix = tuple(ids[i : i + no_repeat_ngram_size - 1])
            nxt = ids[i + no_repeat_ngram_size - 1]
            prefix_to_next.setdefault(prefix, set()).add(nxt)
        current_prefix = tuple(ids[-(no_repeat_ngram_size - 1):])
        banned = prefix_to_next.get(current_prefix, set())
        if banned:
            logits[0, list(banned)] = -float("inf")
        return logits

def load_sage_generator(graph_dim, checkpoint_path=None):
    """
    Initializes the Gemma model with LoRA and wraps it with GraphAwareGemma.
    """
    # 1. Load Base Model 
    base_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    base_model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_NAME)
    
    # 2. Apply LoRA 
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        task_type="CAUSAL_LM",
    )
    gemma_with_lora = get_peft_model(base_model, lora_config)
    gemma_with_lora.config.use_cache = False
    
    # 1. Initialize the wrapper WITHOUT calling .to(DEVICE) on everything
    # This keeps the base Gemma model's device management intact
    model = GraphAwareGemma(gemma_with_lora, graph_dim=graph_dim)
    model.tokenizer = tokenizer
    
    # 2. Manually move ONLY the new SAGE-specific layers to the device
    # These are standard tensors (not meta) so they can be moved safely
    model.graph_proj.to(DEVICE)
    model.gating.data = model.gating.data.to(DEVICE)
    
    # If your architecture includes these additional layers:
    if hasattr(model, 'cross_attn'):
        model.cross_attn.to(DEVICE)
    if hasattr(model, 'graph_to_llm_proj'):
        model.graph_to_llm_proj.to(DEVICE)
        
    # 3. Load SAGE weights if a checkpoint is provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Using strict=False is safer when loading only adapter layers
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if 'trainable_state_dict' in checkpoint:
            state_dict = checkpoint['trainable_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    return model