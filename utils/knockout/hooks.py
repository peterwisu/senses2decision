"""
Knockout Hooks for Qwen2.5-Omni Attention Layers
 
Registers forward pre-hooks on Qwen2_5OmniAttention modules to inject
knockout masks (set specific query-key positions to -inf) at selected layers.
 
Usage:
    from utils.knockout.hooks import KnockoutManager
    from utils.knockout.configs import KNOCKOUT_CONFIGS, TokenPositions
 
    manager = KnockoutManager(model, token_positions, config_id=1, target_layers=[0,1,2,3,4,5])
    
    # Run forward pass — hooks automatically modify attention masks
    outputs = model(**inputs)
    
    # Clean up
    manager.remove_hooks()
"""
 
import torch
from typing import List, Optional, Set
from utils.knockout.configs import KnockoutConfig, TokenPositions, KNOCKOUT_CONFIGS
 
 
# ============================================================
# Build knockout mask from config + token positions
# ============================================================
 
def build_knockout_mask_fast(
    config: KnockoutConfig,
    token_positions: TokenPositions,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorised knockout mask — avoids Python loops over tokens.
    
    Returns mask [seq_len, seq_len] with -inf at blocked positions, 0 elsewhere.
    """
    seq_len = token_positions.seq_len
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    neg_inf = torch.finfo(dtype).min
 
    for query_type, key_type in config.blocked_pairs:
        q_idx = torch.tensor(token_positions.get_indices(query_type), device=device, dtype=torch.long)
        k_idx = torch.tensor(token_positions.get_indices(key_type), device=device, dtype=torch.long)
 
        if len(q_idx) == 0 or len(k_idx) == 0:
            continue
 
        if config.cross_unit is None:
            # Block all pairs: mask[q, k] = -inf for all q in query_type, k in key_type
            mask[q_idx.unsqueeze(1), k_idx.unsqueeze(0)] = neg_inf
 
        elif config.cross_unit == "cross_frame":
            # Build patch assignment for each video token
            q_patches = torch.tensor(
                [token_positions.get_temporal_patch_of(i.item()) for i in q_idx],
                device=device, dtype=torch.long
            )
            k_patches = torch.tensor(
                [token_positions.get_temporal_patch_of(i.item()) for i in k_idx],
                device=device, dtype=torch.long
            )
            # Cross-patch = different patch IDs, both valid (>= 0)
            q_valid = q_patches >= 0
            k_valid = k_patches >= 0
            cross = (q_patches.unsqueeze(1) != k_patches.unsqueeze(0))
            valid = q_valid.unsqueeze(1) & k_valid.unsqueeze(0)
            block = cross & valid  # [len(q_idx), len(k_idx)]
 
            # Map back to full seq_len indices
            qi_grid = q_idx.unsqueeze(1).expand_as(block)
            ki_grid = k_idx.unsqueeze(0).expand_as(block)
            mask[qi_grid[block], ki_grid[block]] = neg_inf
 
        elif config.cross_unit == "cross_chunk":
            # Build chunk assignment for each audio token
            q_chunks = torch.tensor(
                [token_positions.get_chunk_of(i.item()) for i in q_idx],
                device=device, dtype=torch.long
            )
            k_chunks = torch.tensor(
                [token_positions.get_chunk_of(i.item()) for i in k_idx],
                device=device, dtype=torch.long
            )
            q_valid = q_chunks >= 0
            k_valid = k_chunks >= 0
            cross = (q_chunks.unsqueeze(1) != k_chunks.unsqueeze(0))
            valid = q_valid.unsqueeze(1) & k_valid.unsqueeze(0)
            block = cross & valid
 
            qi_grid = q_idx.unsqueeze(1).expand_as(block)
            ki_grid = k_idx.unsqueeze(0).expand_as(block)
            mask[qi_grid[block], ki_grid[block]] = neg_inf
    

    # import matplotlib.pyplot as plt
    # import os

    # os.makedirs("./debug_mask", exist_ok=True)
    # seq_len = mask.shape[0]
    # tp = token_positions
    # binary = (mask != 0).float().cpu().numpy()

    # # Segments
    # segments = []
    # segments.append(("System", tp.system[0], tp.system[1], "#2196F3"))
    # for i, (start, end) in enumerate(tp.video_chunks):
    #     segments.append((f"V{i}", start, end, "#4CAF50"))
    # for i, (start, end) in enumerate(tp.audio_chunks):
    #     segments.append((f"A{i}", start, end, "#FF9800"))
    # segments.append(("Question", tp.user_instruction[0], tp.user_instruction[1], "#9C27B0"))
    # segments.append(("Last", tp.last_token, tp.last_token + 1, "#F44336"))

    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(binary, cmap="gray_r", aspect="auto", interpolation="nearest",
    #           extent=[0, seq_len, seq_len, 0])
    # ax.set_xlim(0, seq_len)
    # ax.set_ylim(seq_len, 0)

    # for name, start, end, color in segments:
    #     ax.axvline(x=start, color=color, linewidth=0.8, alpha=0.7)
    #     ax.axvline(x=end, color=color, linewidth=0.8, alpha=0.7)
    #     ax.axhline(y=start, color=color, linewidth=0.8, alpha=0.7)
    #     ax.axhline(y=end, color=color, linewidth=0.8, alpha=0.7)
    #     mid = (start + end) / 2
    #     ax.text(mid, -seq_len * 0.015, name, ha="center", va="bottom", fontsize=6,
    #             color=color, fontweight="bold", rotation=45)
    #     ax.text(-seq_len * 0.015, mid, name, ha="right", va="center", fontsize=6,
    #             color=color, fontweight="bold")

    # ax.set_xlabel("Key position")
    # ax.set_ylabel("Query position")
    # ax.set_title(f"Knockout Mask ({config.name}) — Black=blocked  White=allowed\n"
    #              f"blocked={int(binary.sum())}  seq_len={seq_len}")
    # plt.tight_layout()
    # plt.savefig(f"./debug_mask/knockout_mask_{config.name}.png", dpi=200)
    # plt.close()

    # print(f"\nSegment layout (seq_len={seq_len}):")
    # for name, start, end, color in segments:
    #     print(f"  {name:15s}: [{start:5d}, {end:5d}) = {end-start} tokens")
    # print(f"  Temporal patches: {len(tp.video_temporal_patches)}")
    # print(f"  Knockout blocked: {int(binary.sum())} positions")
    # print(f"  Saved to ./debug_mask/")
    # exit()
    return mask
 
 
# ============================================================
# Hook function
# ============================================================
 
def _make_knockout_hook(knockout_mask_4d: torch.Tensor, layer_idx: int, target_layers: Set[int]):
    """Create a forward pre-hook that adds knockout mask to attention_mask.
    
    The hook intercepts the attention_mask argument of Qwen2_5OmniAttention.forward()
    and adds -inf at knockout positions for target layers only.
    
    Qwen2_5OmniAttention.forward signature:
        forward(self, hidden_states, attention_mask, position_ids, past_key_values,
                output_attentions, use_cache, cache_position, position_embeddings, **kwargs)
    
    attention_mask is args[1] (after hidden_states) or kwargs["attention_mask"]
    Shape: [batch, num_heads_or_1, seq_len, seq_len] or None
    """
    def hook_fn(module, args, kwargs):
        if layer_idx not in target_layers:
            return args, kwargs
 
        # # attention_mask can be in args[1] or kwargs
        # if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:

        #     print(kwargs["attention_mask"].shape , knockout_mask_4d.shape) 
        #     kwargs["attention_mask"] = kwargs["attention_mask"] + knockout_mask_4d
        #     return args, kwargs
        # elif len(args) > 1 and args[1] is not None:
        #     args = list(args)
        #     args[1] = args[1] + knockout_mask_4d
        #     return tuple(args), kwargs
        
        # # If attention_mask is None, we can't inject — this shouldn't happen with eager attention
        # return args, kwargs

        # Find attention_mask in kwargs or args
        am = None
        in_kwargs = False
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            am = kwargs["attention_mask"]
            in_kwargs = True
        elif len(args) > 1 and args[1] is not None:
            am = args[1]

        # Skip knockout during decode steps (query length = 1)
        if am is not None and am.shape[-2] == 1:
            return args, kwargs

        # Prefill: apply knockout
        if in_kwargs:
            # print(layer_idx,am.shape, knockout_mask_4d.shape)
            kwargs["attention_mask"] = am + knockout_mask_4d
            return args, kwargs
        elif am is not None:
            args = list(args)
            # print(layer_idx,am.shape, knockout_mask_4d.shape)
            args[1] = am + knockout_mask_4d
            return tuple(args), kwargs
 
    return hook_fn
 
 
# ============================================================
# Knockout Manager
# ============================================================
 
class KnockoutManager:
    """Manages knockout hooks on a Qwen2.5-Omni model.
    
    Usage:
        manager = KnockoutManager(model, token_positions, config_id=1, target_layers=[0,1,2,3,4,5])
        outputs = model(**inputs)  # hooks fire automatically
        manager.remove_hooks()
    """
    
    def __init__(
        self,
        model,
        token_positions: TokenPositions,
        config_id: int,
        target_layers: List[int],
        use_fast_mask: bool = True,
    ):
        """
        Args:
            model: The Qwen2_5OmniForConditionalGeneration model (or thinker)
            token_positions: TokenPositions object with modality ranges
            config_id: Which knockout config (1-8)
            target_layers: List of layer indices where knockout is applied
            use_fast_mask: Use vectorised mask building (default True)
        """
        self.config = KNOCKOUT_CONFIGS[config_id]
        self.config_id = config_id
        self.target_layers = set(target_layers)
        self.hooks = []
        
        # Get the thinker's text model layers
        text_model = self._get_thinker(model)
        # text_model = thinker.model  # Qwen2_5OmniThinkerTextModel
        
        # Determine device and dtype from model
        device = next(text_model.parameters()).device
        dtype = next(text_model.parameters()).dtype
        
        # Build the knockout mask [seq_len, seq_len]
        if use_fast_mask:
            knockout_mask = build_knockout_mask_fast(self.config, token_positions, device, dtype)
        else:
            raise ValueError("not here")
        
        # Count blocked positions for logging
        self.num_blocked = (knockout_mask != 0).sum().item()
        self.seq_len = token_positions.seq_len
        
        # Expand to 4D: [1, 1, seq_len, seq_len] to broadcast with attention mask
        knockout_mask_4d = knockout_mask.unsqueeze(0).unsqueeze(0)
        
        # Register hooks on each attention module
        for layer in text_model:
            attn = layer.self_attn
            layer_idx = attn.layer_idx
            
            hook_fn = _make_knockout_hook(knockout_mask_4d, layer_idx, self.target_layers)
            handle = attn.register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(handle)
    
    def _get_thinker(self, model):
        """Navigate model hierarchy to find the thinker component."""
        if hasattr(model, 'thinker'):
            return model.thinker.model.layers
        # VL
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            return model.model.language_model.layers
        # Video salmond
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
          
            return model.model.layers
        else:
            raise ValueError(
                "Cannot find thinker component. Expected model.thinker or model with .model.layers"
            )
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()
    
    def __repr__(self):
        return (
            f"KnockoutManager(config={self.config_id} '{self.config.name}', "
            f"layers={sorted(self.target_layers)}, "
            f"blocked={self.num_blocked}/{self.seq_len**2} positions)"
        )
 
 

 
 
def extract_mcq_probs(logits: torch.Tensor, tokenizer, options=("A", "B", "C", "D")) -> dict:
    """Extract MCQ probabilities from logits at the last token position.
    
    Follows Map the Flow (2510.13251): softmax over FULL vocabulary,
    then index into it for option probabilities.
    
    Args:
        logits: [vocab_size] tensor of logits at last token
        tokenizer: the tokenizer to convert option letters to token ids
        options: tuple of option letters
    
    Returns:
        dict with 'probs', 'logits_raw', 'predicted', 'predicted_id', 'max_prob'
    """
    # Full vocabulary softmax (same as Map the Flow)
    full_probs = torch.softmax(logits, dim=0)
    
    # Get token IDs for options via vocab dict (same as Map the Flow)
    vocab = tokenizer.get_vocab()
    option_ids = {}
    for opt in options:
        if opt in vocab:
            option_ids[opt] = vocab[opt]
        elif f" {opt}" in vocab:
            option_ids[opt] = vocab[f" {opt}"]
        else:
            # Fallback: encode
            ids = tokenizer.encode(opt, add_special_tokens=False)
            option_ids[opt] = ids[0] if ids else None
    
    # Extract probs and logits for each option
    option_probs = {}
    option_logits = {}
    for opt, tid in option_ids.items():
        if tid is not None:
            option_probs[opt] = full_probs[tid].item()
            option_logits[opt] = logits[tid].item()
        else:
            option_probs[opt] = 0.0
            option_logits[opt] = float('-inf')
    
    # Predicted = highest prob among options
    predicted = max(option_probs, key=option_probs.get)
    
    # Also track overall max (Map the Flow tracks prob_max and prob_max_id)
    max_prob, max_id = torch.max(full_probs, dim=0)
    
    return {
        "probs": option_probs,           # P(A), P(B), P(C), P(D) from full vocab softmax
        "logits_raw": option_logits,      # raw logit values
        "predicted": predicted,           # highest prob option letter
        "predicted_id": max_id.item(),    # token id with highest prob in full vocab
        "max_prob": max_prob.item(),      # highest prob in full vocab
    }
 
 
def compute_p_change(base_result: dict, knockout_result: dict, ground_truth: str, base_predicted: str) -> dict:
    """Compute p_change metrics following Map the Flow.
    
    Map the Flow tracks TWO metrics:
      - relative_diff:    change in probability of the BASELINE predicted answer
      - relative_diff_gt: change in probability of the GROUND TRUTH answer
    
    Both computed as: (new_score - base_score) / base_score * 100
    """
    # GT answer probability change
    p_base_gt = base_result["probs"].get(ground_truth, 1e-10)
    p_knockout_gt = knockout_result["probs"].get(ground_truth, 0.0)
    relative_diff_gt = (p_knockout_gt - p_base_gt) / max(p_base_gt, 1e-10) * 100.0
    
    # Predicted answer probability change
    p_base_pred = base_result["probs"].get(base_predicted, 1e-10)
    p_knockout_pred = knockout_result["probs"].get(base_predicted, 0.0)
    relative_diff = (p_knockout_pred - p_base_pred) / max(p_base_pred, 1e-10) * 100.0
    
    return {
        "base_score": p_base_pred,           # P(predicted) baseline
        "new_score": p_knockout_pred,         # P(predicted) after knockout
        "relative_diff": relative_diff,       # % change in P(predicted)
        "gt_score": p_base_gt,               # P(GT) baseline
        "new_score_gt": p_knockout_gt,        # P(GT) after knockout
        "relative_diff_gt": relative_diff_gt, # % change in P(GT)
    }