"""
Knockout Hooks for Interleaved AV-Odyssey Experiments

Built on top of hooks.py (shared: _make_knockout_hook, extract_mcq_probs, compute_p_change).

Adds:
  - build_knockout_mask_interleaved() with cross_target support
  - KnockoutManagerInterleaved that uses KNOCKOUT_CONFIGS_INTERLEAVED + TokenPositionsInterleaved
"""

import torch
from typing import List, Set

from utils.knockout.hooks import (
    _make_knockout_hook,
)
from utils.knockout.configs_interleaved import (
    KNOCKOUT_CONFIGS_INTERLEAVED,
    KnockoutConfig,
    TokenPositionsInterleaved,
)


# ============================================================
# Build knockout mask (interleaved version)
# ============================================================

def build_knockout_mask_interleaved(
    config: KnockoutConfig,
    token_positions: TokenPositionsInterleaved,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorised knockout mask for interleaved AV-Odyssey layout.

    Supports cross_unit types: None, "cross_target".
    (cross_frame and cross_chunk are not used in interleaved configs.)

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
            # Block all pairs: mask[q, k] = -inf
            mask[q_idx.unsqueeze(1), k_idx.unsqueeze(0)] = neg_inf

        elif config.cross_unit == "cross_target":
            # Block target-target attention ACROSS different targets
            # Within same target is allowed (e.g. target_0 tokens can attend to each other)
            q_units = torch.tensor(
                [token_positions.get_target_unit_of(i.item()) for i in q_idx],
                device=device, dtype=torch.long
            )
            k_units = torch.tensor(
                [token_positions.get_target_unit_of(i.item()) for i in k_idx],
                device=device, dtype=torch.long
            )
            q_valid = q_units >= 0
            k_valid = k_units >= 0
            cross = (q_units.unsqueeze(1) != k_units.unsqueeze(0))
            valid = q_valid.unsqueeze(1) & k_valid.unsqueeze(0)
            block = cross & valid

            qi_grid = q_idx.unsqueeze(1).expand_as(block)
            ki_grid = k_idx.unsqueeze(0).expand_as(block)
            mask[qi_grid[block], ki_grid[block]] = neg_inf

        else:
            raise ValueError(
                f"Unknown cross_unit type: {config.cross_unit}. "
                f"Interleaved configs support: None, 'cross_target'."
            )

    return mask


# ============================================================
# Knockout Manager (interleaved version)
# ============================================================

class KnockoutManagerInterleaved:
    """Manages knockout hooks for interleaved AV-Odyssey experiments.

    Usage:
        manager = KnockoutManagerInterleaved(
            model, token_positions, config_id=1, target_layers=[0,1,2,3,4,5]
        )
        outputs = model(**inputs)  # hooks fire automatically
        manager.remove_hooks()

    Or as context manager:
        with KnockoutManagerInterleaved(model, tp, config_id=1, target_layers=layers) as mgr:
            outputs = model(**inputs)
    """

    def __init__(
        self,
        model,
        token_positions: TokenPositionsInterleaved,
        config_id: int,
        target_layers: List[int],
    ):
        """
        Args:
            model: Qwen2_5OmniForConditionalGeneration model
            token_positions: TokenPositionsInterleaved with region ranges
            config_id: Which interleaved knockout config (1-11)
            target_layers: List of layer indices where knockout is applied
        """
        self.config = KNOCKOUT_CONFIGS_INTERLEAVED[config_id]
        self.config_id = config_id
        self.target_layers = set(target_layers)
        self.hooks = []

        # Get the thinker's text model layers
        thinker = self._get_thinker(model)
        text_model = thinker.model

        # Determine device and dtype from model
        device = next(text_model.parameters()).device
        dtype = next(text_model.parameters()).dtype

        # Build the knockout mask [seq_len, seq_len]
        knockout_mask = build_knockout_mask_interleaved(
            self.config, token_positions, device, dtype
        )

        # Count blocked positions for logging
        self.num_blocked = (knockout_mask != 0).sum().item()
        self.seq_len = token_positions.seq_len

        # Expand to 4D: [1, 1, seq_len, seq_len]
        knockout_mask_4d = knockout_mask.unsqueeze(0).unsqueeze(0)

        # Register hooks on each attention module
        for layer in text_model.layers:
            attn = layer.self_attn
            layer_idx = attn.layer_idx

            hook_fn = _make_knockout_hook(knockout_mask_4d, layer_idx, self.target_layers)
            handle = attn.register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(handle)

    def _get_thinker(self, model):
        """Navigate model hierarchy to find the thinker component."""
        if hasattr(model, 'thinker'):
            return model.thinker
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model
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
            f"KnockoutManagerInterleaved(config={self.config_id} '{self.config.name}', "
            f"layers={sorted(self.target_layers)}, "
            f"blocked={self.num_blocked}/{self.seq_len**2} positions)"
        )