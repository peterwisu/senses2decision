"""Reusable inference profiling utilities for lmms-eval models.

Metrics follow standard LLM inference conventions:
    - prefill_latency_ms:  time to process all input tokens (ms)
    - decode_latency_ms:   time per output token during decoding (ms)
    - decode_speed:        token generation rate (tokens/s)
    - peak_vram_gb:        peak GPU memory during generation (GB)

Usage in any model file:

    from lmms_eval.models.model_utils.profiling import PrefillTimingProcessor, log_generate_metrics

    timing_hook = PrefillTimingProcessor()
    torch.cuda.reset_peak_memory_stats(self.device)
    torch.cuda.synchronize()
    gen_start = time.perf_counter()

    output = model.generate(..., logits_processor=[timing_hook])

    torch.cuda.synchronize()
    gen_end = time.perf_counter()

    log_generate_metrics(
        gen_start=gen_start,
        gen_end=gen_end,
        timing_hook=timing_hook,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        device=self.device,
    )
"""

import time
from typing import Optional

import torch

from lmms_eval.models.model_utils.gen_metrics import log_metrics


class PrefillTimingProcessor:
    """LogitsProcessor that captures the exact moment prefill ends.

    HuggingFace's model.generate() calls LogitsProcessors after every
    forward pass.  The first call = prefill just finished.

    Passes scores through unmodified.  Only overhead is one
    cuda.synchronize() + perf_counter() on step 0.
    """

    def __init__(self):
        self.prefill_end_time: Optional[float] = None
        self._step: int = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._step == 0:
            torch.cuda.synchronize()
            self.prefill_end_time = time.perf_counter()
        self._step += 1
        return scores


def log_generate_metrics(
    gen_start: float,
    gen_end: float,
    timing_hook: PrefillTimingProcessor,
    num_input_tokens: int,
    num_output_tokens: int,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute profiling metrics and log them via the framework's log_metrics().

    Args:
        gen_start: perf_counter timestamp (seconds) before model.generate()
        gen_end: perf_counter timestamp (seconds) after model.generate()
        timing_hook: PrefillTimingProcessor passed into generate()
        num_input_tokens: number of input tokens in this batch
        num_output_tokens: number of generated output tokens in this batch
        device: CUDA device for VRAM measurement (None to skip)

    Returns:
        dict with all computed metrics for this batch
    """
    e2e_time = gen_end - gen_start

    # Split prefill vs decode using the hook timestamp
    if timing_hook.prefill_end_time is not None:
        prefill_time_s = timing_hook.prefill_end_time - gen_start
        decode_time_s = gen_end - timing_hook.prefill_end_time
        decode_latency_s = decode_time_s / max(num_output_tokens - 1, 1)
    else:
        # Hook never fired — 0 output tokens, all time is prefill
        prefill_time_s = e2e_time
        decode_latency_s = 0.0

    # Convert to proper units
    prefill_latency_ms = prefill_time_s * 1000                          # seconds → ms
    decode_latency_ms = decode_latency_s * 1000                         # seconds → ms
    decode_speed = (1.0 / decode_latency_s) if decode_latency_s > 0 else 0.0  # tokens/s

    additional = {
        "prefill_latency_ms": prefill_latency_ms,
        "decode_latency_ms": decode_latency_ms,
        "decode_speed": decode_speed,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
    }

    if device is not None:
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        additional["peak_vram_gb"] = peak_vram_bytes / (1024 ** 3)

    log_metrics(
        total_elapsed_time=e2e_time,
        total_gen_tokens=num_output_tokens,
        avg_speed=decode_speed,
        additional_metrics=additional,
    )

    return {"e2e_time_s": e2e_time, **additional}