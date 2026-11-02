"""
Knockout Configurations for Omni-Modal LLM Analysis

Defines:
  - 8 knockout configurations (which attention edges to block)
  - Layer group definitions for 7B (28 layers) and 3B (36 layers)
  - Sliding window sweep logic
  - Token position utilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


# ============================================================
# Knockout Config Definitions
# ============================================================

@dataclass
class KnockoutConfig:
    """One knockout configuration specifying which attention edges to block."""
    id: int
    name: str
    short_name: str
    description: str
    # Which (query_type, key_type) pairs to block
    # Types: "video", "audio", "system", "user_instruction", "question_text", "last"
    blocked_pairs: List[Tuple[str, str]]
    # Whether blocking is cross-unit (across temporal patches / chunks)
    cross_unit: Optional[str] = None  # "cross_frame", "cross_chunk", None


# 8 Knockout Configs
KNOCKOUT_CONFIGS = {
    1: KnockoutConfig(
        id=1,
        name="cross_frame",
        short_name="CF",
        description="Block video tokens from attending to video tokens in different temporal patches",
        blocked_pairs=[("video", "video")],
        cross_unit="cross_frame",
    ),
    2: KnockoutConfig(
        id=2,
        name="cross_chunk_audio",
        short_name="CCA",
        description="Block audio tokens from attending to audio tokens in different chunks",
        blocked_pairs=[("audio", "audio")],
        cross_unit="cross_chunk",
    ),
    31: KnockoutConfig(
        id=31,
        name="video_to_audio",
        short_name="V2A",
        description="Block audio queries from reading video keys (video -> audio direction)",
        blocked_pairs=[("audio", "video")],
        cross_unit=None,
    ),
    32: KnockoutConfig(
        id=32,
        name="audio_to_video",
        short_name="A2V",
        description="Block video queries from reading audio keys (audio -> video direction)",
        blocked_pairs=[("video", "audio")],
        cross_unit=None,
    ),
    4: KnockoutConfig(
        id=4,
        name="video_to_question",
        short_name="V2Q",
        description="Block question text from reading video tokens",
        blocked_pairs=[("question_text", "video")],
        cross_unit=None,
    ),
    5: KnockoutConfig(
        id=5,
        name="audio_to_question",
        short_name="A2Q",
        description="Block question text from reading audio tokens",
        blocked_pairs=[("question_text", "audio")],
        cross_unit=None,
    ),
    6: KnockoutConfig(
        id=6,
        name="question_to_last",
        short_name="Q2L",
        description="Block last token from reading question text (not prompt suffix)",
        blocked_pairs=[("last", "question_text")],
        cross_unit=None,
    ),
    7: KnockoutConfig(
        id=7,
        name="video_to_last",
        short_name="V2L",
        description="Block last token from reading video tokens directly",
        blocked_pairs=[("last", "video")],
        cross_unit=None,
    ),
    8: KnockoutConfig(
        id=8,
        name="audio_to_last",
        short_name="A2L",
        description="Block last token from reading audio tokens directly",
        blocked_pairs=[("last", "audio")],
        cross_unit=None,
    ),
    9: KnockoutConfig(
        id=9,
        name="noq_to_last",
        short_name="NQ2L",
        description="Block last from reading non-option question stem",
        blocked_pairs=[("last", "non_option_question")],
        cross_unit=None,
    ),
    10: KnockoutConfig(
        id=10,
        name="true_to_last",
        short_name="TO2L",
        description="Block last from reading true option",
        blocked_pairs=[("last", "true_option")],
        cross_unit=None,
    ),
    11: KnockoutConfig(
        id=11,
        name="false_to_last",
        short_name="FO2L",
        description="Block last from reading false options",
        blocked_pairs=[("last", "false_options")],
        cross_unit=None,
    ),
    12: KnockoutConfig(
        id=12,
        name="video_to_true",
        short_name="V2TO",
        description="Block true option from reading video",
        blocked_pairs=[("true_option", "video")],
        cross_unit=None,
    ),
    13: KnockoutConfig(
        id=13,
        name="audio_to_true",
        short_name="A2TO",
        description="Block true option from reading audio",
        blocked_pairs=[("true_option", "audio")],
        cross_unit=None,
    ),
    14: KnockoutConfig(
        id=14,
        name="video_to_noq",
        short_name="V2NQ",
        description="Block non-option question from reading video",
        blocked_pairs=[("non_option_question", "video")],
        cross_unit=None,
    ),
    15: KnockoutConfig(
        id=15,
        name="audio_to_noq",
        short_name="A2NQ",
        description="Block non-option question from reading audio",
        blocked_pairs=[("non_option_question", "audio")],
        cross_unit=None,
    ),
    16: KnockoutConfig(
        id=16,
        name="noq_to_true",
        short_name="NQ2TO",
        description="Block true option from reading non-option question stem",
        blocked_pairs=[("true_option", "non_option_question")],
        cross_unit=None,
    ),
    17: KnockoutConfig(
        id=17,
        name="last_to_last",
        short_name="L2L",
        description="Block last token from attending to itself (sanity check control)",
        blocked_pairs=[("last", "last")],
        cross_unit=None,
    ),
    # ---- Combined modality knockouts (upper bound / redundancy tests) ----
    3: KnockoutConfig(
        id=3,
        name="audio_video",
        short_name="AV",
        description="Block audio<->video attention in both directions",
        blocked_pairs=[("audio", "video"), ("video", "audio")],
        cross_unit=None,
    ),
    18: KnockoutConfig(
        id=18,
        name="av_to_question",
        short_name="AV2Q",
        description="Block question text from reading BOTH video and audio tokens (combined, upper bound for 4+5)",
        blocked_pairs=[("question_text", "video"), ("question_text", "audio")],
        cross_unit=None,
    ),
    19: KnockoutConfig(
        id=19,
        name="av_to_last",
        short_name="AV2L",
        description="Block last token from reading BOTH video and audio tokens (combined, upper bound for 7+8)",
        blocked_pairs=[("last", "video"), ("last", "audio")],
        cross_unit=None,
    ),
    20: KnockoutConfig(
        id=20,
        name="av_to_true",
        short_name="AV2TO",
        description="Block true option from reading BOTH video and audio (combined, upper bound for 12+13)",
        blocked_pairs=[("true_option", "video"), ("true_option", "audio")],
        cross_unit=None,
    ),
    21: KnockoutConfig(
        id=21,
        name="av_to_noq",
        short_name="AV2NQ",
        description="Block non-option question from reading BOTH video and audio (combined, upper bound for 14+15)",
        blocked_pairs=[("non_option_question", "video"), ("non_option_question", "audio")],
        cross_unit=None,
    ),


    ## added for late layer experiment
    22: KnockoutConfig(
    id=22,
    name="all_text_to_video",
    short_name="T2V",
    description="Block all text tokens from reading video tokens",
    blocked_pairs=[
        ("instruction_prefix","video"),
        ("question_text", "video"),
        ("prompt_suffix", "video"),
        ("last", "video"),
    ],
    cross_unit=None,
    ),
    23: KnockoutConfig(
        id=23,
        name="all_text_to_av",
        short_name="T2AV",
        description="Block all text tokens from reading both video and audio tokens",
        blocked_pairs=[
            ("instruction_prefix","video"),
            ("question_text", "video"),
            ("prompt_suffix", "video"),
            ("last", "video"),
            ("instruction_prefix", "audio"),
            ("question_text", "audio"),
            ("prompt_suffix", "audio"),
            ("last", "audio"),
        ],
        cross_unit=None,
    ),
}


# ============================================================
# Layer Group Definitions
# ============================================================

# 7B model: 28 layers, 5 groups
LAYER_GROUPS_7B = {
    1: (0, 5),     # L0-L5   (6 layers)
    2: (6, 11),    # L6-L11  (6 layers)
    3: (12, 17),   # L12-L17 (6 layers)
    4: (18, 22),   # L18-L22 (5 layers)
    5: (23, 27),   # L23-L27 (5 layers)
}

# 3B model: 36 layers, 6 groups
LAYER_GROUPS_3B = {
    1: (0, 5),     # L0-L5
    2: (6, 11),    # L6-L11
    3: (12, 17),   # L12-L17
    4: (18, 23),   # L18-L23
    5: (24, 29),   # L24-L29
    6: (30, 35),   # L30-L35
}

LAYER_GROUPS = {
    "7B": LAYER_GROUPS_7B,
    "3B": LAYER_GROUPS_3B,
}


def get_layer_groups(model_size: str) -> Dict[int, Tuple[int, int]]:
    """Get layer groups for a model size."""
    if model_size not in LAYER_GROUPS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(LAYER_GROUPS.keys())}")
    return LAYER_GROUPS[model_size]


def get_num_layers(model_size: str) -> int:
    """Get total number of layers for a model size."""
    return {"7B": 28, "3B": 36}[model_size]


# ============================================================
# Sliding Window Logic
# ============================================================

def get_window_layers(center: int, k: int, num_layers: int) -> List[int]:
    """Get list of layer indices for a sliding window centered at `center`."""
    half = k // 2
    start = max(0, center - half)
    end = min(num_layers, center + (k - half))
    return list(range(start, end))


def get_all_window_sweeps(num_layers: int, k: int = 9, stride: int = 1) -> List[Dict]:
    """Generate all sliding window positions for a sweep."""
    sweeps = []
    for center in range(0, num_layers, stride):
        layers = get_window_layers(center, k, num_layers)
        sweeps.append({
            "center": center,
            "layers": layers,
            "layer_start": layers[0],
            "layer_end": layers[-1],
            "num_masked": len(layers),
        })
    return sweeps


# ============================================================
# Token Range Finding (same approach as Map the Flow)
# ============================================================

def find_token_range(tokenizer, input_ids, text_to_find):
    """Find where a text string appears in the tokenized input_ids.
    
    Tokenizes text_to_find and searches for the subsequence in input_ids.
    Returns (start, end) inclusive indices, or None if not found.
    
    This is the same approach as Map the Flow's find_token_range().
    """
    # Tokenize the target text (no special tokens)
    target_ids = tokenizer.encode(text_to_find, add_special_tokens=False)
    
    if len(target_ids) == 0:
        return None
    
    # Sliding window search
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i + len(target_ids)] == target_ids:
            return (i, i + len(target_ids) - 1)
    
    # BPE boundary mismatch fallback: try matching with some tolerance
    # Sometimes the tokenizer produces different tokens when text is tokenized
    # in isolation vs as part of a longer string. Try trimming first/last token.
    if len(target_ids) > 2:
        # Try without first token (BPE might merge differently at the start)
        inner = target_ids[1:]
        for i in range(len(input_ids) - len(inner) + 1):
            if input_ids[i:i + len(inner)] == inner:
                return (i - 1, i + len(inner) - 1)
        
        # Try without last token
        inner = target_ids[:-1]
        for i in range(len(input_ids) - len(inner) + 1):
            if input_ids[i:i + len(inner)] == inner:
                return (i, i + len(inner))

    # === Attempt 3: decoded string search (handles quotes/BPE mismatches) ===
    token_texts = [tokenizer.decode([tid]) for tid in input_ids]
    cumulative = ""
    token_starts = []
    for t in token_texts:
        token_starts.append(len(cumulative))
        cumulative += t

    pos = cumulative.find(text_to_find)

    # DEBUG: show mismatch
    # if pos == -1:
    #     print(f"ATTEMPT3 FAIL text_to_find[:100] = {repr(text_to_find[:100])}")
    #     # Find roughly where user_instruction starts (last 30% of string)
    #     start_idx = int(len(cumulative) * 0.7)
    #     print(f"ATTEMPT3 FAIL cumulative tail   = {repr(cumulative[start_idx:start_idx+200])}")
    
    if pos != -1:
        end_pos = pos + len(text_to_find)
        token_start = None
        token_end = None
        for idx in range(len(token_starts)):
            if token_start is None and token_starts[idx] + len(token_texts[idx]) > pos:
                token_start = idx
            if token_starts[idx] + len(token_texts[idx]) >= end_pos:
                token_end = idx
                break
        if token_start is not None and token_end is not None:
            return (token_start, token_end)
    
    return None


# ============================================================
# Token Position Extraction
# ============================================================

@dataclass
class TokenPositions:
    """Token position ranges for knockout mask building."""
    system: Tuple[int, int]                          # (start, end) inclusive
    video_chunks: List[Tuple[int, int]]              # list of (start, end) per chunk
    audio_chunks: List[Tuple[int, int]]              # list of (start, end) per chunk
    video_temporal_patches: List[Tuple[int, int]]    # list of (start, end) per temporal patch
    user_instruction: Tuple[int, int]                # (start, end) inclusive — full range after media
    last_token: int                                  # position of last token
    seq_len: int                                     # total sequence length
    # ---- question_text and prompt_suffix sub-ranges ----
    instruction_prefix: Optional[Tuple[int, int]] = None  # task instruction before question
    question_text: Optional[Tuple[int, int]] = None  # (start, end) — actual question + options only
    prompt_suffix: Optional[Tuple[int, int]] = None  # (start, end) — "Only give best option.\nAnswer:(" etc.

    # ---- option-level fields ----
    non_option_question: Optional[Tuple[int, int]] = None
    true_option: Optional[Tuple[int, int]] = None
    false_options: Optional[List[Tuple[int, int]]] = None

    def get_indices(self, token_type: str) -> List[int]:
        """Get all token indices for a given type."""
        if token_type == "system":
            return list(range(self.system[0], self.system[1] + 1))
        elif token_type == "video":
            indices = []
            for start, end in self.video_chunks:
                indices.extend(range(start, end + 1))
            return indices
        elif token_type == "audio":
            indices = []
            for start, end in self.audio_chunks:
                indices.extend(range(start, end + 1))
            return indices
        elif token_type == "user_instruction":
            # Full user instruction range, excluding last_token
            indices = list(range(self.user_instruction[0], self.user_instruction[1] + 1))
            if self.last_token in indices:
                indices.remove(self.last_token)
            return indices
        # ---- question_text type ----
        elif token_type == "question_text":
            if self.question_text is not None:
                return list(range(self.question_text[0], self.question_text[1] + 1))
            else:
                # Fallback: use user_instruction minus last_token (same as before)
                return self.get_indices("user_instruction")
        # ---- prompt_suffix type ----
        elif token_type == "prompt_suffix":
            if self.prompt_suffix is not None:
                return list(range(self.prompt_suffix[0], self.prompt_suffix[1] + 1))
            else:
                return []
        elif token_type == "instruction_prefix":
            if self.instruction_prefix is not None:
                return list(range(self.instruction_prefix[0], self.instruction_prefix[1] + 1))
            else:
                return []
        elif token_type == "non_option_question":
            if self.non_option_question is not None:
                return list(range(self.non_option_question[0], self.non_option_question[1] + 1))
            return self.get_indices("question_text")
        elif token_type == "true_option":
            if self.true_option is not None:
                return list(range(self.true_option[0], self.true_option[1] + 1))
            return []
        elif token_type == "false_options":
            if self.false_options is not None:
                indices = []
                for s, e in self.false_options:
                    indices.extend(range(s, e + 1))
                return indices
            return []
        elif token_type == "last":
            return [self.last_token]
    
        
        ## added for explierment on masking late layer
    
        else:
            raise ValueError(f"Unknown token type: {token_type}")

    def get_temporal_patch_of(self, token_idx: int) -> int:
        """Return which temporal patch a video token belongs to. -1 if not video."""
        for i, (start, end) in enumerate(self.video_temporal_patches):
            if start <= token_idx <= end:
                return i
        return -1

    def get_chunk_of(self, token_idx: int) -> int:
        """Return which chunk an audio token belongs to. -1 if not audio."""
        for i, (start, end) in enumerate(self.audio_chunks):
            if start <= token_idx <= end:
                return i
        return -1

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        d = {
            "system": list(self.system),
            "video_chunks": [list(c) for c in self.video_chunks],
            "audio_chunks": [list(c) for c in self.audio_chunks],
            "video_temporal_patches": [list(p) for p in self.video_temporal_patches],
            "user_instruction": list(self.user_instruction),
            "last_token": self.last_token,
            "seq_len": self.seq_len,
            "question_text": list(self.question_text) if self.question_text else None,
            "prompt_suffix": list(self.prompt_suffix) if self.prompt_suffix else None,
            "instruction_prefix": list(self.instruction_prefix) if self.instruction_prefix else None,

            "non_option_question": list(self.non_option_question) if self.non_option_question else None,
            "true_option": list(self.true_option) if self.true_option else None,
            "false_options": [list(f) for f in self.false_options] if self.false_options else None,
            
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TokenPositions":
        """Deserialise from dict."""
        return cls(
            system=tuple(d["system"]),
            video_chunks=[tuple(c) for c in d["video_chunks"]],
            audio_chunks=[tuple(c) for c in d["audio_chunks"]],
            video_temporal_patches=[tuple(p) for p in d["video_temporal_patches"]],
            user_instruction=tuple(d["user_instruction"]),
            last_token=d["last_token"],
            seq_len=d["seq_len"],
            # ---- Fields added later (backward compatible with old baseline.jsonl) ----
            question_text=tuple(d["question_text"]) if d.get("question_text") else None,
            prompt_suffix=tuple(d["prompt_suffix"]) if d.get("prompt_suffix") else None,
            instruction_prefix=tuple(d["instruction_prefix"]) if d.get("instruction_prefix") else None,

            non_option_question=tuple(d["non_option_question"]) if d.get("non_option_question") else None,
            true_option=tuple(d["true_option"]) if d.get("true_option") else None,
            false_options=[tuple(f) for f in d["false_options"]] if d.get("false_options") else None,
            
        )

def find_option_ranges(input_ids, question_range, answer_letter, tokenizer):
    """Find token ranges for each option A/B/C/D within question_text.
    
    Args:
        input_ids: list of token ids
        question_range: (start, end) inclusive of question_text
        answer_letter: "A", "B", "C", or "D"
        tokenizer: tokenizer instance
    
    Returns:
        question_stem: (start, end) — question text before options
        true_option: (start, end) — the correct option
        false_options: list of (start, end) — the incorrect options
        Or (None, None, None) if parsing fails
    """
    q_start, q_end = question_range

    option_token_ids = {
        letter: tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ["A", "B", "C", "D"]
    }
    dot_id = tokenizer.encode(".", add_special_tokens=False)[0]

    # Find where each option starts: letter followed by "."
    option_starts = {}
    for letter, tok_id in option_token_ids.items():
        for i in range(q_start, q_end + 1):
            if input_ids[i] == tok_id and i + 1 <= q_end and input_ids[i + 1] == dot_id:
                option_starts[letter] = i
                break

    if len(option_starts) < 2:
        print(f"WARNING: find_option_ranges found only {len(option_starts)} options. "
              f"Expected 4. Skipping option parsing.")
        return None, None, None

    # Compute ranges
    letters = sorted(option_starts.keys(), key=lambda x: option_starts[x])
    option_ranges = {}
    for idx, letter in enumerate(letters):
        start = option_starts[letter]
        if idx + 1 < len(letters):
            end = option_starts[letters[idx + 1]] - 1
        else:
            end = q_end
        option_ranges[letter] = (start, end)

    # Question stem = question_text start to first option - 1
    first_option_start = option_starts[letters[0]]
    question_stem = (q_start, first_option_start - 1)

    # Split true/false
    if answer_letter not in option_ranges:
        print(f"WARNING: answer_letter '{answer_letter}' not found in parsed options {list(option_ranges.keys())}")
        return None, None, None

    true_option = option_ranges[answer_letter]
    false_options = [option_ranges[l] for l in letters if l != answer_letter]

    return question_stem, true_option, false_options

def extract_token_positions(
    input_ids,
    special_tokens,
    tokens_per_temporal_patch=144,
    question_text_str=None,      # the raw question + options string from dataset
    tokenizer=None,              # needed to tokenize question_text_str
    answer_letter=None,  # "A", "B", "C", or "D"
) -> TokenPositions:
    """Extract token positions from input_ids and special tokens.
    
    Expected token order:
      [system ~45] [V0_F0 V0_F1 A0] [V1_F0 V1_F1 A1] ... [user_instruction ~157]
    
    Args:
        input_ids: list of token ids
        special_tokens: dict with keys video, audio, vision_start, vision_end, audio_start, audio_end
        tokens_per_temporal_patch: number of video tokens per temporal patch (default 144)
        question_text_str: raw question + options string to find in tokens
        tokenizer: tokenizer instance for encoding question_text_str
    """
    video_id = special_tokens["video"]
    audio_id = special_tokens["audio"]
    vision_start_id = special_tokens["vision_start"]
    vision_end_id = special_tokens["vision_end"]

    n = len(input_ids)

    # Find first vision_start and last vision_end
    vision_bos = None
    vision_eos = None
    for i, tok in enumerate(input_ids):
        if tok == vision_start_id and vision_bos is None:
            vision_bos = i
        if tok == vision_end_id:
            vision_eos = i

    # Label each token
    labels = []
    for i, tok in enumerate(input_ids):
        if tok == video_id:
            labels.append("video")
        elif tok == audio_id:
            labels.append("audio")
        elif vision_bos is not None and i <= vision_bos:
            labels.append("system")
        elif vision_eos is not None and i >= vision_eos:
            labels.append("user_instruction")
        else:
            labels.append("system")

    # System range
    sys_start = 0
    sys_end = vision_bos if vision_bos is not None else 0

    # User instruction range
    usr_start = vision_eos if vision_eos is not None else n - 1
    usr_end = n - 1

    # Extract video and audio chunks
    video_chunks = []
    audio_chunks = []
    video_temporal_patches = []

    i = 0
    while i < n:
        if labels[i] == "video":
            # Start of a video segment
            seg_start = i
            while i < n and labels[i] == "video":
                i += 1
            seg_end = i - 1
            video_chunks.append((seg_start, seg_end))

            # Split into temporal patches
            seg_len = seg_end - seg_start + 1
            for offset in range(0, seg_len, tokens_per_temporal_patch):
                patch_start = seg_start + offset
                patch_end = min(seg_start + offset + tokens_per_temporal_patch - 1, seg_end)
                video_temporal_patches.append((patch_start, patch_end))

        elif labels[i] == "audio":
            seg_start = i
            while i < n and labels[i] == "audio":
                i += 1
            seg_end = i - 1
            audio_chunks.append((seg_start, seg_end))
        else:
            i += 1

    # ---- Find question_text boundary within user_instruction ----
    q_text_range = None
    p_suffix_range = None
    instruction_prefix_range = None

    if question_text_str is not None and tokenizer is not None:
        found = find_token_range(tokenizer, input_ids, question_text_str)
        
        if found is not None:
            q_start, q_end = found
            q_text_range = (q_start, q_end)
            
            # prompt_suffix = everything after question_text, before last_token
            suffix_start = q_end + 1
            suffix_end = n - 2  # n-1 is last_token, so suffix goes up to n-2
            if suffix_start <= suffix_end:
                p_suffix_range = (suffix_start, suffix_end)
        else:
            # Fallback: could not find question tokens
            # Treat everything in user_instruction (minus last_token) as question_text
            q_text_range = (usr_start, n - 2)
            print(f"WARNING: Could not find question_text in input_ids. "
                  f"Using full user_instruction range as fallback. "
                  f"question_text_str[:80]={question_text_str[:80]!r}")

    if q_text_range is not None:
        prefix_start = usr_start  # vision_eos
        prefix_end = q_text_range[0] - 1
        if prefix_start < prefix_end:
            instruction_prefix_range = (prefix_start, prefix_end)

    noq_range = None
    true_opt_range = None
    false_opt_ranges = None
    if q_text_range is not None and answer_letter is not None and tokenizer is not None:
        noq_range, true_opt_range, false_opt_ranges = find_option_ranges(
            input_ids, q_text_range, answer_letter, tokenizer
        )

    return TokenPositions(
        system=(sys_start, sys_end),
        video_chunks=video_chunks,
        audio_chunks=audio_chunks,
        video_temporal_patches=video_temporal_patches,
        user_instruction=(usr_start, usr_end),
        last_token=n - 1,
        seq_len=n,
        question_text=q_text_range,       
        prompt_suffix=p_suffix_range,      
        instruction_prefix=instruction_prefix_range,
        non_option_question=noq_range,
        true_option=true_opt_range,
        false_options=false_opt_ranges,
    )



def extract_token_positions_vs2(
    input_ids,
    special_tokens,
    question_text_str=None,
    tokenizer=None,
    answer_letter=None,
    video_grid_thw=None,   # pass [[T, H, W]] for proper patch sizing
) -> TokenPositions:
    """Extract token positions from video-SALMONN-2+ input_ids.

    Expected token layout:
      [system block] [<|vision_start|>] [video_pad/audio_pad interleaved] [<|vision_end|>] 
      [question + options] [suffix] [last]

    Differences from Qwen2.5-Omni:
      - vision_start = <|vision_start|> (id 151652), not <|vision_bos|>
      - vision_end = <|vision_end|> (id 151653), not <|vision_eos|>
      - video token = <|video_pad|> (id 151656)
      - audio token = <|audio_pad|> (id 151675)
      - No separate audio_start/audio_end markers; audio lives inside vision_start/vision_end
      - Temporal patch size = (H * W / 4) tokens per 2-frame patch (2x2 spatial merge)

    Args:
        input_ids: list of token ids
        special_tokens: dict with keys video, audio, vision_start, vision_end
        question_text_str: raw question + options string to find in tokens
        tokenizer: tokenizer instance
        answer_letter: "A", "B", "C", or "D" for option parsing
        video_grid_thw: [[T, H, W]] from the processor output, used to compute
                        tokens_per_temporal_patch. Optional.
    """
    video_id = special_tokens["video"]
    audio_id = special_tokens["audio"]
    vision_start_id = special_tokens["vision_start"]
    vision_end_id = special_tokens["vision_end"]

    n = len(input_ids)

    # ---- Find vision_start and vision_end ----
    vision_bos = None
    vision_eos = None
    for i, tok in enumerate(input_ids):
        if tok == vision_start_id and vision_bos is None:
            vision_bos = i
        if tok == vision_end_id:
            vision_eos = i   # take the last one

    # ---- Label each token ----
    labels = []
    for i, tok in enumerate(input_ids):
        if tok == video_id:
            labels.append("video")
        elif tok == audio_id:
            labels.append("audio")
        elif vision_bos is not None and i <= vision_bos:
            labels.append("system")
        elif vision_eos is not None and i >= vision_eos:
            labels.append("user_instruction")
        else:
            labels.append("system")

    # ---- System range ----
    sys_start = 0
    sys_end = vision_bos if vision_bos is not None else 0

    # ---- User instruction range ----
    usr_start = vision_eos if vision_eos is not None else n - 1
    usr_end = n - 1

    # ---- Compute tokens per temporal patch from video_grid_thw ----
    # Video-SALMONN uses 2x2 spatial merge + temporal patch size 2 (frame pairs).
    # video_grid_thw = [[T_patches, H_patches, W_patches]]
    # Total video tokens = T * H * W / 4 (2x2 spatial merge)
    # Tokens per temporal patch (1 T step = 2 frames merged) = H * W / 4
    if video_grid_thw is not None and len(video_grid_thw) > 0:
        T, H, W = video_grid_thw[0]
        tokens_per_temporal_patch = (H * W) // 4
    else:
        # Fallback: use a default that matches typical Video-SALMONN setups
        # For 224x420 frames with patch_size=14, H=16, W=30 -> 120 tokens/patch
        tokens_per_temporal_patch = 120

    # ---- Extract video and audio chunks ----
    video_chunks = []
    audio_chunks = []
    video_temporal_patches = []

    i = 0
    while i < n:
        if labels[i] == "video":
            seg_start = i
            while i < n and labels[i] == "video":
                i += 1
            seg_end = i - 1
            video_chunks.append((seg_start, seg_end))

            # Split into temporal patches
            seg_len = seg_end - seg_start + 1
            for offset in range(0, seg_len, tokens_per_temporal_patch):
                patch_start = seg_start + offset
                patch_end = min(
                    seg_start + offset + tokens_per_temporal_patch - 1,
                    seg_end,
                )
                video_temporal_patches.append((patch_start, patch_end))

        elif labels[i] == "audio":
            seg_start = i
            while i < n and labels[i] == "audio":
                i += 1
            seg_end = i - 1
            audio_chunks.append((seg_start, seg_end))
        else:
            i += 1

    # ---- Find question_text boundary within user_instruction ----
    q_text_range = None
    p_suffix_range = None
    instruction_prefix_range = None

    if question_text_str is not None and tokenizer is not None:
        found = find_token_range(tokenizer, input_ids, question_text_str)

        if found is not None:
            q_start, q_end = found
            q_text_range = (q_start, q_end)

            # prompt_suffix = everything after question_text, before last_token
            suffix_start = q_end + 1
            suffix_end = n - 2
            if suffix_start <= suffix_end:
                p_suffix_range = (suffix_start, suffix_end)
        else:
            # Fallback
            q_text_range = (usr_start, n - 2)
            print(
                f"WARNING: Could not find question_text in input_ids. "
                f"Using full user_instruction range as fallback. "
                f"question_text_str[:80]={question_text_str[:80]!r}"
            )

    if q_text_range is not None:
        prefix_start = usr_start
        prefix_end = q_text_range[0] - 1
        if prefix_start < prefix_end:
            instruction_prefix_range = (prefix_start, prefix_end)

    # ---- Option-level parsing ----
    noq_range = None
    true_opt_range = None
    false_opt_ranges = None
    if q_text_range is not None and answer_letter is not None and tokenizer is not None:
        noq_range, true_opt_range, false_opt_ranges = find_option_ranges(
            input_ids, q_text_range, answer_letter, tokenizer
        )

    return TokenPositions(
        system=(sys_start, sys_end),
        video_chunks=video_chunks,
        audio_chunks=audio_chunks,
        video_temporal_patches=video_temporal_patches,
        user_instruction=(usr_start, usr_end),
        last_token=n - 1,
        seq_len=n,
        question_text=q_text_range,
        prompt_suffix=p_suffix_range,
        instruction_prefix=instruction_prefix_range,
        non_option_question=noq_range,
        true_option=true_opt_range,
        false_options=false_opt_ranges,
    )
# ============================================================
# Utility: Result file naming
# ============================================================

def get_result_filename(config_id: int, sweep_type: str, layer_start: int, layer_end: int) -> str:
    """Generate result filename for a knockout run."""
    config = KNOCKOUT_CONFIGS[config_id]
    return f"config{config_id}_{config.name}_{sweep_type}_L{layer_start}-{layer_end}.jsonl"


def get_baseline_filename() -> str:
    return "baseline.jsonl"


# ============================================================
# Summary / Debug
# ============================================================

def print_config_summary():
    """Print all configs for verification."""
    print("=" * 70)
    print("KNOCKOUT CONFIGURATIONS")
    print("=" * 70)
    for cid, cfg in KNOCKOUT_CONFIGS.items():
        print(f"\nConfig {cid}: {cfg.name}")
        print(f"  Description: {cfg.description}")
        print(f"  Blocked pairs: {cfg.blocked_pairs}")
        if cfg.cross_unit:
            print(f"  Cross-unit: {cfg.cross_unit}")


def print_sweep_summary(model_size: str, sweep_type: str, k: int = 9):
    """Print sweep plan for verification."""
    num_layers = get_num_layers(model_size)

    print(f"\n{'=' * 70}")
    print(f"SWEEP PLAN: {model_size} model, {sweep_type}")
    print(f"{'=' * 70}")

    if sweep_type == "group":
        raise NotImplementedError("Group sweep is no longer supported. Use window sweep instead.")
        groups = get_layer_groups(model_size)
        total_runs = len(KNOCKOUT_CONFIGS) * len(groups)
        print(f"  {len(KNOCKOUT_CONFIGS)} configs x {len(groups)} groups = {total_runs} runs")
        for gid, (start, end) in groups.items():
            print(f"  Group {gid}: L{start}-L{end} ({end - start + 1} layers)")

    elif sweep_type == "window":
        sweeps = get_all_window_sweeps(num_layers, k)
        total_runs = len(KNOCKOUT_CONFIGS) * len(sweeps)
        print(f"  {len(KNOCKOUT_CONFIGS)} configs x {len(sweeps)} positions = {total_runs} runs")
        print(f"  Window k={k}, stride=1")
        for s in sweeps[:3]:
            print(f"  Center {s['center']}: L{s['layer_start']}-L{s['layer_end']} ({s['num_masked']} layers)")
        print(f"  ...")
        for s in sweeps[-3:]:
            print(f"  Center {s['center']}: L{s['layer_start']}-L{s['layer_end']} ({s['num_masked']} layers)")


if __name__ == "__main__":
    print_config_summary()
    print_sweep_summary("7B", "group")
    print_sweep_summary("7B", "window", k=9)