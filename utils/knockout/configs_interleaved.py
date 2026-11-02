"""
Knockout Configurations for Interleaved AV-Odyssey Analysis

Built on top of configs.py (shared: KnockoutConfig, layer groups, window sweep, find_token_range).

Adds:
  - AV-Odyssey structure detection (decompose_sample)
  - Option parsing (audit_sample_options)
  - Interleaved knockout configs (question/query/targets/context token types)
  - TokenPositionsInterleaved dataclass
  - extract_token_positions_interleaved()
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ============================================================
# Imports from original configs.py (shared, no duplication)
# ============================================================

from utils.knockout.configs import (
    KnockoutConfig,
    LAYER_GROUPS,
    get_layer_groups,
    get_num_layers,
    get_window_layers,
    get_all_window_sweeps,
    find_token_range,
    get_baseline_filename,
)


# ============================================================
# AV-Odyssey structure detection
# ============================================================

def _count_words(text):
    return len(re.findall(r"\b[a-z']+\b", text.lower()))


def _identify_query_targets(question_text):
    """Identify query (singleton) and targets (multiplet) media from placeholders."""
    matches = list(re.finditer(r"\[(audio|video|img)(\d+)\]", question_text))
    audio_matches = [m for m in matches if m.group(1) == "audio"]
    img_matches = [m for m in matches if m.group(1) in ("img", "video")]

    if len(audio_matches) == 1 and len(img_matches) == 1:
        return {"query_match": None, "target_matches": [], "context_matches": matches,
                "query_modality": None, "target_modality": None,
                "structure_kind": "context"}
    if len(audio_matches) == 1 and len(img_matches) > 1:
        return {"query_match": audio_matches[0], "target_matches": img_matches,
                "context_matches": [],
                "query_modality": "audio", "target_modality": "image",
                "structure_kind": "query_target"}
    if len(img_matches) == 1 and len(audio_matches) > 1:
        return {"query_match": img_matches[0], "target_matches": audio_matches,
                "context_matches": [],
                "query_modality": "image", "target_modality": "audio",
                "structure_kind": "query_target"}
    return {"query_match": None, "target_matches": [], "context_matches": matches,
            "query_modality": None, "target_modality": None,
            "structure_kind": "unknown"}


def decompose_sample(question_text):
    """Decompose a question into media + AT MOST ONE question text segment.

    Returns structure_kind, structure_str, query/target modalities,
    and the ordered list of media placeholder matches for position mapping.
    """
    info = _identify_query_targets(question_text)
    structure_kind = info["structure_kind"]

    media_spans = []
    if structure_kind == "query_target":
        media_spans.append((info["query_match"].start(), info["query_match"].end(), "query"))
        t0 = info["target_matches"][0].start()
        t1 = info["target_matches"][-1].end()
        media_spans.append((t0, t1, "targets"))
    else:
        for m in info["context_matches"]:
            media_spans.append((m.start(), m.end(), "context"))

    media_spans.sort(key=lambda x: x[0])

    raw_segments = []
    cursor = 0
    for span_start, span_end, span_label in media_spans:
        if cursor < span_start:
            text_chunk = question_text[cursor:span_start]
            if text_chunk.strip() and _count_words(text_chunk) > 0:
                raw_segments.append({"label": "text",
                                     "start": cursor, "end": span_start,
                                     "n_words": _count_words(text_chunk)})
        raw_segments.append({"label": span_label,
                             "start": span_start, "end": span_end,
                             "n_words": 0})
        cursor = span_end
    if cursor < len(question_text):
        text_chunk = question_text[cursor:]
        if text_chunk.strip() and _count_words(text_chunk) > 0:
            raw_segments.append({"label": "text",
                                 "start": cursor, "end": len(question_text),
                                 "n_words": _count_words(text_chunk)})

    question_idx = None
    text_segs = [s for s in raw_segments if s["label"] == "text"]
    media_only = [s for s in raw_segments if s["label"] != "text"]
    query_is_last_media = (
        structure_kind == "query_target"
        and len(media_only) > 0
        and media_only[-1]["label"] == "query"
    )

    if query_is_last_media:
        query_pos = None
        for i, s in enumerate(raw_segments):
            if s["label"] == "query":
                query_pos = i
                break
        if query_pos is not None and query_pos > 0 and raw_segments[query_pos - 1]["label"] == "text":
            chosen = raw_segments[query_pos - 1]
            question_idx = (chosen["start"], chosen["end"])
        elif text_segs:
            chosen = text_segs[0]
            for seg in text_segs[1:]:
                if seg["n_words"] >= chosen["n_words"]:
                    chosen = seg
            question_idx = (chosen["start"], chosen["end"])
    elif text_segs:
        chosen = text_segs[0]
        for seg in text_segs[1:]:
            if seg["n_words"] >= chosen["n_words"]:
                chosen = seg
        question_idx = (chosen["start"], chosen["end"])

    final_labels = []
    for seg in raw_segments:
        if seg["label"] != "text":
            final_labels.append(seg["label"])
        elif question_idx is not None and (seg["start"], seg["end"]) == question_idx:
            final_labels.append("question")

    structure_str = " → ".join(final_labels)

    # Build ordered media list for token position mapping
    ordered_media = []
    if structure_kind == "query_target":
        all_matches = []
        qm = info["query_match"]
        all_matches.append({"match": qm, "role": "query"})
        for i, tm in enumerate(info["target_matches"]):
            all_matches.append({"match": tm, "role": f"target_{i}"})
        all_matches.sort(key=lambda x: x["match"].start())
        ordered_media = [{"role": m["role"], "placeholder": m["match"].group()} for m in all_matches]
    elif structure_kind == "context":
        for i, cm in enumerate(info["context_matches"]):
            ordered_media.append({"role": f"context_{i}", "placeholder": cm.group()})

    return {"structure_kind": structure_kind,
            "query_modality": info["query_modality"],
            "target_modality": info["target_modality"],
            "structure_str": structure_str,
            "ordered_media": ordered_media}


# ============================================================
# Option parsing
# ============================================================

ORDINAL_TO_INDEX = {
    "first": 0, "second": 1, "third": 2, "fourth": 3,
    "1st": 0, "2nd": 1, "3rd": 2, "4th": 3,
    "one": 0, "two": 1, "three": 2, "four": 3,
}

MODALITY_WORDS = {
    "image": "image", "images": "image", "picture": "image", "pictures": "image",
    "photo": "image", "photos": "image", "illustration": "image", "illustrations": "image",
    "audio": "audio", "audios": "audio", "sound": "audio", "sounds": "audio",
    "clip": "audio", "clips": "audio", "recording": "audio", "recordings": "audio",
    "video": "video", "videos": "video",
}


def parse_option(option_text):
    """Parse 'the third image' -> {'index': 2, 'modality': 'image'}. None if unparseable."""
    if not isinstance(option_text, str):
        return None
    s = option_text.lower().strip()
    words = re.findall(r"\b[a-z\d]+\b", s)
    if not words:
        return None
    found_index = None
    found_modality = None
    for w in words:
        if found_index is None and w in ORDINAL_TO_INDEX:
            found_index = ORDINAL_TO_INDEX[w]
        if found_modality is None and w in MODALITY_WORDS:
            found_modality = MODALITY_WORDS[w]
        if found_index is None and w in {"1", "2", "3", "4"}:
            found_index = int(w) - 1
    if found_index is None:
        return None
    return {"index": found_index, "modality": found_modality}


def audit_sample_options(doc):
    """Audit one doc's options and compute correct_target_index."""
    options = doc.get("options", None)
    answer = doc.get("answer", None)

    if options is None or not hasattr(options, "__len__"):
        return {"parsed_options": [], "correct_target_index": None,
                "all_parsed": False, "full_coverage": False,
                "modality_consistent": False, "issues": ["no options"]}

    parsed = [parse_option(opt) for opt in options]
    issues = []

    unparseable = [i for i, p in enumerate(parsed) if p is None]
    all_parsed = len(unparseable) == 0
    if not all_parsed:
        issues.append(f"{len(unparseable)} option(s) unparseable")

    indices = [p["index"] for p in parsed if p is not None]
    coverage = set(indices)
    full_coverage = coverage == {0, 1, 2, 3}
    if all_parsed and not full_coverage:
        issues.append(f"missing positions: {sorted({0, 1, 2, 3} - coverage)}")
    if len(indices) != len(set(indices)):
        issues.append("duplicate positions")

    modalities = set(p["modality"] for p in parsed if p is not None and p["modality"])
    modality_consistent = len(modalities) <= 1
    if not modality_consistent:
        issues.append(f"mixed modalities: {sorted(modalities)}")

    correct_target_index = None
    if isinstance(answer, str) and len(answer) == 1:
        ans_idx = ord(answer.upper()) - ord("A")
        if 0 <= ans_idx < len(parsed) and parsed[ans_idx] is not None:
            correct_target_index = parsed[ans_idx]["index"]

    return {"parsed_options": parsed,
            "correct_target_index": correct_target_index,
            "all_parsed": all_parsed,
            "full_coverage": full_coverage,
            "modality_consistent": modality_consistent,
            "issues": issues}


# ============================================================
# Interleaved Knockout Configs
# ============================================================

# KNOCKOUT_CONFIGS_INTERLEAVED = {
#     # ---- Independence: do targets interact with each other? ----
#     1: KnockoutConfig(
#         id=1, name="cross_target", short_name="CrT",
#         description="Block attention between individual targets (each target isolated from others)",
#         blocked_pairs=[("targets", "targets")],
#         cross_unit="cross_target",
#     ),

#     # ---- Cross-modal: do query and targets communicate? ----
#     2: KnockoutConfig(
#         id=2, name="query_targets_bidirectional", short_name="QT",
#         description="Block attention between query and targets in both directions",
#         blocked_pairs=[("query", "targets"), ("targets", "query")],
#     ),

#     # ---- Text-media: do question text and targets communicate? ----
#     3: KnockoutConfig(
#         id=3, name="question_targets_bidirectional", short_name="QN_T",
#         description="Block attention between question text and targets in both directions",
#         blocked_pairs=[("question", "targets"), ("targets", "question")],
#     ),

#     # ---- Region-to-last: what does the decision token need? ----
#     4: KnockoutConfig(
#         id=4, name="query_to_last", short_name="Q2L",
#         description="Block last token from reading query (singleton) media tokens",
#         blocked_pairs=[("last", "query")],
#     ),
#     5: KnockoutConfig(
#         id=5, name="targets_to_last", short_name="T2L",
#         description="Block last token from reading all target media tokens",
#         blocked_pairs=[("last", "targets")],
#     ),
#     6: KnockoutConfig(
#         id=6, name="question_to_last", short_name="QN2L",
#         description="Block last token from reading question text tokens",
#         blocked_pairs=[("last", "question")],
#     ),

#     # ---- Per-target: correct vs incorrect ----
#     7: KnockoutConfig(
#         id=7, name="correct_target_to_last", short_name="CT2L",
#         description="Block last token from reading ONLY the correct target media tokens",
#         blocked_pairs=[("last", "target_correct")],
#     ),
#     8: KnockoutConfig(
#         id=8, name="incorrect_targets_to_last", short_name="IT2L",
#         description="Block last token from reading ONLY the incorrect target media tokens",
#         blocked_pairs=[("last", "target_incorrect")],
#     ),
#     # ---- Option-level: correct vs incorrect option letters ----
#     9: KnockoutConfig(
#         id=9, name="correct_option_to_last", short_name="CO2L",
#         description="Block last token from reading the correct option letter text",
#         blocked_pairs=[("last", "option_correct")],
#     ),
#     10: KnockoutConfig(
#         id=10, name="incorrect_options_to_last", short_name="IO2L",
#         description="Block last token from reading the incorrect option letter texts",
#         blocked_pairs=[("last", "option_incorrect")],
#     ),
#     # ---- Media-to-correct-option: does the correct option need to see the media? ----
#     11: KnockoutConfig(
#         id=11, name="correct_target_to_correct_option", short_name="CT2CO",
#         description="Block correct option letter from reading ONLY the correct target media",
#         blocked_pairs=[("option_correct", "target_correct")],
#     ),
#     12: KnockoutConfig(
#         id=12, name="incorrect_targets_to_correct_option", short_name="IT2CO",
#         description="Block correct option letter from reading ONLY the incorrect target media",
#         blocked_pairs=[("option_correct", "target_incorrect")],
#     ),
#     13: KnockoutConfig(
#         id=13, name="query_to_correct_option", short_name="Q2CO",
#         description="Block correct option letter from reading query media tokens",
#         blocked_pairs=[("option_correct", "query")],
#     ),
# }

KNOCKOUT_CONFIGS_INTERLEAVED = {
    # ---- Independence: do targets interact with each other? ----
    1: KnockoutConfig(
        id=1, name="cross_target", short_name="CrT",
        description="Block attention between individual targets (each target isolated from others)",
        blocked_pairs=[("targets", "targets")],
        cross_unit="cross_target",
    ),
    # ---- Cross-modal / Text-media interactions ----
    2: KnockoutConfig(
        id=2, name="query_targets_bidirectional", short_name="QT",
        description="Block attention between query and targets in both directions",
        blocked_pairs=[("query", "targets"), ("targets", "query")],
    ),
    3: KnockoutConfig(
        id=3, name="question_targets_bidirectional", short_name="QN_T",
        description="Block attention between question text and targets in both directions",
        blocked_pairs=[("question", "targets"), ("targets", "question")],
    ),
    4: KnockoutConfig(
        id=4, name="question_query_bidirectional", short_name="QN_Q",
        description="Block attention between question text and query media in both directions",
        blocked_pairs=[("question", "query"), ("query", "question")],
    ),
    # ---- Region-to-last: what does the decision token need? ----
    5: KnockoutConfig(
        id=5, name="query_to_last", short_name="Q2L",
        description="Block last token from reading query (singleton) media tokens",
        blocked_pairs=[("last", "query")],
    ),
    6: KnockoutConfig(
        id=6, name="targets_to_last", short_name="T2L",
        description="Block last token from reading all target media tokens",
        blocked_pairs=[("last", "targets")],
    ),
    7: KnockoutConfig(
        id=7, name="question_to_last", short_name="QN2L",
        description="Block last token from reading question text tokens",
        blocked_pairs=[("last", "question")],
    ),
    # ---- Per-target: correct vs incorrect ----
    8: KnockoutConfig(
        id=8, name="correct_target_to_last", short_name="CT2L",
        description="Block last token from reading ONLY the correct target media tokens",
        blocked_pairs=[("last", "target_correct")],
    ),
    9: KnockoutConfig(
        id=9, name="incorrect_targets_to_last", short_name="IT2L",
        description="Block last token from reading ONLY the incorrect target media tokens",
        blocked_pairs=[("last", "target_incorrect")],
    ),
    # ---- Option-level: correct vs incorrect option letters ----
    10: KnockoutConfig(
        id=10, name="correct_option_to_last", short_name="CO2L",
        description="Block last token from reading the correct option letter text",
        blocked_pairs=[("last", "option_correct")],
    ),
    11: KnockoutConfig(
        id=11, name="incorrect_options_to_last", short_name="IO2L",
        description="Block last token from reading the incorrect option letter texts",
        blocked_pairs=[("last", "option_incorrect")],
    ),
    # ---- Media-to-correct-option: does the correct option need to see the media? ----
    12: KnockoutConfig(
        id=12, name="correct_target_to_correct_option", short_name="CT2CO",
        description="Block correct option letter from reading ONLY the correct target media",
        blocked_pairs=[("option_correct", "target_correct")],
    ),
    13: KnockoutConfig(
        id=13, name="incorrect_targets_to_correct_option", short_name="IT2CO",
        description="Block correct option letter from reading ONLY the incorrect target media",
        blocked_pairs=[("option_correct", "target_incorrect")],
    ),
    14: KnockoutConfig(
        id=14, name="query_to_correct_option", short_name="Q2CO",
        description="Block correct option letter from reading query media tokens",
        blocked_pairs=[("option_correct", "query")],
    ),
    15: KnockoutConfig(
        id=15, name="question_to_correct_option", short_name="QN2CO",
        description="Block correct option letter from reading question text tokens",
        blocked_pairs=[("option_correct", "question")],
    ),
}


# ============================================================
# Interleaved Token Positions
# ============================================================

@dataclass
class TokenPositionsInterleaved:
    """Token positions for interleaved media layout (AV-Odyssey)."""
    system: Tuple[int, int]
    question: List[Tuple[int, int]]                      # question text (ONE range only)
    query: Optional[Tuple[int, int]]                     # singleton media
    targets: Optional[List[Tuple[int, int]]]             # per-target media [t0, t1, t2, t3]
    contexts: Optional[List[Tuple[int, int]]]            # task 26 [ctx0, ctx1]
    last_token: int
    seq_len: int
    structure_kind: str
    structure_str: str
    query_modality: Optional[str]
    target_modality: Optional[str]
    correct_target_index: Optional[int]
    answer_letter: Optional[str] = None
    option_ranges: Optional[List[Tuple[int, int]]] = None  # [optA, optB, optC, optD]
    prompt_suffix: Optional[Tuple[int, int]] = None

    def get_indices(self, token_type: str) -> List[int]:
        if token_type == "system":
            return list(range(self.system[0], self.system[1] + 1))
        elif token_type == "question":
            indices = []
            for start, end in self.question:
                indices.extend(range(start, end + 1))
            return indices
        elif token_type == "query":
            if self.query is not None:
                return list(range(self.query[0], self.query[1] + 1))
            return []
        elif token_type == "targets":
            indices = []
            if self.targets is not None:
                for start, end in self.targets:
                    indices.extend(range(start, end + 1))
            return indices
        elif token_type.startswith("target_"):
            suffix = token_type.split("_", 1)[1]
            if suffix == "correct":
                if self.correct_target_index is not None and self.targets is not None:
                    t = self.targets[self.correct_target_index]
                    return list(range(t[0], t[1] + 1))
                return []
            elif suffix == "incorrect":
                indices = []
                if self.correct_target_index is not None and self.targets is not None:
                    for i, (start, end) in enumerate(self.targets):
                        if i != self.correct_target_index:
                            indices.extend(range(start, end + 1))
                return indices
            else:
                idx = int(suffix)
                if self.targets is not None and idx < len(self.targets):
                    start, end = self.targets[idx]
                    return list(range(start, end + 1))
                return []
        elif token_type == "options":
            indices = []
            if self.option_ranges is not None:
                for start, end in self.option_ranges:
                    indices.extend(range(start, end + 1))
            return indices
        elif token_type == "option_correct":
            if self.answer_letter is not None and self.option_ranges is not None:
                ans_idx = ord(self.answer_letter.upper()) - ord("A")
                if 0 <= ans_idx < len(self.option_ranges):
                    start, end = self.option_ranges[ans_idx]
                    return list(range(start, end + 1))
            return []
        elif token_type == "option_incorrect":
            indices = []
            if self.answer_letter is not None and self.option_ranges is not None:
                ans_idx = ord(self.answer_letter.upper()) - ord("A")
                for i, (start, end) in enumerate(self.option_ranges):
                    if i != ans_idx:
                        indices.extend(range(start, end + 1))
            return indices
        elif token_type == "context":
            indices = []
            if self.contexts is not None:
                for start, end in self.contexts:
                    indices.extend(range(start, end + 1))
            return indices
        elif token_type.startswith("context_"):
            idx = int(token_type.split("_")[1])
            if self.contexts is not None and idx < len(self.contexts):
                start, end = self.contexts[idx]
                return list(range(start, end + 1))
            return []
        elif token_type == "last":
            return [self.last_token]
        else:
            raise ValueError(f"Unknown token type: {token_type}")

    def get_target_unit_of(self, token_idx: int) -> int:
        if self.targets is not None:
            for i, (start, end) in enumerate(self.targets):
                if start <= token_idx <= end:
                    return i
        return -1

    def get_context_unit_of(self, token_idx: int) -> int:
        if self.contexts is not None:
            for i, (start, end) in enumerate(self.contexts):
                if start <= token_idx <= end:
                    return i
        return -1

    def to_dict(self) -> dict:
        return {
            "system": list(self.system),
            "question": [list(r) for r in self.question],
            "query": list(self.query) if self.query else None,
            "targets": [list(t) for t in self.targets] if self.targets else None,
            "contexts": [list(c) for c in self.contexts] if self.contexts else None,
            "last_token": self.last_token,
            "seq_len": self.seq_len,
            "structure_kind": self.structure_kind,
            "structure_str": self.structure_str,
            "query_modality": self.query_modality,
            "target_modality": self.target_modality,
            "correct_target_index": self.correct_target_index,
            "answer_letter": self.answer_letter,
            "option_ranges": [list(r) for r in self.option_ranges] if self.option_ranges else None,
            "prompt_suffix": list(self.prompt_suffix) if self.prompt_suffix else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenPositionsInterleaved":
        return cls(
            system=tuple(d["system"]),
            question=[tuple(r) for r in d["question"]],
            query=tuple(d["query"]) if d.get("query") else None,
            targets=[tuple(t) for t in d["targets"]] if d.get("targets") else None,
            contexts=[tuple(c) for c in d["contexts"]] if d.get("contexts") else None,
            last_token=d["last_token"],
            seq_len=d["seq_len"],
            structure_kind=d["structure_kind"],
            structure_str=d["structure_str"],
            query_modality=d.get("query_modality"),
            target_modality=d.get("target_modality"),
            correct_target_index=d.get("correct_target_index"),
            answer_letter=d.get("answer_letter"),
            option_ranges=[tuple(r) for r in d["option_ranges"]] if d.get("option_ranges") else None,
            prompt_suffix=tuple(d["prompt_suffix"]) if d.get("prompt_suffix") else None,
        )


# ============================================================
# Token Position Extraction (Interleaved)
# ============================================================

def extract_token_positions_interleaved(
    input_ids,
    special_tokens,
    question_with_options,
    tokenizer,
    decomp,
    audit,
    answer_letter=None,
    options_list=None,
) -> TokenPositionsInterleaved:
    """Extract token positions for interleaved AV-Odyssey layout.

    Args:
        input_ids: list of token ids
        special_tokens: dict with vision_start, vision_end, audio_start, audio_end, etc.
        question_with_options: question string with special token strings
        tokenizer: tokenizer instance
        decomp: output of decompose_sample()
        audit: output of audit_sample_options()
        answer_letter: "A"/"B"/"C"/"D" correct answer
        options_list: list of option strings ["the first audio", ...]
    """
    n = len(input_ids)

    vision_start_id = special_tokens["vision_start"]
    vision_end_id = special_tokens["vision_end"]
    audio_start_id = special_tokens["audio_start"]
    audio_end_id = special_tokens["audio_end"]
    im_start_id = special_tokens["im_start"]

    # ================================================================
    # 1. Find system region
    # ================================================================
    im_start_positions = [i for i, tid in enumerate(input_ids) if tid == im_start_id]
    if len(im_start_positions) >= 2:
        user_message_start = im_start_positions[1]
    else:
        user_message_start = 0
    system_range = (0, user_message_start - 1) if user_message_start > 0 else (0, 0)

    # ================================================================
    # 2. Find all media regions in order of appearance
    # ================================================================
    media_regions = []

    i = 0
    while i < n:
        if input_ids[i] == vision_start_id:
            j = i + 1
            while j < n and input_ids[j] != vision_end_id:
                j += 1
            if j < n:
                media_regions.append({"start": i, "end": j, "type": "vision"})
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    i = 0
    while i < n:
        if input_ids[i] == audio_start_id:
            j = i + 1
            while j < n and input_ids[j] != audio_end_id:
                j += 1
            if j < n:
                media_regions.append({"start": i, "end": j, "type": "audio"})
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    media_regions.sort(key=lambda x: x["start"])
    media_regions = [m for m in media_regions if m["start"] > system_range[1]]

    # ================================================================
    # 3. Map each media region to its semantic role
    # ================================================================
    ordered_media = decomp.get("ordered_media", [])

    query_range = None
    target_ranges = []
    context_ranges = []

    if len(ordered_media) == len(media_regions):
        for media_info, region in zip(ordered_media, media_regions):
            role = media_info["role"]
            token_range = (region["start"], region["end"])
            if role == "query":
                query_range = token_range
            elif role.startswith("target_"):
                idx = int(role.split("_")[1])
                while len(target_ranges) <= idx:
                    target_ranges.append(None)
                target_ranges[idx] = token_range
            elif role.startswith("context_"):
                idx = int(role.split("_")[1])
                while len(context_ranges) <= idx:
                    context_ranges.append(None)
                context_ranges[idx] = token_range
    else:
        print(f"WARNING: extract_token_positions_interleaved: "
              f"ordered_media has {len(ordered_media)} items but found "
              f"{len(media_regions)} media regions in tokens.")

    target_ranges = [t for t in target_ranges if t is not None]
    context_ranges = [c for c in context_ranges if c is not None]

    # ================================================================
    # 4. Find question text (NOT options, NOT suffix)
    #
    #    a) Split question_with_options into question_part and options_part
    #    b) Split question_part at media token strings to get text chunks
    #    c) Search for each non-empty text chunk in token sequence
    #    d) Keep only the LONGEST one as the question text
    # ================================================================

    # Split at the options boundary (\nA:)
    options_split = re.search(r'\n[A-D]:', question_with_options)
    if options_split:
        question_part = question_with_options[:options_split.start()]
    else:
        question_part = question_with_options

    # Split question_part at media token strings
    media_token_strings = [
        "<|audio_bos|><|AUDIO|><|audio_eos|>",
        "<|vision_bos|><|IMAGE|><|vision_eos|>",
        "<|vision_bos|><|VIDEO|><|vision_eos|>",
    ]
    split_pattern = "|".join(re.escape(s) for s in media_token_strings)
    text_chunks = re.split(split_pattern, question_part)
    text_chunks = [chunk for chunk in text_chunks if chunk.strip()]

    # Search for each text chunk in the token sequence
    all_question_ranges = []
    for chunk in text_chunks:
        found = find_token_range(tokenizer, input_ids, chunk)
        if found is None:
            found = find_token_range(tokenizer, input_ids, chunk.strip())
        if found is not None:
            all_question_ranges.append(found)

    all_question_ranges.sort(key=lambda x: x[0])

    # # Pick the text chunk immediately before query in token space
    # # (same heuristic as decompose_sample)
    # question_ranges = []
    # if len(all_question_ranges) == 1:
    #     question_ranges = all_question_ranges
    # elif len(all_question_ranges) > 1:
    #     if query_range is not None:
    #         best = None
    #         best_dist = float('inf')
    #         for r in all_question_ranges:
    #             dist = query_range[0] - r[1]
    #             if 0 <= dist < best_dist:
    #                 best = r
    #                 best_dist = dist
    #         if best is not None:
    #             question_ranges = [best]
    #         else:
    #             best = max(all_question_ranges, key=lambda r: r[1] - r[0])
    #             question_ranges = [best]
    #     else:
    #         best = max(all_question_ranges, key=lambda r: r[1] - r[0])
    #         question_ranges = [best]
    

    # Use decompose's structure to pick the correct text chunk
    question_ranges = []
    if len(all_question_ranges) == 1:
        question_ranges = all_question_ranges
    elif len(all_question_ranges) > 1:
        struct = decomp["structure_str"]
        struct_parts = [p.strip() for p in struct.split("→")]

        if "question" in struct_parts:
            qn_pos = struct_parts.index("question")

            # Build lower and upper bounds from surrounding regions in structure
            lower_bound = 0
            upper_bound = float('inf')

            for pos, label in enumerate(struct_parts):
                if pos < qn_pos:
                    if label == "query" and query_range is not None:
                        lower_bound = max(lower_bound, query_range[1])
                    elif label == "targets" and target_ranges:
                        lower_bound = max(lower_bound, max(t[1] for t in target_ranges))
                elif pos > qn_pos:
                    if label == "query" and query_range is not None:
                        upper_bound = min(upper_bound, query_range[0])
                    elif label == "targets" and target_ranges:
                        upper_bound = min(upper_bound, target_ranges[0][0])

            candidates = [r for r in all_question_ranges
                          if r[0] >= lower_bound and r[1] <= upper_bound]

            if candidates:
                # Pick longest among candidates, ties go to later
                best = candidates[0]
                best_len = best[1] - best[0]
                for r in candidates[1:]:
                    r_len = r[1] - r[0]
                    if r_len >= best_len:
                        best = r
                        best_len = r_len
                question_ranges = [best]
            else:
                # Fallback: pick longest overall, ties go to later
                best = all_question_ranges[0]
                best_len = best[1] - best[0]
                for r in all_question_ranges[1:]:
                    r_len = r[1] - r[0]
                    if r_len >= best_len:
                        best = r
                        best_len = r_len
                question_ranges = [best]
        else:
            best = all_question_ranges[0]
            best_len = best[1] - best[0]
            for r in all_question_ranges[1:]:
                r_len = r[1] - r[0]
                if r_len >= best_len:
                    best = r
                    best_len = r_len
            question_ranges = [best]


    ###### ASSERTION format check question position matches decompose_sample structure
    if question_ranges and query_range is not None and target_ranges:
            # Rebuild structure from token positions
            parts = []
            regions = [
                ("question", question_ranges[0][0]),
                ("query", query_range[0]),
                ("targets", target_ranges[0][0]),
            ]
            regions.sort(key=lambda x: x[1])
            token_struct = " → ".join(r[0] for r in regions)

            # print("Assertion check: token_struct =", token_struct, "decomp['structure_str'] =", decomp["structure_str"])
            assert token_struct == decomp["structure_str"], (
                f"Structure mismatch: decompose says '{decomp['structure_str']}' "
                f"but token positions give '{token_struct}' "
                f"question={question_ranges[0]} query={query_range} targets={target_ranges[0]} "
                f"all={all_question_ranges}"
            )



    # ================================================================
    # 5. Find individual option ranges
    #    Search for "A: the first audio", "B: the third audio", etc.
    # ================================================================
    option_ranges_final = None
    if options_list is not None and len(options_list) > 0:
        option_ranges = []
        letters = "ABCDEFGHIJ"
        for i, opt_text in enumerate(options_list):
            if i < len(letters):
                opt_with_letter = f"{letters[i]}: {opt_text}"
                found = find_token_range(tokenizer, input_ids, opt_with_letter)
                if found is not None:
                    option_ranges.append(found)
                else:
                    found = find_token_range(tokenizer, input_ids, opt_text)
                    if found is not None:
                        option_ranges.append(found)
                    else:
                        option_ranges.append(None)

        if all(r is not None for r in option_ranges):
            option_ranges_final = option_ranges
        else:
            clean = [r for r in option_ranges if r is not None]
            option_ranges_final = clean if clean else None

    # ================================================================
    # 6. Find prompt suffix by searching for known suffix strings
    # ================================================================
    prompt_suffix_range = None
    suffix_candidates = [
        "Answer with the option's letter from the given choices directly.\nOnly give the best option.",
        "Only give the best option.",
        "Answer with the option's letter from the given choices directly.",
    ]
    for suffix_str in suffix_candidates:
        found = find_token_range(tokenizer, input_ids, suffix_str)
        if found is not None:
            prompt_suffix_range = (found[0], n - 2)
            break

    # ================================================================
    # 7. Build result
    # ================================================================
    return TokenPositionsInterleaved(
        system=system_range,
        question=question_ranges,
        query=query_range,
        targets=target_ranges if target_ranges else None,
        contexts=context_ranges if context_ranges else None,
        last_token=n - 1,
        seq_len=n,
        structure_kind=decomp["structure_kind"],
        structure_str=decomp["structure_str"],
        query_modality=decomp["query_modality"],
        target_modality=decomp["target_modality"],
        correct_target_index=audit.get("correct_target_index"),
        answer_letter=answer_letter,
        option_ranges=option_ranges_final,
        prompt_suffix=prompt_suffix_range,
    )


# ============================================================
# Utility: Result file naming
# ============================================================

def get_result_filename(config_id: int, sweep_type: str, layer_start: int, layer_end: int) -> str:
    config = KNOCKOUT_CONFIGS_INTERLEAVED[config_id]
    return f"config{config_id}_{config.name}_{sweep_type}_L{layer_start}-{layer_end}.jsonl"


# ============================================================
# Summary / Debug
# ============================================================

def print_config_summary():
    print("=" * 70)
    print("KNOCKOUT CONFIGURATIONS (INTERLEAVED)")
    print("=" * 70)
    for cid, cfg in KNOCKOUT_CONFIGS_INTERLEAVED.items():
        print(f"\nConfig {cid}: {cfg.name}")
        print(f"  Description: {cfg.description}")
        print(f"  Blocked pairs: {cfg.blocked_pairs}")
        if cfg.cross_unit:
            print(f"  Cross-unit: {cfg.cross_unit}")


def print_sweep_summary(model_size: str, sweep_type: str, k: int = 9):
    num_layers = get_num_layers(model_size)
    print(f"\n{'=' * 70}")
    print(f"SWEEP PLAN (INTERLEAVED): {model_size} model, {sweep_type}")
    print(f"{'=' * 70}")
    if sweep_type == "window":
        sweeps = get_all_window_sweeps(num_layers, k)
        total_runs = len(KNOCKOUT_CONFIGS_INTERLEAVED) * len(sweeps)
        print(f"  {len(KNOCKOUT_CONFIGS_INTERLEAVED)} configs x {len(sweeps)} positions = {total_runs} runs")
        print(f"  Window k={k}, stride=1")
        for s in sweeps[:3]:
            print(f"  Center {s['center']}: L{s['layer_start']}-L{s['layer_end']} ({s['num_masked']} layers)")
        print(f"  ...")
        for s in sweeps[-3:]:
            print(f"  Center {s['center']}: L{s['layer_start']}-L{s['layer_end']} ({s['num_masked']} layers)")
    else:
        raise ValueError(f"Unknown sweep_type: {sweep_type}. Use 'window'.")


if __name__ == "__main__":
    print_config_summary()
    print_sweep_summary("7B", "window", k=9)