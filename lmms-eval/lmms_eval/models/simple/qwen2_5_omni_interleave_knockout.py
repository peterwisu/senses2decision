"""
Qwen2.5-Omni Interleave Knockout Model Wrapper for AV-Odyssey

Standalone wrapper, structured identically to qwen2_5_omni_knockout.py.
The only differences from the AV-SpeakerBench version:
  - Message-building handles multiple image/audio paths per sample
  - <media_N> placeholder rewriting
  - Task dispatch for "av_odyssey"

Currently implements: BASELINE MODE only (knockout to be added later).

Modes (controlled by LMM_EVAL_SAVE_TYPE env var):
  "baseline"  — Prefill-only, save MCQ probabilities + token positions.

Env vars:
  LMM_EVAL_SAVE_TYPE    "baseline" (only mode supported for now)
  LMM_EVAL_OUTPUT_DIR   Directory to save results
  KNOCKOUT_MODEL_SIZE   "7B" or "3B"
  KNOCKOUT_GT_FIELD     Field name for ground truth answer (default: "answer")

Usage:
  LMM_EVAL_SAVE_TYPE=baseline \
  LMM_EVAL_OUTPUT_DIR=/path/to/output \
  KNOCKOUT_MODEL_SIZE=7B \
  python -m lmms_eval --model qwen2_5_omni_interleave_knockout --tasks av_odyssey ...
"""

import json
import os
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import yaml

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_omni_utils; install via `pip install qwen-omni-utils[decord]`"
    )


from utils.knockout.configs import (
    get_layer_groups,
    get_all_window_sweeps,
    get_window_layers,
    get_num_layers,
)
from utils.knockout.configs_interleaved import (
    KNOCKOUT_CONFIGS_INTERLEAVED,
    extract_token_positions_interleaved,
    decompose_sample,
    audit_sample_options,
    TokenPositionsInterleaved,
)

from utils.knockout.hooks import (
    extract_mcq_probs,
    compute_p_change,
)
from utils.knockout.hooks_interleaved import (
    KnockoutManagerInterleaved,
)
import re


# ============================================================
# Helpers
# ============================================================

def load_existing_knockout_ids(filepath: str) -> set:
    """Load already-saved doc_ids from a knockout center JSONL file."""
    saved_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        saved_ids.add(record["doc_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return saved_ids

def load_existing_results(filepath: str) -> dict:
    """Load already-saved results from a JSONL file for crash recovery.
    Returns dict: doc_id -> parsed_answer"""
    saved = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        saved[record["doc_id"]] = record.get("parsed_answer", "")
                    except (json.JSONDecodeError, KeyError):
                        continue
    return saved


def append_jsonl(filepath: str, record: dict):
    """Append one JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")



# ============================================================
# Model
# ============================================================

@register_model("qwen2_5_omni_interleave_knockout")
class Qwen2_5_Omni_Interleave_Knockout(lmms):
    """Qwen2.5-Omni knockout experiment wrapper for AV-Odyssey."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "eager",
        max_num_frames: int = 768,
        fps: Optional[float] = None,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        exp_config: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if attn_implementation != "eager":
            eval_logger.warning(
                f"attn_implementation={attn_implementation} but knockout REQUIRES eager. Forcing eager."
            )
            attn_implementation = "eager"

        self.fps = fps

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if exp_config and os.path.exists(exp_config):
            with open(exp_config) as f:
                self.exp_config = yaml.safe_load(f)
            eval_logger.info(f"Experiment config: {self.exp_config}")
        else:
            self.exp_config = None

        from models.qwen2_5.modelling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        print("### Using default method (eager attention) ###")

        Qwen2_5OmniForConditionalGeneration._tp_plan = (
            [] if Qwen2_5OmniForConditionalGeneration._tp_plan is None
            else Qwen2_5OmniForConditionalGeneration._tp_plan
        )
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype="auto",
            device_map=self.device_map,
            attn_implementation=attn_implementation,
        ).eval()

        if self.exp_config:
            self._model.thinker.exp_config = self.exp_config
        else:
            self._model.thinker.exp_config = None

        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._model.disable_talker()
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self.save_type = os.environ.get("LMM_EVAL_SAVE_TYPE", "baseline")
        self.output_dir = os.environ.get("LMM_EVAL_OUTPUT_DIR", "./knockout_results")
        self.model_size = os.environ.get("KNOCKOUT_MODEL_SIZE", "7B")
        self.gt_field = os.environ.get("KNOCKOUT_GT_FIELD", "answer")

        actual_layers = len(self.model.thinker.model.layers)
        expected_layers = get_num_layers(self.model_size)
        if actual_layers != expected_layers:
            eval_logger.warning(
                f"Model has {actual_layers} layers but KNOCKOUT_MODEL_SIZE={self.model_size} "
                f"expects {expected_layers}. Adjust KNOCKOUT_MODEL_SIZE."
            )

        print(f"\n{'='*60}")
        print(f"AV-ODYSSEY KNOCKOUT MODE: {self.save_type}")
        print(f"Output dir: {self.output_dir}")
        print(f"Model size: {self.model_size} ({actual_layers} layers)")
        print(f"Attention: {attn_implementation}")
        print(f"GT field: {self.gt_field}")
        print(f"{'='*60}\n")

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        if current_sample_rate != 16000:
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def _get_special_tokens(self) -> dict:
        tok = lambda s: self.tokenizer.convert_tokens_to_ids(s)
        return {
            "im_start": tok("<|im_start|>"),
            "im_end": tok("<|im_end|>"),
            "vision_start": tok("<|vision_bos|>"),
            "vision_end": tok("<|vision_eos|>"),
            "audio_start": tok("<|audio_bos|>"),
            "audio_end": tok("<|audio_eos|>"),
            "video": tok("<|VIDEO|>"),
            "audio": tok("<|AUDIO|>"),
            "image": tok("<|IMAGE|>"),
        }

    def _prepare_inputs(self, contexts, visuals):
        """Build message, process media, return (inputs, current_use_audio).

        AV-Odyssey-specific: handles multiple image/audio file paths per sample.
        Interleaves media content at <media_N> placeholder positions in the text,
        matching the original qwen2_5_omni_interleaved.py behavior.
        """
        AUDIO_EXTS = (".wav",)
        VIDEO_EXTS = (".mp4",)
        IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

        current_use_audio = False
        message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

        for i, context in enumerate(contexts):
            doc_visuals = visuals[i] if i < len(visuals) else []
            if not isinstance(doc_visuals, (list, tuple)):
                doc_visuals = [doc_visuals]

            # Pre-process each media item into content blocks
            # (audio gets split into chunks, so one media item -> multiple content blocks)
            media_blocks_per_item = []
            for v in doc_visuals:
                v_lower = v.lower()
                if v_lower.endswith(VIDEO_EXTS):
                    if self._check_if_video_has_audio(v):
                        current_use_audio = True
                    media_blocks_per_item.append([{"type": "video", "video": v}])
                elif v_lower.endswith(AUDIO_EXTS):
                    samples, sr = librosa.load(v, sr=None, mono=True)
                    samples = samples.astype(np.float32)
                    audio = self.resample_audio(samples, sr)
                    chunks = [{"type": "audio", "audio": c} for c in split_audio(audio, 4800000)]
                    media_blocks_per_item.append(chunks)
                elif v_lower.endswith(IMAGE_EXTS):
                    img = Image.open(v).convert("RGB")
                    media_blocks_per_item.append([{"type": "image", "image": img}])
                else:
                    raise ValueError(f"Unknown file extension: {v}")

            # Split the question text at <media_N> markers and interleave
            parts = re.split(r"<media_(\d+)>", context)
            # re.split with a capturing group gives: [text, idx, text, idx, text, ...]

            content = []
            for j, part in enumerate(parts):
                if j % 2 == 0:
                    # Plain text segment
                    if part:
                        content.append({"type": "text", "text": part})
                else:
                    # Media index
                    media_idx = int(part)
                    if media_idx < len(media_blocks_per_item):
                        content.extend(media_blocks_per_item[media_idx])
                    else:
                        raise ValueError(
                            f"Question references <media_{media_idx}> "
                            f"but only {len(media_blocks_per_item)} media items available"
                        )

            single_message = {"role": "user", "content": content}
            message.append(single_message)

        text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        text = text + "Answer:("

        audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)


        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=current_use_audio,
        )

        if self.device_map == "auto":
            inputs = inputs.to("cuda").to(self.model.dtype)
        else:
            inputs = inputs.to(self.model.device).to(self.model.dtype)

        return inputs, current_use_audio

    # ============================================================
    # Prefill-only forward pass (shared by baseline and knockout)
    # ============================================================
    # def _prefill_forward(self, inputs):
    #     """Run prefill-only forward pass, return logits at last position."""
    #     with torch.no_grad():
    #         outputs = self.model.thinker(
    #             **inputs,
    #             use_cache=False,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #             return_dict=True,
    #         )
    #     return outputs.logits[0, -1, :]

    def _generate_forward(self, inputs, max_new_tokens=32):
        with torch.no_grad():
            outputs = self.model.thinker.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=self.use_cache,
                return_dict_in_generate=True,
                output_logits=True,
            )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text, generated_ids, outputs.logits

    # ============================================================
    # BASELINE MODE
    # ============================================================
    def _run_baseline(self, requests: List[Instance]) -> List[str]:
        """Run baseline: prefill-only, save MCQ probabilities + token positions.
        Per-sample crash recovery via doc_id check."""
        res = []
        baseline_file = os.path.join(self.output_dir, "baseline", "baseline.jsonl")
        os.makedirs(os.path.dirname(baseline_file), exist_ok=True)

        saved_results = load_existing_results(baseline_file)
        if saved_results:
            eval_logger.info(f"Crash recovery: found {len(saved_results)} already-saved samples")

        # print(baseline_file)
        # exit()
        special_tokens = self._get_special_tokens()

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Baseline")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            contexts = list(contexts)
            contexts[0] = contexts[0] + "\nOnly give the best option."
            task = task[0]
            split = split[0]

            # Skip if already saved
            if doc_id[0] in saved_results:
                saved_answer = saved_results[doc_id[0]]
                for context in contexts:
                    res.append(saved_answer if saved_answer else "")
                    self.cache_hook.add_partial(
                        "generate_until",
                        (context, all_gen_kwargs[0]),
                        saved_answer if saved_answer else "",
                    )
                    pbar.update(1)
                continue

            # Build visuals (NOTE: do NOT flatten — keep per-doc grouping for AV-Odyssey)
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            print("________")
            doc = self.task_dict[task][split][doc_id[0]]
            print(doc.get("question_id"))
            print("________")
            try:
                inputs, current_use_audio = self._prepare_inputs(contexts, visuals)
            except Exception as e:
                eval_logger.error(f"Error preparing inputs for doc_id={doc_id[0]}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            try:
                # logits = self._prefill_forward(inputs)
                raw_response, generated_ids, logits_per_step = self._generate_forward(inputs, max_new_tokens=32)
            except Exception as e:
                eval_logger.error(f"Error in forward pass for doc_id={doc_id[0]}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            # mcq = extract_mcq_probs(logits, self.tokenizer)
            letter_logits = logits_per_step[0][0]


            ## DEBUG
            top_probs, top_ids = torch.topk(torch.softmax(letter_logits, dim=0), 5)
            print(f"[doc_id={doc_id[0]}] step 0 top-5:")
            for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                print(f"  p={p:.4f} tok={self._tokenizer.decode([tid])!r}")


            mcq = extract_mcq_probs(letter_logits, self.tokenizer)
            parsed_answer = mcq["predicted"]

            input_ids_list = inputs["input_ids"][0].cpu().tolist()
            doc = self.task_dict[task][split][doc_id[0]]

            if task == "av_odyssey":
                options = doc["options"]
                
                # question_with_options = doc["question"] + "\n" + "\n".join(options)
                options_with_letters = [f"{chr(ord('A') + i)}: {opt}" for i, opt in enumerate(options)]
                question_with_options = doc["question"] + "\n" + "\n".join(options_with_letters)
                

                # Replace [audioN]/[imgN]/[videoN] -> <media_N> (same as lmms-eval)
                media_idx = [0]
                def _replace_placeholder(m):
                    idx = media_idx[0]
                    media_idx[0] += 1
                    return f"<media_{idx}>"
                question_with_options = re.sub(
                    r"\[(audio|video|img)\d+\]", _replace_placeholder, question_with_options
                )

                # Replace <media_N> -> actual special token strings (same as tokenized text)
                visuals_for_doc = doc_to_visual[0](self.task_dict[task][split][doc_id[0]])
                if not isinstance(visuals_for_doc, (list, tuple)):
                    visuals_for_doc = [visuals_for_doc]

                for idx, v in enumerate(visuals_for_doc):
                    v_lower = v.lower()
                    if v_lower.endswith(".wav"):
                        replacement = "<|audio_bos|><|AUDIO|><|audio_eos|>"
                    elif v_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                        replacement = "<|vision_bos|><|IMAGE|><|vision_eos|>"
                    elif v_lower.endswith(".mp4"):
                        replacement = "<|vision_bos|><|VIDEO|><|vision_eos|>"
                    else:
                        replacement = ""
                    question_with_options = question_with_options.replace(f"<media_{idx}>", replacement)


                # print("Question with options:", question_with_options)
            else:
                raise ValueError(
                    f"Unknown dataset: {task}. This wrapper only supports 'av_odyssey'. "
                    f"Use qwen2_5_omni_knockout for AV-SpeakerBench / WorldSense."
                )

            # Compute structure + correct target index
            decomp = decompose_sample(doc["question"])
            audit = audit_sample_options(doc)

            # Extract token positions (interleaved layout)
            token_positions = extract_token_positions_interleaved(
                input_ids_list,
                special_tokens,
                question_with_options=question_with_options,
                tokenizer=self.tokenizer,
                decomp=decomp,
                audit=audit,
                answer_letter=doc.get("answer"),
                options_list=doc.get("options"),
            )


            # DEBUG
            # print(f"\n{'='*60}")
            # print(f"doc_id: {doc_id[0]}")
            # print(f"structure: {token_positions.structure_str}")
            # print(f"structure_kind: {token_positions.structure_kind}")
            # print(f"query_modality: {token_positions.query_modality}")
            # print(f"target_modality: {token_positions.target_modality}")
            # print(f"correct_target_index: {token_positions.correct_target_index}")
            # print(f"answer_letter: {token_positions.answer_letter}")
            # print(f"seq_len: {token_positions.seq_len}")
            # print(f"system: {token_positions.system}")
            # print(f"query: {token_positions.query}")
            # print(f"targets: {token_positions.targets}")
            # print(f"contexts: {token_positions.contexts}")
            # print(f"question ranges: {token_positions.question}")
            # print(f"option_ranges: {token_positions.option_ranges}")
            # print(f"prompt_suffix: {token_positions.prompt_suffix}")
            # print(f"last_token: {token_positions.last_token}")
            # # --- SYSTEM ---
            # print(f"\n--- SYSTEM [{token_positions.system[0]}:{token_positions.system[1]}] ---")
            # print(self.tokenizer.decode(input_ids_list[token_positions.system[0]:token_positions.system[1]+1]))
            # # --- QUESTION ---
            # print(f"\n--- QUESTION ---")
            # for i, (s, e) in enumerate(token_positions.question):
            #     print(f"question[{i}] [{s}:{e}]:")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- QUERY ---
            # if token_positions.query:
            #     s, e = token_positions.query
            #     print(f"\n--- QUERY [{s}:{e}] ---")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- TARGETS (with TRUE/FALSE) ---
            # if token_positions.targets:
            #     print(f"\n--- TARGETS ---")
            #     for i, (s, e) in enumerate(token_positions.targets):
            #         is_correct = "TRUE" if i == token_positions.correct_target_index else "FALSE"
            #         print(f"target_{i} [{is_correct}] [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- CONTEXTS ---
            # if token_positions.contexts:
            #     print(f"\n--- CONTEXTS ---")
            #     for i, (s, e) in enumerate(token_positions.contexts):
            #         print(f"context_{i} [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- OPTIONS (with TRUE/FALSE) ---
            # if token_positions.option_ranges:
            #     print(f"\n--- OPTIONS ---")
            #     letters = "ABCDEFGHIJ"
            #     for i, (s, e) in enumerate(token_positions.option_ranges):
            #         letter = letters[i] if i < len(letters) else "?"
            #         is_correct = "TRUE" if letter == token_positions.answer_letter else "FALSE"
            #         print(f"option_{letter} [{is_correct}] [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # else:
            #     print(f"\n--- OPTIONS: None (task 26 / unparseable) ---")
            # # --- PROMPT SUFFIX ---
            # if token_positions.prompt_suffix:
            #     s, e = token_positions.prompt_suffix
            #     print(f"\n--- PROMPT SUFFIX [{s}:{e}] ---")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # else:
            #     print(f"\n--- PROMPT SUFFIX: None ---")
            # # --- LAST TOKEN ---
            # print(f"\n--- LAST TOKEN [{token_positions.last_token}] ---")
            # print(self.tokenizer.decode(input_ids_list[token_positions.last_token:token_positions.last_token+1]))
            # # --- FULL TOKEN SEQUENCE ---
            # print(f"\n--- FULL TOKEN SEQUENCE ---")
            # print(self.tokenizer.decode(input_ids_list))



            # print("GT answer:",  doc[self.gt_field])
            # print(f"{'='*60}\n")


            # exit()
            gt = doc[self.gt_field]
            is_correct = (parsed_answer == gt)
            
            
            if task == "av_odyssey":
                record = {
                    # ---- IDs ----
                    "doc_id": doc_id[0],
                    "task": task,
                    "split": split,
                    # ---- Dataset fields ----
                    "question_id": doc.get("question_id", ""),
                    "question_type_id": doc.get("question_type_id", -1),
                    "data_type": doc.get("data_type", ""),
                    "subfield": doc.get("subfield", ""),
                    "question": doc["question"],
                    "options": doc["options"],
                    # ---- Paths ----
                    "image_path": doc.get("image_path", None),
                    "video_path": doc.get("video_path", None),
                    "audio_path": doc.get("audio_path", None),
                    # ---- Results ----
                    "parsed_answer": parsed_answer,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "probs": mcq["probs"],
                    "logits_raw": mcq["logits_raw"],
                    # ---- Structure ----
                    "structure": token_positions.structure_str,
                    "structure_kind": token_positions.structure_kind,
                    "query_modality": token_positions.query_modality,
                    "target_modality": token_positions.target_modality,
                    "correct_target_index": token_positions.correct_target_index,
                    "answer_letter": token_positions.answer_letter,
                    # ---- Token positions ----
                    "token_positions": token_positions.to_dict(),
                    "seq_len": token_positions.seq_len,
                }
            else:
                raise ValueError(f"Unknown dataset: {task} please fix this to use other datasets")

            append_jsonl(baseline_file, record)

            status = "CORRECT" if is_correct else "WRONG"
            eval_logger.info(
                f"[{status}] doc_id={doc_id[0]} task_id={doc.get('question_type_id')} "
                f"pred={parsed_answer} gt={gt} "
                f"P(A)={mcq['probs'].get('A', 0):.3f} P(B)={mcq['probs'].get('B', 0):.3f} "
                f"P(C)={mcq['probs'].get('C', 0):.3f} P(D)={mcq['probs'].get('D', 0):.3f} "
                f"seq_len={token_positions.seq_len}"
            )

            gen_kwargs = all_gen_kwargs[0]
            for i, context in enumerate(contexts):
                res.append(parsed_answer if parsed_answer else "")
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), parsed_answer if parsed_answer else ""
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()

        saved_ids = load_existing_results(baseline_file)
        correct = sum(1 for _ in open(baseline_file) if '"is_correct": true' in _)
        eval_logger.info(f"\nBaseline complete: {len(saved_ids)} samples saved to {baseline_file}")
        eval_logger.info(f"Correct samples: {correct}/{len(saved_ids)}")
        return res


# ============================================================
    # KNOCKOUT MODE
    # ============================================================
    def _run_knockout(self, requests: List[Instance]) -> List[str]:
        """Run knockout: strictly 1 config x 1 center per invocation.

        Per-sample crash recovery: checks doc_id in center file before processing.

        Env vars:
          KNOCKOUT_CONFIG_IDS    single config id e.g. "1"
          KNOCKOUT_SWEEP_TYPE    "window" (default)
          KNOCKOUT_WINDOW_K      Window size (default: 9)
          KNOCKOUT_SWEEP_RANGE   single center e.g. "10-10"
          KNOCKOUT_SAMPLE_MODE   "correct" (default) or "all"
        """
        # ---- Read settings ----
        config_id = int(os.environ.get("KNOCKOUT_CONFIG_IDS", "1"))
        sweep_type = os.environ.get("KNOCKOUT_SWEEP_TYPE", "window")
        window_k = int(os.environ.get("KNOCKOUT_WINDOW_K", "9"))
        sample_mode = os.environ.get("KNOCKOUT_SAMPLE_MODE", "correct")
        sweep_range_str = os.environ.get("KNOCKOUT_SWEEP_RANGE", "0-0")

        config = KNOCKOUT_CONFIGS_INTERLEAVED[config_id]
        num_layers = get_num_layers(self.model_size)

        # Parse single center
        range_start, range_end = map(int, sweep_range_str.split("-"))
        assert range_start == range_end, f"Single center expected, got {sweep_range_str}"
        center = range_start
        target_layers = get_window_layers(center, window_k, num_layers)

        # ---- Load baseline ----
        baseline_file = os.path.join(self.output_dir, "baseline", "baseline.jsonl")
        assert os.path.exists(baseline_file), f"Baseline not found: {baseline_file}"

        print(f"Loading baseline results from {baseline_file}...")
        baseline_data = {}
        with open(baseline_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    baseline_data[record["doc_id"]] = record

        correct_ids = {did for did, rec in baseline_data.items() if rec["is_correct"]}

        # ---- Output file ----
        knockout_dir = os.path.join(self.output_dir, f"{sweep_type}_k{window_k}")
        config_dir = os.path.join(knockout_dir, f"config{config_id}_{config.name}")
        os.makedirs(config_dir, exist_ok=True)
        center_file = os.path.join(config_dir, f"center_{center:02d}.jsonl")

        # ---- Crash recovery ----
        saved_ids = load_existing_knockout_ids(center_file)

        print(f"\n{'='*60}")
        print(f"AV-ODYSSEY KNOCKOUT EXPERIMENT")
        print(f"  Config:          {config_id} ({config.name})")
        print(f"  Center:          {center}")
        print(f"  Target layers:   {target_layers}")
        print(f"  Window k:        {window_k}")
        print(f"  Sample mode:     {sample_mode}")
        print(f"  Baseline:        {len(baseline_data)} samples ({len(correct_ids)} correct)")
        print(f"  Already saved:   {len(saved_ids)} records")
        print(f"  Remaining:       ~{len(baseline_data) - len(saved_ids)}")
        print(f"  Output:          {center_file}")
        print(f"{'='*60}\n")

        # ---- Main loop ----
        res = []
        special_tokens = self._get_special_tokens()

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=f"KO cfg{config_id} c{center}")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        knockout_count = 0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            did = doc_id[0]

            base_record = baseline_data.get(did, {})
            baseline_answer = base_record.get("parsed_answer", "")
            is_correct_sample = did in correct_ids

            # verify question_id matches baseline
            current_question_id = self.task_dict[task][split][did].get("question_id")
            baseline_question_id = base_record.get("question_id")
            if current_question_id != baseline_question_id:
                raise ValueError(
                    f"question_id mismatch for doc_id={did}: "
                    f"current={current_question_id} baseline={baseline_question_id}"
                )

            # print(current_question_id) 
            # print(baseline_question_id)
            # print(current_question_id == baseline_question_id)
        

            # ---- Already saved: skip ----
            if did in saved_ids:
                current_question_id = self.task_dict[task][split][did].get("question_id")
                baseline_question_id = base_record.get("question_id")
                if current_question_id != baseline_question_id:
                    raise ValueError(
                        f"question_id mismatch for doc_id={did}: "
                        f"current={current_question_id} baseline={baseline_question_id}"
                    )
                for context in contexts:
                    res.append(baseline_answer if baseline_answer else "")
                    self.cache_hook.add_partial(
                        "generate_until", (context, all_gen_kwargs[0]),
                        baseline_answer if baseline_answer else "",
                    )
                    pbar.update(1)
                continue

            # ---- Wrong baseline: write skip record ----
            if sample_mode == "correct" and not is_correct_sample:
                skip_record = {
                    "doc_id": did,
                    "config_id": config_id,
                    "config_name": config.name,
                    "center": center,
                    "target_layers": target_layers,
                    "ground_truth": base_record.get("ground_truth", ""),
                    "base_predicted": baseline_answer,
                    "knockout_predicted": None,
                    "skipped": True,
                    "skip_reason": "baseline_wrong",
                    "base_probs": base_record.get("probs", {}),
                    "knockout_probs": {},
                    "relative_diff": None,
                    "relative_diff_gt": None,
                    # AV-Odyssey fields
                    "question_type_id": base_record.get("question_type_id", -1),
                    "data_type": base_record.get("data_type", ""),
                    "subfield": base_record.get("subfield", ""),
                }
                append_jsonl(center_file, skip_record)
                saved_ids.add(did)

                for context in contexts:
                    res.append(baseline_answer if baseline_answer else "")
                    self.cache_hook.add_partial(
                        "generate_until", (context, all_gen_kwargs[0]),
                        baseline_answer if baseline_answer else "",
                    )
                    pbar.update(1)
                continue

            # ---- Prepare inputs ----
            contexts = list(contexts)
            contexts[0] = contexts[0] + "\nOnly give the best option."

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            try:
                inputs, current_use_audio = self._prepare_inputs(contexts, visuals)
            except Exception as e:
                eval_logger.error(f"Error preparing inputs for doc_id={did}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            # ---- Extract token positions ----
            input_ids_list = inputs["input_ids"][0].cpu().tolist()
            doc = self.task_dict[task][split][did]

            options = doc["options"]
            options_with_letters = [f"{chr(ord('A') + i)}: {opt}" for i, opt in enumerate(options)]
            question_with_options = doc["question"] + "\n" + "\n".join(options_with_letters)

            # Replace [audioN]/[imgN]/[videoN] -> <media_N>
            media_idx = [0]
            def _replace_placeholder(m):
                idx = media_idx[0]
                media_idx[0] += 1
                return f"<media_{idx}>"
            question_with_options = re.sub(
                r"\[(audio|video|img)\d+\]", _replace_placeholder, question_with_options
            )

            # Replace <media_N> -> special token strings
            visuals_for_doc = doc_to_visual[0](self.task_dict[task][split][did])
            if not isinstance(visuals_for_doc, (list, tuple)):
                visuals_for_doc = [visuals_for_doc]
            for idx, v in enumerate(visuals_for_doc):
                v_lower = v.lower()
                if v_lower.endswith(".wav"):
                    replacement = "<|audio_bos|><|AUDIO|><|audio_eos|>"
                elif v_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                    replacement = "<|vision_bos|><|IMAGE|><|vision_eos|>"
                elif v_lower.endswith(".mp4"):
                    replacement = "<|vision_bos|><|VIDEO|><|vision_eos|>"
                else:
                    replacement = ""
                question_with_options = question_with_options.replace(f"<media_{idx}>", replacement)

            decomp = decompose_sample(doc["question"])

            # # DEBUG: skip until target structure (comment out for real runs)
            # if decomp["structure_str"] != "targets → question → query": continue

            # print(decomp["structure_str"])
            # exit()

            audit = audit_sample_options(doc)

            token_positions = extract_token_positions_interleaved(
                input_ids_list,
                special_tokens,
                question_with_options=question_with_options,
                tokenizer=self.tokenizer,
                decomp=decomp,
                audit=audit,
                answer_letter=doc.get("answer"),
                options_list=doc.get("options"),
            )

            for layer in self.model.thinker.model.layers:
                layer.self_attn._token_positions = token_positions


            ## DEBUG
            # print(f"\n{'='*60}")
            # print(f"doc_id: {doc_id[0]}")
            # print(f"structure: {token_positions.structure_str}")
            # print(f"structure_kind: {token_positions.structure_kind}")
            # print(f"query_modality: {token_positions.query_modality}")
            # print(f"target_modality: {token_positions.target_modality}")
            # print(f"correct_target_index: {token_positions.correct_target_index}")
            # print(f"answer_letter: {token_positions.answer_letter}")
            # print(f"seq_len: {token_positions.seq_len}")
            # print(f"system: {token_positions.system}")
            # print(f"query: {token_positions.query}")
            # print(f"targets: {token_positions.targets}")
            # print(f"contexts: {token_positions.contexts}")
            # print(f"question ranges: {token_positions.question}")
            # print(f"option_ranges: {token_positions.option_ranges}")
            # print(f"prompt_suffix: {token_positions.prompt_suffix}")
            # print(f"last_token: {token_positions.last_token}")
            # # --- SYSTEM ---
            # print(f"\n--- SYSTEM [{token_positions.system[0]}:{token_positions.system[1]}] ---")
            # print(self.tokenizer.decode(input_ids_list[token_positions.system[0]:token_positions.system[1]+1]))
            # # --- QUESTION ---
            # print(f"\n--- QUESTION ---")
            # for i, (s, e) in enumerate(token_positions.question):
            #     print(f"question[{i}] [{s}:{e}]:")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- QUERY ---
            # if token_positions.query:
            #     s, e = token_positions.query
            #     print(f"\n--- QUERY [{s}:{e}] ---")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- TARGETS (with TRUE/FALSE) ---
            # if token_positions.targets:
            #     print(f"\n--- TARGETS ---")
            #     for i, (s, e) in enumerate(token_positions.targets):
            #         is_correct = "TRUE" if i == token_positions.correct_target_index else "FALSE"
            #         print(f"target_{i} [{is_correct}] [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- CONTEXTS ---
            # if token_positions.contexts:
            #     print(f"\n--- CONTEXTS ---")
            #     for i, (s, e) in enumerate(token_positions.contexts):
            #         print(f"context_{i} [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # # --- OPTIONS (with TRUE/FALSE) ---
            # if token_positions.option_ranges:
            #     print(f"\n--- OPTIONS ---")
            #     letters = "ABCDEFGHIJ"
            #     for i, (s, e) in enumerate(token_positions.option_ranges):
            #         letter = letters[i] if i < len(letters) else "?"
            #         is_correct = "TRUE" if letter == token_positions.answer_letter else "FALSE"
            #         print(f"option_{letter} [{is_correct}] [{s}:{e}]:")
            #         print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # else:
            #     print(f"\n--- OPTIONS: None (task 26 / unparseable) ---")
            # # --- PROMPT SUFFIX ---
            # if token_positions.prompt_suffix:
            #     s, e = token_positions.prompt_suffix
            #     print(f"\n--- PROMPT SUFFIX [{s}:{e}] ---")
            #     print(self.tokenizer.decode(input_ids_list[s:e+1]))
            # else:
            #     print(f"\n--- PROMPT SUFFIX: None ---")
            # # --- LAST TOKEN ---
            # print(f"\n--- LAST TOKEN [{token_positions.last_token}] ---")
            # print(self.tokenizer.decode(input_ids_list[token_positions.last_token:token_positions.last_token+1]))
            # # --- FULL TOKEN SEQUENCE ---
            # print(f"\n--- FULL TOKEN SEQUENCE ---")
            # print(self.tokenizer.decode(input_ids_list))



            # print("GT answer:",  doc[self.gt_field])
            # print(f"{'='*60}\n")
            # exit()
            gt = base_record["ground_truth"]
            base_predicted = base_record["parsed_answer"]
            base_probs = base_record["probs"]

            # ---- Run knockout forward pass ----
            try:
                with KnockoutManagerInterleaved(
                    self.model, token_positions, config_id, target_layers
                ) as manager:
                    # logits = self._prefill_forward(inputs)
                    raw_response_ko, generated_ids_ko, logits_per_step_ko = self._generate_forward(inputs, max_new_tokens=32)
                ko_letter_logits = logits_per_step_ko[0][0]

                ## DEBUG
                top_probs, top_ids = torch.topk(torch.softmax(ko_letter_logits, dim=0), 5)
                print(f"[doc_id={doc_id[0]}] step 0 top-5:")
                for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                    print(f"  p={p:.4f} tok={self._tokenizer.decode([tid])!r}")
                knockout_mcq = extract_mcq_probs(ko_letter_logits, self.tokenizer)
                # knockout_mcq = extract_mcq_probs(logits, self.tokenizer)
            except Exception as e:
                eval_logger.error(f"Error knockout cfg={config_id} c={center} doc={did}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            p_change = compute_p_change(
                {"probs": base_probs},
                {"probs": knockout_mcq["probs"]},
                gt, base_predicted,
            )

            # ---- Save result ----
            result = {
                "doc_id": did,
                "config_id": config_id,
                "config_name": config.name,
                "center": center,
                "target_layers": target_layers,
                "ground_truth": gt,
                "base_predicted": base_predicted,
                "knockout_predicted": knockout_mcq["predicted"],
                "skipped": False,
                "base_probs": base_probs,
                "knockout_probs": knockout_mcq["probs"],
                **p_change,
                # AV-Odyssey fields
                "question_type_id": base_record.get("question_type_id", -1),
                "data_type": base_record.get("data_type", ""),
                "subfield": base_record.get("subfield", ""),
                "structure": base_record.get("structure", ""),
                "structure_kind": base_record.get("structure_kind", ""),
                "query_modality": base_record.get("query_modality", ""),
                "target_modality": base_record.get("target_modality", ""),
                "correct_target_index": base_record.get("correct_target_index", None),
            }
            append_jsonl(center_file, result)
            saved_ids.add(did)
            knockout_count += 1

            eval_logger.info(
                f"[{knockout_count}] doc={did} "
                f"task_id={base_record.get('question_type_id')} "
                f"cfg={config_id} c={center} "
                f"{base_predicted}>{knockout_mcq['predicted']} gt={gt} "
                f"diff={p_change['relative_diff']:.1f}%"
            )

            for i, context in enumerate(contexts):
                res.append(baseline_answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, all_gen_kwargs[0]), baseline_answer,
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()

        total_lines = len(load_existing_knockout_ids(center_file))
        eval_logger.info(
            f"\nDone: config={config_id} center={center} | "
            f"{knockout_count} new + {len(saved_ids) - knockout_count} skipped/recovered = "
            f"{total_lines} total lines in {center_file}"
        )
        return res
    

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    def generate_until(self, requests: List[Instance]) -> List[str]:
        if self.save_type == "baseline":
            return self._run_baseline(requests)
        elif self.save_type == "knockout":
            return self._run_knockout(requests)
        else:
            raise ValueError(
                f"Unknown LMM_EVAL_SAVE_TYPE='{self.save_type}'. Use 'baseline' for now."
            )

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported for knockout")