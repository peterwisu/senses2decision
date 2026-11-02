"""
video-SALMONN-2+ Knockout Model Wrapper for lmms-eval

Dedicated model wrapper for knockout experiments on video-SALMONN-2+.
Keeps the original video_salmonn_2_full.py untouched for regular benchmarking.

Modes (controlled by LMM_EVAL_SAVE_TYPE env var):
  "baseline"  — Prefill-only, save MCQ probabilities + token positions.
                Run ONCE. Produces baseline.jsonl with all samples.
  "knockout"  — Prefill with knockout hooks.
                Reads baseline.jsonl to filter correct-only samples.

Env vars:
  LMM_EVAL_SAVE_TYPE    "baseline" or "knockout"
  LMM_EVAL_OUTPUT_DIR   Directory to save results
  KNOCKOUT_MODEL_SIZE   "7B" or "3B"
  KNOCKOUT_CONFIG_IDS    single config id e.g. "1"
  KNOCKOUT_SWEEP_TYPE    "window" (default)
  KNOCKOUT_WINDOW_K      Window size (default: 9)
  KNOCKOUT_SAMPLE_MODE   "correct" (default) or "all"
  KNOCKOUT_SWEEP_RANGE   single center e.g. "10-10"
"""

import ast
import json
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.tasks.av_speakerbench.utils import _parse_choices, parse_multi_choice_response
# Add video-SALMONN-2 repo to sys.path
_repo_path = os.environ.get(
    "VIDEO_SALMONN_REPO",
    "/path/to/sense2decision/models/video-SALMONN-2/video_SALMONN2_plus",
)
if _repo_path and _repo_path not in sys.path:
    sys.path.insert(0, _repo_path)

from transformers import AutoTokenizer, WhisperFeatureExtractor

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import DataArguments

from utils.knockout.configs import (
    KNOCKOUT_CONFIGS,
    extract_token_positions_vs2,
    get_layer_groups,
    get_all_window_sweeps,
    get_window_layers,
    get_num_layers,
)
from utils.knockout.hooks import (
    KnockoutManager,
    extract_mcq_probs,
    compute_p_change,
)


# ============================================================
# Helpers
# ============================================================

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


def append_jsonl(filepath: str, record: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================
# Model
# ============================================================

@register_model("video_salmonn_2_full_knockout")
class VideoSALMONN2Knockout(lmms):
    """video-SALMONN-2+ knockout experiment wrapper."""

    def __init__(
        self,
        pretrained: str = "tsinghua-ee/video-SALMONN2_plus_3B_full",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "eager",
        max_num_frames: int = 128,
        max_pixels: int = 25088,
        min_pixels: int = 3136,
        base_interval: float = 0.5,
        model_max_length: int = 131072,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if attn_implementation != "eager":
            eval_logger.warning(f"attn_implementation={attn_implementation} but knockout REQUIRES eager. Forcing eager.")
            attn_implementation = "eager"

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

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "attn_implementation": attn_implementation,
        }
        print(f"Loading video-SALMONN-2+ from {pretrained}")
        self._model = video_SALMONN2_plus.from_pretrained(pretrained, **model_kwargs).eval()
        self.attn_implementation = attn_implementation

        # Data args / test_data for input preparation
        self.data_args = DataArguments()
        self.data_args.dataset_use = ""
        self.data_args.video_max_frames = max_num_frames
        self.data_args.video_min_frames = 16
        self.data_args.base_interval = base_interval
        self.data_args.max_pixels = max_pixels
        self.data_args.min_pixels = min_pixels
        self.data_args.video_max_frame_pixels = max_pixels
        self.data_args.video_min_frame_pixels = min_pixels
        self.data_args.run_test = True
        self.data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(pretrained)
        self.data_args.audio_processor = WhisperFeatureExtractor(
            feature_size=self.data_args.feature_size,
            sampling_rate=self.data_args.sampling_rate,
            hop_length=self.data_args.hop_length,
            chunk_length=self.data_args.chunk_length,
        )
        self.data_args.model_type = "qwen2.5vl"

        data_module = make_supervised_data_module(tokenizer=self._tokenizer, data_args=self.data_args)
        self.test_data = data_module["train_dataset"]

        # Special tokens
        cfg = self._model.config

        def _get_id(name, fallback_str):
            val = getattr(cfg, name, None)
            if val is not None:
                return val
            tid = self._tokenizer.convert_tokens_to_ids(fallback_str)
            return tid if tid != self._tokenizer.unk_token_id else None

        self.special_token_ids = {
            "im_start":     _get_id("im_start_token_id",     "<|im_start|>"),
            "im_end":       _get_id("im_end_token_id",       "<|im_end|>"),
            "vision_start": _get_id("vision_start_token_id", "<|vision_start|>"),
            "vision_end":   _get_id("vision_end_token_id",   "<|vision_end|>"),
            "video":        _get_id("video_token_id",        "<|video_pad|>"),
            "image":        _get_id("image_token_id",        "<|image_pad|>"),
            "audio":        _get_id("audio_token_id",        "<|audio_pad|>"),
            "audio_start":  None,   # video-SALMONN does not use these
            "audio_end":    None,
        }
        print(f"[SPECIAL_TOKENS] {self.special_token_ids}")

        self.max_num_frames = max_num_frames
        self.max_pixels = max_pixels
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        # Knockout env settings
        self.save_type = os.environ.get("LMM_EVAL_SAVE_TYPE", "baseline")
        self.output_dir = os.environ.get("LMM_EVAL_OUTPUT_DIR", "./knockout_results")
        self.model_size = os.environ.get("KNOCKOUT_MODEL_SIZE", "3B")
        self.gt_field = os.environ.get("KNOCKOUT_GT_FIELD", "answer")

        actual_layers = len(self.model.model.layers)
        expected_layers = get_num_layers(self.model_size)
        if actual_layers != expected_layers:
            eval_logger.warning(
                f"Model has {actual_layers} layers but KNOCKOUT_MODEL_SIZE={self.model_size} "
                f"expects {expected_layers}."
            )

        print(f"\n{'='*60}")
        print(f"KNOCKOUT MODE: {self.save_type}")
        print(f"Output dir: {self.output_dir}")
        print(f"Model size: {self.model_size} ({actual_layers} layers)")
        print(f"Attention: {attn_implementation}")
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
        return self._tokenizer.eos_token_id

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

    def loglikelihood(self, requests):
        raise NotImplementedError

    def flatten(self, input):
        return [item for sub in input for item in sub]

    def _check_video_has_audio(self, video_path):
        try:
            clip = VideoFileClip(video_path)
            has_audio = clip.audio is not None
            clip.close()
            return has_audio
        except Exception as e:
            eval_logger.warning(f"Could not check audio for {video_path}: {e}")
            return False

    def _build_input_dict(self, context: str, visual) -> dict:
        """Convert lmms-eval (context, visual) into video-SALMONN-2+ input_dict format."""
        input_dict = {
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ]
        }

        if visual is None:
            input_dict["conversations"][0]["value"] = context
            return input_dict

        if isinstance(visual, (list, tuple)):
            if len(visual) == 0:
                input_dict["conversations"][0]["value"] = context
                return input_dict
            visual = visual[0]

        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            input_dict["video"] = visual
            input_dict["use_audio"] = self._check_video_has_audio(visual)
            input_dict["conversations"][0]["value"] = f"<video>\n{context}"
        elif isinstance(visual, Image.Image):
            input_dict["image"] = visual
            input_dict["conversations"][0]["value"] = f"<image>\n{context}"
        else:
            eval_logger.warning(f"Unknown visual type: {type(visual)}, falling back to text-only")
            input_dict["conversations"][0]["value"] = context

        return input_dict

    def _strip_and_move(self, raw_inputs: dict) -> dict:
        """Strip non-tensor fields and move tensors to GPU."""
        for key in ["video", "image", "prompt", "ref", "audio", "use_audio", "should_use"]:
            raw_inputs.pop(key, None)
        target_device = "cuda" if self.device_map == "auto" else self.device
        return {
            k: v.to(target_device) for k, v in raw_inputs.items() if isinstance(v, torch.Tensor)
        }

    # ============================================================
    # Input preparation
    # ============================================================
    def _prepare_inputs(self, context, visual, task):
        """Build input_dict -> _get_item -> add Answer:( suffix -> strip and move."""
        # Apply AV SpeakerBench-specific instruction suffix on context
        context = context + "\nOnly give the best option."

        input_dict = self._build_input_dict(context, visual)
        raw_inputs = self.test_data._get_item(input_dict)

        # Append "Answer:(" suffix
        answer_prefix = "Answer:("
        suffix_ids = self._tokenizer.encode(answer_prefix, add_special_tokens=False)
        suffix_tensor = torch.tensor([suffix_ids], dtype=raw_inputs["input_ids"].dtype)
        raw_inputs["input_ids"] = torch.cat([raw_inputs["input_ids"], suffix_tensor], dim=1)

        if "attention_mask" in raw_inputs and isinstance(raw_inputs["attention_mask"], torch.Tensor):
            mask_suffix = torch.ones((1, len(suffix_ids)), dtype=raw_inputs["attention_mask"].dtype)
            raw_inputs["attention_mask"] = torch.cat([raw_inputs["attention_mask"], mask_suffix], dim=1)

        if "position_ids" in raw_inputs and isinstance(raw_inputs["position_ids"], torch.Tensor):
            pos = raw_inputs["position_ids"]
            T_old = pos.shape[-1]
            new_positions = torch.arange(T_old, T_old + len(suffix_ids), dtype=pos.dtype).view(1, 1, -1).expand(3, 1, -1)
            raw_inputs["position_ids"] = torch.cat([pos, new_positions], dim=-1)

            ## debug the the whole input
        # all_ids = raw_inputs["input_ids"][0].tolist()
        # print(f"  [SUFFIX CHECK] total length: {len(all_ids)}")
        # print(f"  [SUFFIX CHECK] last 20 token ids: {all_ids[-20:]}")
        # print(f"  [SUFFIX CHECK] full decoded input:")
        # print(self._tokenizer.decode(all_ids))
        # print(f"  [SUFFIX CHECK] end of decode marker ^^^")
            
        # exit()

        # Save input_ids list BEFORE stripping
        input_ids_list = raw_inputs["input_ids"][0].cpu().tolist()
        video_grid_thw_list = (
            raw_inputs["video_grid_thw"].cpu().tolist()
            if "video_grid_thw" in raw_inputs and isinstance(raw_inputs["video_grid_thw"], torch.Tensor)
            else None
        )

        # print(self._tokenizer.decode(input_ids_list))
        # exit()
        # print(f"second-to-last:    {repr(self._tokenizer.decode([input_ids_list[-2]]))}")
            # exit()


        inputs = self._strip_and_move(raw_inputs)
        return inputs, input_ids_list, video_grid_thw_list

    # ============================================================
    # Prefill-only forward pass
    # ============================================================
    # def _prefill_forward(self, inputs):
    #     """Run prefill-only forward pass, return logits at last position."""
    #     with torch.no_grad():
    #         outputs = self.model(
    #             **inputs,
    #             use_cache=False,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #             return_dict=True,
    #         )
    #     return outputs.logits[0, -1, :]
    

    def _generate_forward(self, inputs, max_new_tokens=32):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=self.use_cache,
                return_dict_in_generate=True,
                output_logits=True,
            )
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text, generated_ids, outputs.logits

    # ============================================================
    # BASELINE MODE
    # ============================================================
    def _run_baseline(self, requests: List[Instance]) -> List[str]:
        res = []
        baseline_file = os.path.join(self.output_dir, "baseline", "baseline.jsonl")
        os.makedirs(os.path.dirname(baseline_file), exist_ok=True)

        saved_results = load_existing_results(baseline_file)
        if saved_results:
            eval_logger.info(f"Crash recovery: found {len(saved_results)} already-saved samples")

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Baseline")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            contexts = list(contexts)
            task = task[0]
            split = split[0]

            if doc_id[0] in saved_results:
                saved_answer = saved_results[doc_id[0]]
                for context in contexts:
                    res.append(saved_answer if saved_answer else "")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), saved_answer if saved_answer else "")
                    pbar.update(1)
                continue

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            # Flatten visuals, take first one (matches _build_input_dict expectation)
            try:
                inputs, input_ids_list, video_grid_thw_list = self._prepare_inputs(contexts[0], visuals[0] if visuals else None, task)
            except Exception as e:
                eval_logger.error(f"Error preparing inputs for doc_id={doc_id[0]}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue
            

            # doc = self.task_dict[task][split][doc_id[0]]
            # if task == "av_speakerbench_audiovisual":
            #     _, all_choices, _ = _parse_choices(doc["choices"])
            # else:
            #     raise ValueError("unsupported task")

            try:
                raw_response, generated_ids, logits_per_step = self._generate_forward(inputs, max_new_tokens=32)
                # parsed_answer = parse_multi_choice_response(raw_response, all_choices)
            except Exception as e:
                eval_logger.error(f"Error in forward pass for doc_id={doc_id[0]}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue


            letter_logits = logits_per_step[0][0] # first token for the generated token and first batch whihc onlt 1 batch

            # DEBUG: print top-5 tokens from step 0 logits
            top_probs, top_ids = torch.topk(torch.softmax(letter_logits, dim=0), 5)
            print(f"[doc_id={doc_id[0]}] step 0 top-5:")
            for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
                print(f"  p={p:.4f} tok={self._tokenizer.decode([tid])!r}")
            
            mcq = extract_mcq_probs(letter_logits, self._tokenizer)

            parsed_answer = mcq["predicted"]

            doc = self.task_dict[task][split][doc_id[0]]

            if task == "worldsense_shorts":
                question_with_options = doc["question"] + "\n" + "\n".join(doc["candidates"])
            elif task == "av_speakerbench_audiovisual":
                choices = doc["choices"]
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                question_with_options = doc["question"] + "\n" + "\n".join(choices)
            else:
                raise ValueError(f"Unknown dataset: {task}")

            # Extract token positions with Video-SALMONN-specific function
            answer_letter = doc.get("answer")
            token_positions = extract_token_positions_vs2(
                input_ids_list,
                self.special_token_ids,
                question_text_str=question_with_options,
                tokenizer=self._tokenizer,
                answer_letter=answer_letter,
                video_grid_thw=video_grid_thw_list,
            )

            # # ==== DEBUG: verify extracted positions ====
            # print(f"\n--- doc_id={doc_id[0]} ---")
            # print(f"seq_len:           {token_positions.seq_len}")
            # print(f"system:            {token_positions.system}")
            # print(f"user_instruction:  {token_positions.user_instruction}")
            # print(f"video_chunks:      {len(token_positions.video_chunks)} chunks")
            # print(f"  first 3:         {token_positions.video_chunks[:3]}")
            # print(f"  last 3:          {token_positions.video_chunks[-3:]}")
            # print(f"audio_chunks:      {len(token_positions.audio_chunks)} chunks")
            # print(f"  first 3:         {token_positions.audio_chunks[:3]}")
            # print(f"  last 3:          {token_positions.audio_chunks[-3:]}")
            # print(f"video_patches:     {len(token_positions.video_temporal_patches)} patches")
            # print(f"  first 3:         {token_positions.video_temporal_patches[:3]}")
            # print(f"  last 3:          {token_positions.video_temporal_patches[-3:]}")
            # print(f"instruction_prefix: {token_positions.instruction_prefix}")
            # print(f"question_text:     {token_positions.question_text}")
            # print(f"prompt_suffix:     {token_positions.prompt_suffix}")
            # print(f"last_token:        {token_positions.last_token}")

            # print(f"\n--- Option-level ---")
            # print(f"ground_truth:        {doc.get('answer')}")
            # print(f"non_option_question: {token_positions.non_option_question}")
            # print(f"true_option:         {token_positions.true_option}")
            # print(f"false_options:       {token_positions.false_options}")

            # print("\n--- Decoded ranges ---")
            # if token_positions.instruction_prefix:
            #     p_s, p_e = token_positions.instruction_prefix
            #     print(f"instruction_prefix decoded: {repr(self._tokenizer.decode(input_ids_list[p_s:p_e+1]))}")

            # if token_positions.question_text:
            #     q_s, q_e = token_positions.question_text
            #     print(f"question decoded:  {repr(self._tokenizer.decode(input_ids_list[q_s:q_e+1]))}")

            # if token_positions.prompt_suffix:
            #     s_s, s_e = token_positions.prompt_suffix
            #     print(f"suffix decoded:    {repr(self._tokenizer.decode(input_ids_list[s_s:s_e+1]))}")

            # if token_positions.non_option_question:
            #     s, e = token_positions.non_option_question
            #     print(f"non_option_question decoded: {repr(self._tokenizer.decode(input_ids_list[s:e+1]))}")

            # if token_positions.true_option:
            #     s, e = token_positions.true_option
            #     print(f"true_option decoded: {repr(self._tokenizer.decode(input_ids_list[s:e+1]))}")

            # if token_positions.false_options:
            #     for i, (s, e) in enumerate(token_positions.false_options):
            #         print(f"false_option[{i}] decoded: {repr(self._tokenizer.decode(input_ids_list[s:e+1]))}")

            # print(f"\nlast token decoded: {repr(self._tokenizer.decode([input_ids_list[-1]]))}")
            # print(f"second-to-last:    {repr(self._tokenizer.decode([input_ids_list[-2]]))}")
            # exit()
            # # ==== END DEBUG ====
  

            gt = doc["answer"]
            is_correct = (parsed_answer == gt)

            if task == "worldsense_shorts":
                record = {
                    "doc_id": doc_id[0],
                    "task": task,
                    "split": split,
                    "sample_idx" : doc["index"],
                    "task_type": doc["task_type"],
                    "task_domain": doc["task_domain"],
                    "audio_class": doc["audio_class"],
                    "video_path": doc["video_path"],
                    "audio_path": doc["audio_path"],
                    "question": doc["question"],
                    "candidates": doc["candidates"],
                    "parsed_answer": parsed_answer,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "probs": mcq["probs"],
                    "logits_raw": mcq["logits_raw"],
                    "token_positions": token_positions.to_dict(),
                    "seq_len": token_positions.seq_len,
                    "num_video_chunks": len(token_positions.video_chunks),
                    "num_audio_chunks": len(token_positions.audio_chunks),
                    "num_temporal_patches": len(token_positions.video_temporal_patches),
                }
            elif task == "av_speakerbench_audiovisual":
                record = {
                    # ---- IDs ----
                    "doc_id": doc_id[0],
                    "task": task,
                    "split": split,
                    # ---- Dataset fields ----
                    "question_id": doc.get("question_id", ""),
                    "video_id": doc.get("video_id", ""),
                    "category": doc.get("category", ""),
                    "sub_category": doc.get("sub_category", ""),
                    "task_id": doc.get("task_id", ""),
                    "question": doc["question"],
                    "choices": doc["choices"],
                    "start_time": doc.get("start_time", ""),
                    "end_time": doc.get("end_time", ""),
                    "global_consistent": doc.get("global_consistent", ""),
                    # ---- Paths ----
                    "video_path": doc.get("video_path", ""),
                    "audio_path": doc.get("audio_path", ""),
                    "visual_path": doc.get("visual_path", ""),
                    "audio_visual_path": doc.get("audio_visual_path", ""),
                    # ---- Results ----
                    "parsed_answer": parsed_answer,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "probs": mcq["probs"],
                    "logits_raw": mcq["logits_raw"],
                    # ---- Token positions (model-specific, not dataset-specific) ----
                    "token_positions": token_positions.to_dict(),
                    "seq_len": token_positions.seq_len,
                    "num_video_chunks": len(token_positions.video_chunks),
                    "num_audio_chunks": len(token_positions.audio_chunks),
                    "num_temporal_patches": len(token_positions.video_temporal_patches),
                }
            else:
                raise ValueError(f"Unknown dataset: {task} please fix this to use others dataset")
            append_jsonl(baseline_file, record)

            status = "CORRECT" if is_correct else "WRONG"
            eval_logger.info(
                f"[{status}] doc_id={doc_id[0]} pred={parsed_answer} gt={gt} "
                f"P(A)={mcq['probs'].get('A',0):.3f} P(B)={mcq['probs'].get('B',0):.3f} "
                f"P(C)={mcq['probs'].get('C',0):.3f} P(D)={mcq['probs'].get('D',0):.3f} "
                f"seq_len={token_positions.seq_len}"
            )

            gen_kwargs = all_gen_kwargs[0]
            for i, context in enumerate(contexts):
                res.append(parsed_answer if parsed_answer else "")
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), parsed_answer if parsed_answer else "")
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
        Safe to restart without duplicates.
        
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

        config = KNOCKOUT_CONFIGS[config_id]
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

        # ---- Crash recovery: load already-saved doc_ids ----
        saved_ids = load_existing_knockout_ids(center_file)

        print(f"\n{'='*60}")
        print(f"KNOCKOUT EXPERIMENT")
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

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc=f"KO cfg{config_id} c{center}")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        knockout_count = 0

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            contexts = list(contexts)
            task = task[0]
            split = split[0]
            did = doc_id[0]

            base_record = baseline_data.get(did, {})
            baseline_answer = base_record.get("parsed_answer", "")
            is_correct_sample = did in correct_ids

            # Verify question_id matches baseline
            if task == "av_speakerbench_audiovisual":
                current_question_id = self.task_dict[task][split][did].get("question_id")
                baseline_question_id = base_record.get("question_id")
            elif task == "worldsense_shorts":
                #raise NotImplementedError("Add question_id check for worldsense_shorts")
                current_question_id = self.task_dict[task][split][did].get("index")
                baseline_question_id = base_record.get("sample_idx")
            else:
                raise NotImplementedError(f"Add question_id check for {task}")

            if current_question_id != baseline_question_id:
                raise ValueError(f"question id mismatch: {current_question_id} vs {baseline_question_id}")
            
            # print(current_question_id)
            # print(baseline_question_id)
            # exit()

            if did in saved_ids:
                # Verify question_id matches baseline
                if task == "av_speakerbench_audiovisual":
                    current_question_id = self.task_dict[task][split][did].get("question_id")
                    baseline_question_id = base_record.get("question_id")
                elif task == "worldsense_shorts":
                    current_question_id = self.task_dict[task][split][did].get("index")
                    baseline_question_id = base_record.get("index")
                else:
                    raise NotImplementedError(f"Add question_id check for {task}")

                if current_question_id != baseline_question_id:
                    raise ValueError(f"question id mismatch: {current_question_id} vs {baseline_question_id}")
                # pbar.update(1)
                for context in contexts:
                    res.append(baseline_answer if baseline_answer else "")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), baseline_answer if baseline_answer else "")
                    pbar.update(1)
                continue

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
                }

                if task == "worldsense_shorts":
                    skip_record["task_type"] = base_record.get("task_type", "")
                    skip_record["task_domain"] = base_record.get("task_domain", "")
                elif task == "av_speakerbench_audiovisual":
                    skip_record["category"] = base_record.get("category", "")
                    skip_record["sub_category"] = base_record.get("sub_category", "")

                append_jsonl(center_file, skip_record)
                saved_ids.add(did)

                for context in contexts:
                    res.append(baseline_answer if baseline_answer else "")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), baseline_answer if baseline_answer else "")
                    pbar.update(1)
                continue

            # Prepare inputs
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            try:
                inputs, input_ids_list, video_grid_thw_list = self._prepare_inputs(contexts[0], visuals[0] if visuals else None, task)
            except Exception as e:
                eval_logger.error(f"Error preparing inputs for doc_id={did}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            # Reconstruct question_with_options for position extraction
            if task == "worldsense_shorts":
                question_with_options = base_record["question"] + "\n" + "\n".join(base_record["candidates"])
            elif task == "av_speakerbench_audiovisual":
                choices = base_record["choices"]
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                question_with_options = base_record["question"] + "\n" + "\n".join(choices)
            else:
                raise ValueError(f"Unknown dataset: {task}")

            answer_letter = base_record.get("ground_truth")
            token_positions = extract_token_positions_vs2(
                input_ids_list,
                self.special_token_ids,
                question_text_str=question_with_options,
                tokenizer=self._tokenizer,
                answer_letter=answer_letter,
                video_grid_thw=video_grid_thw_list,
            )

            # Set token positions on each attention layer for debug
            for layer in self.model.model.layers:
                layer.self_attn._token_positions = token_positions

            base_probs = base_record["probs"]
            gt = base_record["ground_truth"]
            base_predicted = base_record["parsed_answer"]


            # print(f"model type: {type(self.model).__name__}")
            # print(f"hasattr thinker: {hasattr(self.model, 'thinker')}")
            # print(f"hasattr model.layers: {hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')}")
            # if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            #     print(f"num layers: {len(self.model.model.layers)}")
            # exit()

            # # Run knockout forward
            # doc = self.task_dict[task][split][doc_id[0]]
            # if task == "av_speakerbench_audiovisual":
            #     _, all_choices, _ = _parse_choices(doc["choices"])
            # else:
            #     raise ValueError("why here")
            #     all_choices = ["A", "B", "C", "D"]
            try:
                with KnockoutManager(self.model, token_positions, config_id, target_layers) as manager:
                    # logits = self._prefill_forward(inputs)
                    raw_response_ko, generated_ids_ko, logits_per_step_ko = self._generate_forward(inputs, max_new_tokens=32)
            except Exception as e:
                eval_logger.error(f"Error knockout cfg={config_id} c={center} doc={did}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue
            ko_letter_logits = logits_per_step_ko[0][0]
            knockout_mcq = extract_mcq_probs(ko_letter_logits, self._tokenizer)
            # knockout_predicted = knockout_mcq["predicted"]
            p_change = compute_p_change(
                {"probs": base_probs},
                {"probs": knockout_mcq["probs"]},
                gt, base_predicted,
            )

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
            }

            if task == "worldsense_shorts":
                result["task_type"] = base_record.get("task_type", "")
                result["task_domain"] = base_record.get("task_domain", "")
            elif task == "av_speakerbench_audiovisual":
                result["category"] = base_record.get("category", "")
                result["sub_category"] = base_record.get("sub_category", "")
            else:
                raise ValueError(f"Unknown dataset: {task}")

            append_jsonl(center_file, result)
            saved_ids.add(did)
            knockout_count += 1

            eval_logger.info(
                f"[{knockout_count}] doc={did} cfg={config_id} c={center} "
                f"{base_predicted}>{knockout_mcq['predicted']} gt={gt} "
                f"diff={p_change['relative_diff']:.1f}%"
            )

            for i, context in enumerate(contexts):
                res.append(baseline_answer)
                self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), baseline_answer)
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
            raise ValueError(f"Unknown LMM_EVAL_SAVE_TYPE='{self.save_type}'. Use 'baseline' or 'knockout'.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation not supported for knockout")