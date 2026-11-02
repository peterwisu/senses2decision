"""
Qwen2.5-Omni Knockout Model Wrapper for lmms-eval

Dedicated model wrapper for knockout experiments. Keeps the original
qwen2_5_omni.py untouched for regular benchmarking.

Modes (controlled by LMM_EVAL_SAVE_TYPE env var):
  "baseline"  — Prefill-only, save MCQ probabilities + token positions.
                Run ONCE. Produces baseline.jsonl with all samples.
                lmms-eval scoring tells you which are correct.
  "knockout"  — Prefill with knockout hooks.
                Reads baseline.jsonl to filter correct-only samples.

Env vars:
  LMM_EVAL_SAVE_TYPE    "baseline" or "knockout"
  LMM_EVAL_OUTPUT_DIR   Directory to save results
  KNOCKOUT_MODEL_SIZE   "7B" or "3B" (for layer group definitions)
  KNOCKOUT_GT_FIELD     Field name in dataset doc for ground truth answer (default: "answer")

Knockout-specific env vars:
  KNOCKOUT_CONFIG_IDS    "all" or comma-separated e.g. "1,3,7" (default: all)
  KNOCKOUT_SWEEP_TYPE    "window" or "group" (default: window)
  KNOCKOUT_WINDOW_K      Window size for sliding window (default: 9)
  KNOCKOUT_SAMPLE_MODE   "correct" (default) or "all"
  KNOCKOUT_SWEEP_RANGE   "all" or "start-end" e.g. "0-0" for single center

Usage:
  # Baseline
  LMM_EVAL_SAVE_TYPE=baseline \
  LMM_EVAL_OUTPUT_DIR=/path/to/output \
  KNOCKOUT_MODEL_SIZE=7B \
  python -m lmms_eval --model qwen2_5_omni_knockout --tasks worldsense_shorts ...

  # Knockout (1 config, 1 center)
  LMM_EVAL_SAVE_TYPE=knockout \
  KNOCKOUT_CONFIG_IDS=4 \
  KNOCKOUT_SWEEP_RANGE=0-0 \
  python -m lmms_eval --model qwen2_5_omni_knockout --tasks worldsense_shorts ...
"""

import json
import os
import pickle
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
from lmms_eval.models.model_utils.load_video import read_video
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")

from utils.knockout.configs import (
    KNOCKOUT_CONFIGS,
    extract_token_positions,
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


import ast
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
    """Load already-saved doc_ids from a knockout center JSONL file.
    Returns set of doc_ids already processed (both knockout and skipped)."""
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
    """Append one JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================
# Model
# ============================================================

@register_model("qwen2_5_omni_knockout")
class Qwen2_5_OmniKnockout(lmms):
    """Qwen2.5-Omni knockout experiment wrapper."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "eager",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        exp_config: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if attn_implementation != "eager":
            eval_logger.warning(f"attn_implementation={attn_implementation} but knockout REQUIRES eager. Forcing eager.")
            attn_implementation = "eager"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

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

        # print(exp_config)
        # exit()
        if exp_config and os.path.exists(exp_config):
            with open(exp_config) as f:
                self.exp_config = yaml.safe_load(f)
            eval_logger.info(f"Experiment config: {self.exp_config}")
        else:
            self.exp_config = None

        from models.qwen2_5.modelling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        print("### Using default method (eager attention) ###")

        Qwen2_5OmniForConditionalGeneration._tp_plan = [] if Qwen2_5OmniForConditionalGeneration._tp_plan is None else Qwen2_5OmniForConditionalGeneration._tp_plan
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation=attn_implementation
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
        print(f"KNOCKOUT MODE: {self.save_type}")
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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

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

    # ============================================================
    # Input preparation
    # ============================================================
    def _prepare_inputs(self, contexts, visuals):
        """Build message, process media, return (inputs, current_use_audio)."""
        current_use_audio = False
        message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

        for i, context in enumerate(contexts):
            if len(visuals) > 0:
                visual = visuals[i] if i < len(visuals) else None

                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                    current_use_audio = self._check_if_video_has_audio(visual)
                    if self.use_custom_video_loader:
                        frames = read_video(visual, num_frm=self.max_num_frames, fps=self.fps)
                        image_contents = []
                        for frame in frames:
                            img = Image.fromarray(frame)
                            b64 = encode_image_to_base64(img, image_format="JPEG", convert_rgb=True, quality=85)
                            image_contents.append(f"data:image/jpeg;base64,{b64}")
                        message.append({"role": "user", "content": [
                            {"type": "video", "video": image_contents},
                            {"type": "text", "text": context},
                        ]})
                    else:
                        message.append({"role": "user", "content": [
                            {"type": "video", "video": visual},
                            {"type": "text", "text": context},
                        ]})
                elif isinstance(visual, Image.Image):
                    message.append({"role": "user", "content": [
                        {"type": "image", "image": visual},
                        {"type": "text", "text": context},
                    ]})
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    single_message = {"role": "user", "content": []}
                    for v in visual:
                        single_message["content"].append({"type": "image", "image": v})
                    single_message["content"].append({"type": "text", "text": context})
                    message.append(single_message)
                elif 'AudioDecoder' in type(visual).__name__:
                    current_use_audio = True
                    decoded = visual.get_all_samples()
                    samples = decoded.data.numpy().flatten()
                    sr = decoded.sample_rate
                    visual = {"array": samples, "sampling_rate": sr}
                    audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                    audio_splits = split_audio(audio, 4800000)
                    single_message = {"role": "user", "content": []}
                    for j in range(len(audio_splits)):
                        single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                    single_message["content"].append({"type": "text", "text": context})
                    message.append(single_message)
                elif isinstance(visual, (list, tuple)) and len(visual) > 0 and 'AudioDecoder' in type(visual[0]).__name__:
                    current_use_audio = True
                    for ii, v in enumerate(visual):
                        decoded = v.get_all_samples()
                        samples = decoded.data.numpy().flatten()
                        sr = decoded.sample_rate
                        v = {"array": samples, "sampling_rate": sr}
                        audio = self.resample_audio(v["array"], v["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)
                        single_message = {"role": "user", "content": []}
                        for j in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)
                elif isinstance(visual, dict):
                    current_use_audio = True
                    audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                    audio_splits = split_audio(audio, 4800000)
                    single_message = {"role": "user", "content": []}
                    for ii in range(len(audio_splits)):
                        single_message["content"].append({"type": "audio", "audio": audio_splits[ii]})
                    single_message["content"].append({"type": "text", "text": context})
                    message.append(single_message)
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):
                    current_use_audio = True
                    for ii, v in enumerate(visual):
                        audio = self.resample_audio(v["array"], v["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)
                        single_message = {"role": "user", "content": []}
                        for j in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)
                else:
                    raise ValueError(f"Unknown visual type: {type(visual)}")

        text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        text = text + "Answer:("

        # print(text)
        # exit()
        audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)
        
        # print(text)

        # # print(f"doc_id: {doc_id[0]}")
        # print(f"visual (video path): {visuals[0]}")
        # print("Audios:", type(audios), audios[0].shape )
        # print("Images:", type(images) )
        # print("Videos:", type(videos), videos[0].shape)
        
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=current_use_audio,
        )

        input_ids = inputs["input_ids"][0]
        video_token_id = self.model.config.thinker_config.video_token_id
        audio_token_id = self.model.config.thinker_config.audio_token_id
        image_token_id = self.model.config.thinker_config.image_token_id

        # print(f"Total seq len: {len(input_ids)}")
        # print(f"Video tokens: {(input_ids == video_token_id).sum().item()}")
        # print(f"Audio tokens: {(input_ids == audio_token_id).sum().item()}")
        # print(f"Image tokens: {(input_ids == image_token_id).sum().item()}")
        # print(f"Text tokens: {len(input_ids) - (input_ids == video_token_id).sum().item() - (input_ids == audio_token_id).sum().item() - (input_ids == image_token_id).sum().item()}")
        # exit()


        if self.device_map == "auto":
            inputs = inputs.to("cuda").to(self.model.dtype)
        else:
            inputs = inputs.to(self.model.device).to(self.model.dtype)

        return inputs, current_use_audio

    # # ============================================================
    # # Prefill-only forward pass (shared by baseline and knockout)
    # # ============================================================
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
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), saved_answer if saved_answer else "")
                    pbar.update(1)
                continue

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

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

            # input_ids_list = inputs["input_ids"][0].cpu().tolist()
            # token_positions = extract_token_positions(input_ids_list, special_tokens)

            input_ids_list = inputs["input_ids"][0].cpu().tolist()
            doc = self.task_dict[task][split][doc_id[0]]

           
            if task == "worldsense_shorts":
                question_with_options = doc["question"] + "\n" + "\n".join(doc["candidates"])
            elif task == "av_speakerbench_audiovisual":
                # question_with_options = doc["question"] + "\n" + "\n".join(doc["choices"])
                choices = doc["choices"]
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                question_with_options = doc["question"] + "\n" + "\n".join(choices)
            else:
                raise ValueError(f"Unknown dataset: {task} please fix this to use others dataset")


            # DEBUG: see what we're searching for
            # print(f"question_with_options repr: {repr(question_with_options)}")
            # exit()
            

            token_positions = extract_token_positions(
                input_ids_list,
                special_tokens,
                question_text_str=question_with_options,
                tokenizer=self.tokenizer,
            )

            # # # --- DEBUG: remove after verifying ---
            # print(f"\n--- doc_id={doc_id[0]} ---")
            # print(f"user_instruction: {token_positions.user_instruction}")
            # print(f"question_text:    {token_positions.question_text}")
            # print(f"prompt_suffix:    {token_positions.prompt_suffix}")
            # print(f"last_token:       {token_positions.last_token}")

            # print("--------")
            # if token_positions.question_text:
            #     q_s, q_e = token_positions.question_text
            #     print(f"question decoded:  {self.tokenizer.decode(input_ids_list[q_s:q_e+1])}")

            # print("--------")
            # if token_positions.prompt_suffix:
            #     s_s, s_e = token_positions.prompt_suffix
            #     print(f"suffix decoded:    {self.tokenizer.decode(input_ids_list[s_s:s_e+1])}")
            # # print(f"last 5 decoded:    {self.tokenizer.decode(input_ids_list[-5:])}")
            # print("---")

            # print(f"last token decoded: {repr(self.tokenizer.decode([input_ids_list[-1]]))}")
            # print(f"second-to-last decoded: {repr(self.tokenizer.decode([input_ids_list[-2]]))}")
            
            
            # print(self.tokenizer.decode(input_ids_list))
            
            # # print(doc)
            # exit()
            # --- END DEBUG --- 
            

            doc = self.task_dict[task][split][doc_id[0]]
            gt = doc["answer"]
            is_correct = (parsed_answer == gt)

            if task == "worldsense_shorts":
                record = {
                    # ---- IDs ----
                    "doc_id": doc_id[0],
                    "task": task,
                    "split": split,
                    # ---- Dataset fields ----
                    "sample_idx" : doc["index"],
                    "task_type": doc["task_type"],
                    "task_domain": doc["task_domain"],
                    "audio_class": doc["audio_class"],
                    "video_path": doc["video_path"],
                    "audio_path": doc["audio_path"],
                    "question": doc["question"],
                    "candidates": doc["candidates"],
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

            # Verify question_id matches baseline
            if task == "av_speakerbench_audiovisual":
                current_question_id = self.task_dict[task][split][did].get("question_id")
                baseline_question_id = base_record.get("question_id")
            elif task == "worldsense_shorts":
                current_question_id = self.task_dict[task][split][did].get("index")
                baseline_question_id = base_record.get("sample_idx")
                # print(current_question_id) 
                # print(baseline_question_id)


                
                # exit()
            else:
                raise NotImplementedError(f"Add question_id check for {task}")

            print(current_question_id, baseline_question_id)
            if current_question_id != baseline_question_id:
                raise ValueError(f"question id mismatch: {current_question_id} vs {baseline_question_id}")
            
            # print(current_question_id)
            # print(baseline_question_id)
            # exit()
            
            # print(current_video_id)
            # print(baseline_video_id)
            # print(current_video_id == baseline_video_id)


            if did in saved_ids:
                # verify video_id matches baseline
                if task == "av_speakerbench_audiovisual":
                    current_question_id = self.task_dict[task][split][did].get("question_id")
                    baseline_question_id = base_record.get("question_id")
                elif task == "worldsense_shorts":
                    current_question_id = self.task_dict[task][split][did].get("index")
                    baseline_question_id = base_record.get("index")
                    # print(current_question_id) 
                    # print(baseline_question_id)
                    
                    # exit()


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

            # # ---- Already saved: skip ----
            # if did in saved_ids:

            #     for context in contexts:
            #         res.append(baseline_answer if baseline_answer else "")
            #         self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), baseline_answer if baseline_answer else "")
            #         pbar.update(1)
            #     continue

            # ---- Wrong baseline: write skip record ----
            if sample_mode == "correct" and not is_correct_sample:
                # skip_record = {
                #     "doc_id": did,
                #     "config_id": config_id,
                #     "config_name": config.name,
                #     "center": center,
                #     "target_layers": target_layers,
                #     "task_type": base_record.get("task_type", ""),
                #     "task_domain": base_record.get("task_domain", ""),
                #     "ground_truth": base_record.get("ground_truth", ""),
                #     "base_predicted": baseline_answer,
                #     "knockout_predicted": None,
                #     "skipped": True,
                #     "skip_reason": "baseline_wrong",
                #     "base_probs": base_record.get("probs", {}),
                #     "knockout_probs": {},
                #     "relative_diff": None,
                #     "relative_diff_gt": None,
                # }

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

            # ---- Prepare inputs ----
            contexts = list(contexts)
            contexts[0] = contexts[0] + "\nOnly give the best option."
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            try:
                inputs, current_use_audio = self._prepare_inputs(contexts, visuals)
            except Exception as e:
                eval_logger.error(f"Error preparing inputs for doc_id={did}: {e}")
                for context in contexts:
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), "")
                    pbar.update(1)
                continue

            # input_ids_list = inputs["input_ids"][0].cpu().tolist()
            # token_positions = extract_token_positions(input_ids_list, special_tokens)
            # base_probs = base_record["probs"]

            input_ids_list = inputs["input_ids"][0].cpu().tolist()
            # question_with_options = base_record["question"] + "\n" + "\n".join(base_record["candidates"])

           
            if task == "worldsense_shorts":
                question_with_options = base_record["question"] + "\n" + "\n".join(base_record["candidates"])
            elif task == "av_speakerbench_audiovisual":
                # question_with_options = doc["question"] + "\n" + "\n".join(doc["choices"])
                choices = base_record["choices"]
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                question_with_options = base_record["question"] + "\n" + "\n".join(choices)

            else:
                raise ValueError(f"Unknown dataset: {task} please fix this to use others dataset")

            answer_letter = base_record.get("ground_truth") 
            # print(answer_letter)
            token_positions = extract_token_positions(
                input_ids_list,
                special_tokens,
                question_text_str=question_with_options,
                tokenizer=self.tokenizer,
                answer_letter=answer_letter
            )

            
            ### FOR DEBUG only
            # Set token positions on each attention layer for debug plotting
            for layer in self.model.thinker.model.layers:
                layer.self_attn._token_positions = token_positions
            #######
            # for idx, tok_id in enumerate(input_ids_list):
            #     decoded = self.tokenizer.decode([tok_id])
            #     print(f"{idx:5d} | {tok_id:8d} | {repr(decoded)}")
            # exit()
            # # --- DEBUG: remove after verifying ---
            # print(f"\n--- doc_id={did} ---")
            # print(f"user_instruction: {token_positions.user_instruction}")
            # print(f"system: {token_positions.system} ")
            # print(f"video_chunks:     {token_positions.video_chunks}")
            # print(f"audio_chunks:     {token_positions.audio_chunks}")
            # print(f"video_patches:    {token_positions.video_temporal_patches}")
            # print(f"instruction_prefix: {token_positions.instruction_prefix}")
            # print(f"question_text:    {token_positions.question_text}")
            # print(f"prompt_suffix:    {token_positions.prompt_suffix}")
            # print(f"last_token:       {token_positions.last_token}")
            # print(f"non_option_question: {token_positions.non_option_question}")
            # print(f"true_option:      {token_positions.true_option}")
            # print(f"false_options:    {token_positions.false_options}")

            # print("----"*10)

            # print("question_with_options : ", question_with_options)
            # print("GT ",base_record.get("ground_truth", ""))

            # print("**"*20)
            # if token_positions.question_text:
            #     q_s, q_e = token_positions.question_text
            #     print(f"question decoded:  {self.tokenizer.decode(input_ids_list[q_s:q_e+1])}")
            # print("----"*10)
            # if token_positions.prompt_suffix:
            #     s_s, s_e = token_positions.prompt_suffix
            #     print(f"suffix decoded:    {self.tokenizer.decode(input_ids_list[s_s:s_e+1])}")
            # print("----"*10)
            # exit()
            # # --- END DEBUG --- 

            # # --- DEBUG MASK ---
            # from utils.knockout.hooks import build_knockout_mask_fast

            # config = KNOCKOUT_CONFIGS[config_id]
            # device = next(self.model.thinker.model.parameters()).device
            # dtype = next(self.model.thinker.model.parameters()).dtype

            # mask = build_knockout_mask_fast(config, token_positions, device, dtype)

            # last = token_positions.last_token
            # last_row = mask[last, :]

            # blocked = (last_row < -1000).nonzero().squeeze().tolist()
            # if isinstance(blocked, int):
            #     blocked = [blocked]

            # q_s, q_e = token_positions.question_text
            # s_s, s_e = token_positions.prompt_suffix

            # print(f"Config {config_id} ({config.name})")
            # print(f"Blocked positions: {min(blocked)} to {max(blocked)} ({len(blocked)} total)")
            # print(f"question_text:     {q_s} to {q_e} ({q_e - q_s + 1} tokens)")
            # print(f"prompt_suffix:     {s_s} to {s_e} ({s_e - s_s + 1} tokens)")
            # print(f"last_token:        {last}")
            # print(f"mask[last, question_start] = {last_row[q_s].item():.1f}  (should be -inf)")
            # print(f"mask[last, question_end]   = {last_row[q_e].item():.1f}  (should be -inf)")
            # print(f"mask[last, suffix_start]   = {last_row[s_s].item():.1f}  (should be 0)")
            # print(f"mask[last, suffix_end]     = {last_row[s_e].item():.1f}  (should be 0)")
            # print(f"mask[last, last_token]     = {last_row[last].item():.1f}  (should be 0)")

            # print(last_row.min())
            # print((last_row == last_row.min()).nonzero(as_tuple=True)[0].tolist())
            # exit()
            # # --- END DEBUG ---

            base_probs = base_record["probs"]

            gt = base_record["ground_truth"]
            base_predicted = base_record["parsed_answer"]

            # ---- Run knockout forward pass ----
            try:
                with KnockoutManager(self.model, token_positions, config_id, target_layers) as manager:
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

            # result = {
            #     "doc_id": did,
            #     "config_id": config_id,
            #     "config_name": config.name,
            #     "center": center,
            #     "target_layers": target_layers,
            #     "task_type": base_record.get("task_type", ""),
            #     "task_domain": base_record.get("task_domain", ""),
            #     "ground_truth": gt,
            #     "base_predicted": base_predicted,
            #     "knockout_predicted": knockout_mcq["predicted"],
            #     "skipped": False,
            #     "base_probs": base_probs,
            #     "knockout_probs": knockout_mcq["probs"],
            #     **p_change,
            # }
            # ---- Common fields ----
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

            # ---- Dataset-specific fields ----
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

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported for knockout")