from typing import List, Optional, Tuple, Union
import os
import sys
import pickle
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from moviepy import VideoFileClip

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add the video-SALMONN-2 repo to sys.path so we can import qwenvl
# Set VIDEO_SALMONN_REPO to your video_SALMONN2_plus folder, or edit the path below
_repo_path = os.environ.get(
    "VIDEO_SALMONN_REPO",
    "/path/to/sense2decision/models/video-SALMONN-2/video_SALMONN2_plus",
)
if _repo_path and _repo_path not in sys.path:
    sys.path.insert(0, _repo_path)

# Transformers imports are always safe — keep them outside the try block
from transformers import AutoTokenizer, WhisperFeatureExtractor

# qwenvl imports may fail if the repo path isn't set up — let it raise loudly
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import DataArguments


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    return obj


def mean_attentions(attentions):
    """Mean across heads for all generation steps. Returns None if attentions
    are unavailable (flash-attn doesn't return them)."""
    if attentions is None:
        return None
    if all(layer is None for step in attentions for layer in step):
        return None
    return tuple(
        tuple(layer.mean(dim=1).cpu() if layer is not None else None for layer in step)
        for step in attentions
    )


@register_model("video_salmonn_2_full")
class VideoSALMONN2Full(lmms):
    """
    video-SALMONN-2+ FULL audio-visual model.
    Loads tsinghua-ee/video-SALMONN2_plus_3B_full or _7B_full.

    Uses the custom video_SALMONN2_plus model class with Whisper-based audio
    encoder + Q-Former, and the custom dataset preprocessing that interleaves
    video and audio tokens at temporal patch granularity.

    Built on Qwen2.5-VL-3B (36 layers, 2048 hidden) for the 3B variant.
    """

    def __init__(
        self,
        pretrained: str = "tsinghua-ee/video-SALMONN2_plus_3B_full",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 768,
        max_pixels: int = 61250,
        min_pixels: int = 3136,
        base_interval: float = 0.5,
        model_max_length: int = 131072,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

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

        # Load tokenizer with video-SALMONN-2+'s settings
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # Load with the custom video_SALMONN2_plus class
        # liger_kernel is REQUIRED — qwenvl imports it at module load time
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation


        print(f"Load  VideoSalmon 2 + model from {pretrained}")
        self._model = video_SALMONN2_plus.from_pretrained(pretrained, **model_kwargs).eval()
        self.attn_implementation = attn_implementation

        # Build data_args as in inference.py's prepare_dataset
        # dataset_use="" means we never load from disk; we call _get_item directly
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

        # Pre-compute special token IDs from config
        # audio_token_id: 151665 confirmed at top-level config
        cfg = self._model.config

        def _get_id(name, fallback_str):
            val = getattr(cfg, name, None)
            if val is not None:
                return val
            tid = self._tokenizer.convert_tokens_to_ids(fallback_str)
            return tid if tid != self._tokenizer.unk_token_id else None

        self.special_token_ids = {
            "im_start": _get_id("im_start_token_id", "<|im_start|>"),
            "im_end": _get_id("im_end_token_id", "<|im_end|>"),
            "vision_start": _get_id("vision_start_token_id", "<|vision_start|>"),
            "vision_end": _get_id("vision_end_token_id", "<|vision_end|>"),
            "video": _get_id("video_token_id", "<|video_pad|>"),
            "image": _get_id("image_token_id", "<|image_pad|>"),
            "audio": _get_id("audio_token_id", "<|audio_pad|>"),
            # video-SALMONN-2+ does NOT use separate audio_start/end markers
            # Audio tokens are placed inside <|vision_start|>...<|vision_end|>
            "audio_start": None,
            "audio_end": None,
        }
        print(f"[SPECIAL_TOKENS] {self.special_token_ids}")

        print(f"[SHARD] hf_device_map: {getattr(self._model, 'hf_device_map', 'NOT SET')}")
        print(f"[SHARD] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        for i in range(torch.cuda.device_count()):
            print(f"[SHARD] cuda:{i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB allocated")

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
        raise NotImplementedError("Loglikelihood is not implemented for VideoSALMONN2Full")

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

        # AV benchmarks usually return a list with a single video
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

    def _prepare_inputs(self, raw_inputs: dict) -> dict:
        """Strip non-tensor fields and move tensors to GPU. Mirrors inference.py prepare_inputs."""
        for key in ["video", "image", "prompt", "ref", "audio", "use_audio", "should_use"]:
            raw_inputs.pop(key, None)
        target_device = "cuda" if self.device_map == "auto" else self.device
        return {
            k: v.to(target_device) for k, v in raw_inputs.items() if isinstance(v, torch.Tensor)
        }

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # Hidden state logging mode
        global_idx = 0
        all_samples_data = []
        base_save_path = os.environ.get("LMM_EVAL_OUTPUT_DIR", None)
        total_save = int(os.environ.get("LMM_EVAL_TOTAL_SAVE", 100))
        save_step = int(os.environ.get("LMM_EVAL_SAVE_STEP", 25))
        save_type = os.environ.get("LMM_EVAL_SAVE_TYPE", "hidden")
        if base_save_path == "":
            base_save_path = None
        if base_save_path is not None:
            base_save_path = os.path.join(base_save_path, f"{save_type}_outputs")
            print(f"Saving intermediate outputs to {base_save_path}")

        def _collate(x):
            toks = self._tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None

                context = context + "\nOnly give the best option."

                input_dict = self._build_input_dict(context, visual)


                try:
                    raw_inputs = self.test_data._get_item(input_dict)

                except Exception as e:
                    eval_logger.error(f"Error in _get_item for doc_id={doc_id[i]}: {e}")
                    res.append("")
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                    pbar.update(1)
                    continue
        
                # AV SpeakerBench-specific prompt suffix
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
                # =====================================================


                ## debug the the whole input
                # all_ids = raw_inputs["input_ids"][0].tolist()
                # print(f"  [SUFFIX CHECK] total length: {len(all_ids)}")
                # print(f"  [SUFFIX CHECK] last 20 token ids: {all_ids[-20:]}")
                # print(f"  [SUFFIX CHECK] full decoded input:")
                # print(self._tokenizer.decode(all_ids))
                # print(f"  [SUFFIX CHECK] end of decode marker ^^^")
                    
                # exit()


                # print(f"\n[INPUT CHECK] doc_id={doc_id[i]}, use_audio={input_dict.get('use_audio', False)}")
                # print(f"  keys in raw_inputs: {list(raw_inputs.keys())}")
                # for k, v in raw_inputs.items():
                #     if isinstance(v, torch.Tensor):
                #         print(f"  {k}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                #     elif isinstance(v, list):
                #         print(f"  {k}: list len={len(v)}, first type={type(v[0]).__name__ if v else 'empty'}")
                #     else:
                #         print(f"  {k}: {type(v).__name__} = {v if not hasattr(v, '__len__') else f'len={len(v)}'}")


                # print(f"  doc_id: {doc_id[i]}")
                # print(f"  visual (video path): {visual}")
                # if "audio_feature" in raw_inputs and isinstance(raw_inputs["audio_feature"], torch.Tensor):
                #     af = raw_inputs["audio_feature"]
                #     print(f"  audio_feature shape: {tuple(af.shape)}")
                # if "video_grid_thw" in raw_inputs:
                #     thw = raw_inputs["video_grid_thw"]
                #     T, H, W = thw[0].tolist()
                #     print(f"  T (temporal patches) = {T}")
                #     print(f"  H (height patches)   = {H}")
                #     print(f"  W (width patches)    = {W}")
                #     print(f"  effective frames     = 2 * T = {2*T}")
                #     print(f"  effective resolution = {H*14} x {W*14} = {H*14*W*14} pixels/frame")
                # exit()

                # # DEBUG
                # ids = raw_inputs["input_ids"][0].tolist()
                # from collections import Counter
                # counts = Counter(ids)
                # print(f"\n[DEBUG] doc_id={doc_id[i]}, total tokens: {len(ids)}, use_audio: {input_dict.get('use_audio', False)}")
                # for tok_id, count in counts.most_common(10):
                #     print(f"  id={tok_id:6d}  count={count:5d}  {repr(self._tokenizer.decode([tok_id]))}")

                # # Interleave pattern: collapse consecutive runs of video/audio pad tokens
                # video_id = self.special_token_ids["video"]
                # audio_id = self.special_token_ids["audio"]
                # from itertools import groupby
                # runs = [(k, sum(1 for _ in g)) for k, g in groupby(ids) if k in (video_id, audio_id)]
                # pattern = " ".join(f"{'V' if k==video_id else 'A'}×{n}" for k, n in runs[:12])
                # print(f"  pattern (first 12 runs): {pattern}  | total runs: {len(runs)}")
                # exit() 

                # Save metadata BEFORE stripping non-tensor fields
                input_ids_for_save = raw_inputs["input_ids"][0].cpu().tolist() if "input_ids" in raw_inputs else []
                video_grid_thw_for_save = (
                    raw_inputs["video_grid_thw"].cpu().tolist()
                    if "video_grid_thw" in raw_inputs and isinstance(raw_inputs["video_grid_thw"], torch.Tensor)
                    else None
                )

                inputs = self._prepare_inputs(raw_inputs)

                if base_save_path is None:

                    
                    # print(self.model)
                    # exit()
                    # Normal generation path
                    try:
                        with torch.no_grad():
                            cont = self.model.generate(
                                **inputs,
                                eos_token_id=self._tokenizer.eos_token_id,
                                pad_token_id=self._tokenizer.pad_token_id,
                                do_sample=(gen_kwargs["temperature"] > 0),
                                temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
                                top_p=gen_kwargs["top_p"],
                                num_beams=gen_kwargs["num_beams"],
                                max_new_tokens=gen_kwargs["max_new_tokens"],
                                use_cache=self.use_cache,
                            )
                    except Exception as e:
                        eval_logger.error(f"Error {e} in generating")
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)
                        continue

                    trimmed = cont[0, inputs["input_ids"].shape[1]:]
                    answer = self._tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    res.append(answer)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                    pbar.update(1)

                else:
                    raise ValueError("Why here")
                    # Logging mode
                    if save_type == "positions":
                        global_idx += 1
                        sample_data = {
                            "metadata": {
                                "task": task,
                                "doc_id": doc_id[i],
                                "split": split,
                                "input_ids": input_ids_for_save,
                                "video_grid_thw": video_grid_thw_for_save,
                                "special_tokens": self.special_token_ids,
                            },
                            "model_outputs": {},
                        }
                        all_samples_data.append(sample_data)
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)

                    else:
                        # hidden or attention mode
                        want_attn = (save_type == "attention")
                        if "flash" in str(self.attn_implementation).lower():
                            if want_attn:
                                raise RuntimeError(
                                    "save_type=attention requires eager/sdpa, not flash_attention_2."
                                )
                            want_attn = False

                        with torch.no_grad():
                            cont = self.model.generate(
                                **inputs,
                                eos_token_id=self._tokenizer.eos_token_id,
                                pad_token_id=self._tokenizer.pad_token_id,
                                do_sample=(gen_kwargs["temperature"] > 0),
                                temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
                                top_p=gen_kwargs["top_p"],
                                num_beams=gen_kwargs["num_beams"],
                                max_new_tokens=gen_kwargs["max_new_tokens"],
                                use_cache=self.use_cache,
                                output_attentions=want_attn,
                                output_hidden_states=True,
                                return_dict_in_generate=True,
                            )

                        if want_attn and getattr(cont, "attentions", None) is not None:
                            cont.attentions = mean_attentions(cont.attentions)

                        safe_outputs = move_to_cpu(dict(cont))
                        safe_outputs.pop("past_key_values", None)
                        torch.cuda.empty_cache()

                        sequences = cont.sequences
                        trimmed = sequences[0, inputs["input_ids"].shape[1]:]
                        answer = self._tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                        global_idx += 1
                        output_to_save = {}
                        print(f"\n[LOGGING] Sample {global_idx} - Task: {task}, Doc ID: {doc_id[i]}")
                        if save_type == "attention":
                            output_to_save["attentions"] = safe_outputs.get("attentions")
                        elif save_type == "hidden":
                            output_to_save["hidden_states"] = safe_outputs.get("hidden_states")

                        sample_data = {
                            "metadata": {
                                "task": task,
                                "doc_id": doc_id[i],
                                "split": split,
                                "input_ids": input_ids_for_save,
                                "video_grid_thw": video_grid_thw_for_save,
                                "special_tokens": self.special_token_ids,
                            },
                            "model_outputs": output_to_save,
                        }
                        all_samples_data.append(sample_data)

                        res.append(answer)
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                        pbar.update(1)

                    # Periodic save
                    if len(all_samples_data) >= save_step:
                        os.makedirs(base_save_path, exist_ok=True)
                        save_file = os.path.join(base_save_path, f"infer_batch_upto_{global_idx}.pkl")
                        with open(save_file, "wb") as f:
                            pickle.dump(all_samples_data, f)
                        print(f"\n[SAVE] {len(all_samples_data)} samples to {save_file} (Total: {global_idx})")
                        all_samples_data.clear()

                    if global_idx == total_save:
                        if all_samples_data:
                            os.makedirs(base_save_path, exist_ok=True)
                            save_file = os.path.join(base_save_path, f"infer_batch_upto_{global_idx}.pkl")
                            with open(save_file, "wb") as f:
                                pickle.dump(all_samples_data, f)
                        exit()

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: multi-round")