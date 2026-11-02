import os
import pickle
from pydoc import doc
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
    # import qwen_omni_utils
    # print(f"qwen_omni_utils loaded from: {qwen_omni_utils.__file__}")
    # exit()
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")

# added for profiling
from lmms_eval.models.model_utils.profiling import PrefillTimingProcessor, log_generate_metrics
import time

# added for log the attention weights and hidden states
def move_to_cpu(obj):
    """Recursively moves any torch.Tensor inside a data structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    return obj  # Return strings, ints, bools, etc. untouched

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

@register_model("qwen2_5_omni_interleave")
class Qwen2_5_Omni_Interleave(lmms):
    """
    Qwen2.5-Omni-7B
    "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"

    For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on your running tasks.
    https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
    """

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
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        exp_config: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
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


        ################ added for experiment config
        if exp_config and os.path.exists(exp_config):
            with open(exp_config) as f:
                self.exp_config = yaml.safe_load(f)
            eval_logger.info(f"Experiment config: {self.exp_config}")
      
        else:
            self.exp_config = None

        method = self.exp_config.get("method") if self.exp_config else None

        if method == "tokendrop":
            from models.qwen2_5.tokendrop_modelling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
            print("###"*10)
            print("Using TokenDrop")
            print("Configuration:", self.exp_config)
            print("###"*10)
        else:
            from models.qwen2_5.modelling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
            print("###"*10)
            print("Using default method with no token drop")
            print("Configuration:", self.exp_config)
            print("###"*10)
        ###################

        Qwen2_5OmniForConditionalGeneration._tp_plan = [] if Qwen2_5OmniForConditionalGeneration._tp_plan is None else Qwen2_5OmniForConditionalGeneration._tp_plan
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation=attn_implementation).eval()

        ######### Added for experiment config
        if self.exp_config:
            self._model.thinker.exp_config = self.exp_config
        else:
            self._model.thinker.exp_config = None
        ##############
        self.processor = Qwen2_5OmniProcessor.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._model.disable_talker()
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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


    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        """
        Resample the audio to the target sample rate.
        """
        if current_sample_rate != 16000:  # The sample rate for Qwen2.5-Omni is 16kHz
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False  # Flag to check whether we are using video or not


        global_idx = 0 # count how many samples we have processed in total, used for saving intermediate outputs
        all_samples_data = [] # Initialize a master list here for saving input and output data for all samples in the batch
        base_save_path = os.environ.get("LMM_EVAL_OUTPUT_DIR", None)
        total_save = int(os.environ.get("LMM_EVAL_TOTAL_SAVE", 100)) # you can set this environment variable to control how many total samples to save, default is 100 
        save_step = int(os.environ.get("LMM_EVAL_SAVE_STEP", 25)) # you can set this environment variable to control how often to save intermediate outputs, default is every 10 samples
        save_type = os.environ.get("LMM_EVAL_SAVE_TYPE", "attention") # you can set this environment variable to control how to save intermediate outputs, "batch" means saving by batch, "iter" means saving by iteration (every sample)
        if base_save_path is "":
            base_save_path = None
        else:
            base_save_path = os.path.join(base_save_path, f"{save_type}_outputs")
            print(f"Saving intermediate outputs to {base_save_path}")

            # Mapping save_type to the actual keys returned by the model
            key_map = {
                "attention": "attentions",
                "hidden": "hidden_states",
                "logits_len": "logits_len",
                "positions" : "positions",
            }

            # Select the target data from the model output (cont)
            selection_key = key_map.get(save_type)



        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            # Get the duration metadata
            # task might still be a tuple
          
 
            

            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
    
            # visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on tasks.
            # https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
            message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

            AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
            IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


  
            # for i, context in enumerate(contexts):
            #     if len(visuals) > 0:
            #         visual = visuals[i] if i < len(visuals) else None
            #         print(visual)
            #         # exit()
            #         if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
            #             current_use_audio = self._check_if_video_has_audio(visual)
            #             if self.use_custom_video_loader:
            #                 frames = read_video(visual, num_frm=self.max_num_frames, fps=self.fps)
            #                 image_contents = []
            #                 for frame in frames:
            #                     img = Image.fromarray(frame)
            #                     b64 = encode_image_to_base64(img, image_format="JPEG", convert_rgb=True, quality=85)
            #                     image_contents.append(f"data:image/jpeg;base64,{b64}")

                    
            #                 message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
            #             else:  # Model video loader
            #                 message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})
             

            #         elif isinstance(visual, Image.Image):  # Single image
            #             raise ValueError("Why here")
            #             message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

            #         elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
            #             raise ValueError("Why here")
            #             single_message = {"role": "user", "content": []}
            #             for v in visual:
            #                 single_message["content"].append({"type": "image", "image": v})
            #             single_message["content"].append({"type": "text", "text": context})
            #             message.append(single_message)


            #         # Single audio (dict OR AudioDecoder)
            #         elif 'AudioDecoder' in type(visual).__name__:
            #             current_use_audio = True
            #             decoded = visual.get_all_samples()
            #             samples = decoded.data.numpy().flatten()
            #             sr = decoded.sample_rate
            #             visual = {"array": samples, "sampling_rate": sr}
            #             audio = self.resample_audio(visual["array"], visual["sampling_rate"])
            #             audio_splits = split_audio(audio, 4800000)
            #             single_message = {"role": "user", "content": []}
            #             for j in range(len(audio_splits)):
            #                 single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
            #             single_message["content"].append({"type": "text", "text": context})
            #             message.append(single_message)

            #         # Multiple audios (list of dict OR list of AudioDecoder)
            #         elif isinstance(visual, (list, tuple)) and len(visual) > 0 and 'AudioDecoder' in type(visual[0]).__name__:
            #             current_use_audio = True
            #             for i, v in enumerate(visual):
            #                 decoded = v.get_all_samples()
            #                 samples = decoded.data.numpy().flatten()
            #                 sr = decoded.sample_rate
            #                 v = {"array": samples, "sampling_rate": sr}
            #                 audio = self.resample_audio(v["array"], v["sampling_rate"])
            #                 audio_splits = split_audio(audio, 4800000)
            #                 single_message = {"role": "user", "content": []}
            #                 for j in range(len(audio_splits)):
            #                     single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
            #                 single_message["content"].append({"type": "text", "text": context})
            #                 message.append(single_message)


            #         # Fixed code for audio messages
            #         elif isinstance(visual, dict):  # Single audio
            #             # raise ValueError("Why here")
            #             current_use_audio = True
            #             audio = self.resample_audio(visual["array"], visual["sampling_rate"])
            #             audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
            #             single_message = {"role": "user", "content": []}
            #             for i in range(len(audio_splits)):
            #                 single_message["content"].append({"type": "audio", "audio": audio_splits[i]})
            #             single_message["content"].append({"type": "text", "text": context})
            #             message.append(single_message)

            #         elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
            #             # raise ValueError("Why here")
            #             current_use_audio = True
            #             for i, v in enumerate(visual):
            #                 audio = self.resample_audio(v["array"], v["sampling_rate"])
            #                 audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
            #                 single_message = {"role": "user", "content": []}
            #                 for j in range(len(audio_splits)):
            #                     single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
            #                 single_message["content"].append({"type": "text", "text": context})
            #                 message.append(single_message)

            #         else:
            #             raise ValueError(f"Unknown visual type: {type(visual)}")

            import re

            for i, context in enumerate(contexts):
                doc_visuals = visuals[i] if i < len(visuals) else []
                if not isinstance(doc_visuals, (list, tuple)):
                    doc_visuals = [doc_visuals]

                # Pre-process each media item into the content blocks it produces
                # (audio gets split into chunks, so one media item → multiple content blocks)
                media_blocks_per_item = []
                for v in doc_visuals:
                    v_lower = v.lower()
                    if v_lower.endswith(VIDEO_EXTS):
                        media_blocks_per_item.append([{"type": "video", "video": v}])
                    elif v_lower.endswith(AUDIO_EXTS):
                        samples, sr = librosa.load(v, sr=None, mono=True)
                        samples = samples.astype(np.float32)
                        audio = self.resample_audio(samples, sr)
                        chunks = [{"type": "audio", "audio": c} for c in split_audio(audio, 4800000)]
                        media_blocks_per_item.append(chunks)
                    elif v_lower.endswith(IMAGE_EXTS):
                        img = Image.open(v).convert("RGB")
                        # print(f"Original image size: {img.size}")
                        media_blocks_per_item.append([{"type": "image", "image": img}])
                    else:
                        raise ValueError(f"Unknown file extension: {v}")

   
                # Split the question text at <media_N> markers and interleave
                # Pattern captures the index inside the tag
                parts = re.split(r"<media_(\d+)>", context)
                # re.split with a capturing group gives: [text, idx, text, idx, text, ...]

                content = []
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        # Plain text segment (could be empty if marker is at the start/end)
                        if part:
                            content.append({"type": "text", "text": part})
                    else:
                        # Media index (string), convert to int and look up
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

            # print("&&&&&&"*10)
            # print(text)

            # print("------"*10)
            # exit()


            
            audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)

            # print("Processed multimedia info:")


            # print("Audios:", type(audios) ,len(audios), audios[0].shape)
            # print("Images:", type(images), len(images), (images[0].size) if len(images) > 0 else None)
            # # print("Videos:", type(videos), videos[0].shape)
            # print("Text:", text )
            

            # print(f"Chunk: {self.task_dict[task][split][doc_id[0]].get('question_id', '')}")
    
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=current_use_audio)


            # print(inputs["pixel_values"].shape)

            # exit()

            # input_ids = inputs["input_ids"][0]
            # video_token_id = self.model.config.thinker_config.video_token_id
            # audio_token_id = self.model.config.thinker_config.audio_token_id
            # image_token_id = self.model.config.thinker_config.image_token_id

            # print(f"Total seq len: {len(input_ids)}")
            # print(f"Video tokens: {(input_ids == video_token_id).sum().item()}")
            # print(f"Audio tokens: {(input_ids == audio_token_id).sum().item()}")
            # print(f"Image tokens: {(input_ids == image_token_id).sum().item()}")
            # print(f"Text tokens: {len(input_ids) - (input_ids == video_token_id).sum().item() - (input_ids == audio_token_id).sum().item() - (input_ids == image_token_id).sum().item()}")

            # # if "video_grid_thw" in inputs:
            # #     print(f"video_grid_thw: {inputs['video_grid_thw']}")
            # #     expected = inputs['video_grid_thw'].prod() // 4  # spatial_merge 2x2
            # #     print(f"Expected video tokens: {expected.item()}")

            # input_ids = inputs["input_ids"][0]
            # tokenizer = self.tokenizer

            # print(" | ".join(f"{self.tokenizer.decode([tok_id])}" for tok_id in input_ids.tolist()))

            # print("*******************************")
            # from collections import Counter

            # input_ids = inputs["input_ids"][0].tolist()
            # counts = Counter(input_ids)

            # for tok_id, count in counts.most_common():
            #     tok_str = self.tokenizer.decode([tok_id])
            #     print(f"id={tok_id:6d}  count={count:5d}  {repr(tok_str)}")
            # exit()
    


            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            # print(inputs.keys())  
            # print(inputs.input_ids.shape)
            # print(inputs.attention_mask.shape)
            # print(inputs.pixel_values.shape)
            # print(inputs.video_grid_thw)
            
            # exit()


            if base_save_path is None:


                # Attach token positions for tokendrop method
                method = self.exp_config.get("method") if self.exp_config else None
                if method == "tokendrop":
                    from utils.knockout.configs_interleaved import (
                        extract_token_positions_interleaved,
                        decompose_sample,
                        audit_sample_options,
                    )
                    import re as _re

                    tok = lambda s: self.tokenizer.convert_tokens_to_ids(s)
                    special_tokens = {
                        "im_start":     tok("<|im_start|>"),
                        "im_end":       tok("<|im_end|>"),
                        "vision_start": tok("<|vision_bos|>"),
                        "vision_end":   tok("<|vision_eos|>"),
                        "audio_start":  tok("<|audio_bos|>"),
                        "audio_end":    tok("<|audio_eos|>"),
                        "video":        tok("<|VIDEO|>"),
                        "audio":        tok("<|AUDIO|>"),
                        "image":        tok("<|IMAGE|>"),
                    }

                    input_ids_list = inputs["input_ids"][0].cpu().tolist()
                    doc = self.task_dict[task][split][doc_id[0]]

                    if task == "av_odyssey":
                        # ---- Build question_with_options with letter prefixes ----
                        options = doc["options"]
                        options_with_letters = [
                            f"{chr(ord('A') + i)}: {opt}" for i, opt in enumerate(options)
                        ]
                        question_with_options = doc["question"] + "\n" + "\n".join(options_with_letters)

                        # ---- Replace [audioN]/[imgN]/[videoN] -> <media_N> ----
                        media_idx = [0]
                        def _replace_placeholder(m):
                            idx = media_idx[0]
                            media_idx[0] += 1
                            return f"<media_{idx}>"
                        question_with_options = _re.sub(
                            r"\[(audio|video|img)\d+\]",
                            _replace_placeholder,
                            question_with_options,
                        )

                        # ---- Replace <media_N> -> special token strings ----
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
                            question_with_options = question_with_options.replace(
                                f"<media_{idx}>", replacement
                            )

                        # ---- Decompose + audit ----
                        decomp = decompose_sample(doc["question"])
                        audit = audit_sample_options(doc)

                        # ---- Extract token positions for interleaved layout ----
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
                    else:
                        raise ValueError(
                            f"tokendrop not supported for task '{task}' in interleaved wrapper. "
                            f"Only 'av_odyssey' is currently supported."
                        )

                    # Hand positions to the modified modelling file via the thinker
                    self._model.thinker.model._tokendrop_positions = token_positions
                    self._model.thinker.model._tokendrop_schedule = self.exp_config["drop_schedule"]

                    # print("=" * 60)
                    # print("TokenPositionsInterleaved fields:")
                    # for field_name in vars(token_positions):
                    #     value = getattr(token_positions, field_name)
                    #     if value is None:
                    #         print(f"  {field_name:25s} = None")
                    #     elif isinstance(value, list):
                    #         if len(value) > 0 and isinstance(value[0], (tuple, list)) and len(value[0]) == 2:
                    #             total_tokens = sum(int(e) - int(s) + 1 for s, e in value)
                    #             print(f"  {field_name:25s} = {len(value)} ranges, {total_tokens} tokens total | {value}")
                    #         else:
                    #             print(f"  {field_name:25s} = {len(value)} items | {value}")
                    #     elif isinstance(value, tuple) and len(value) == 2:
                    #         n_tokens = int(value[1]) - int(value[0]) + 1
                    #         print(f"  {field_name:25s} = range, {n_tokens} tokens | {value}")
                    #     elif isinstance(value, int):
                    #         print(f"  {field_name:25s} = single position | {value}")
                    #     elif isinstance(value, str):
                    #         print(f"  {field_name:25s} = str | {value!r}")
                    #     else:
                    #         print(f"  {field_name:25s} = {type(value).__name__} | {value}")
                    # print("=" * 60)
                    # exit()
                ##### Added for profiling #####
                num_input_tokens = inputs.input_ids.shape[-1]
                timing_hook = PrefillTimingProcessor()
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()
                gen_start = time.perf_counter()
                try:
                    cont = self.model.generate(
                            **inputs,
                            return_audio=False,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=pad_token_id,
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                            use_audio_in_video=current_use_audio,
                            thinker_do_sample=False,
                            logits_processor=[timing_hook],
                        )

                 
                except Exception as e:
                    eval_logger.error(f"Error {e} in generating")
                    answer = ""
                    res.append(answer)
                    pbar.update(1)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                    continue
                # exit()
                ##### Added for profiling #####
                torch.cuda.synchronize()
                gen_end = time.perf_counter() 
                ##### End for profiling #####   

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]

                ### record generation metrics
                num_output_tokens = sum(len(ids) for ids in generated_ids_trimmed)
                log_generate_metrics(
                                gen_start=gen_start,
                                gen_end=gen_end,
                                timing_hook=timing_hook,
                                num_input_tokens=num_input_tokens,
                                num_output_tokens=num_output_tokens,
                                device=self.device,
                            )  
                ### end record generation metrics #####   

                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i, ans in enumerate(answers):
                    answers[i] = ans
                content = []
                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    content.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)
            
            else:
                raise ValueError("Why here")
                if save_type == "positions":
            
                    # ============================================
                    # Position-only logging — NO model computation
                    # ============================================
                    for i in range(len(doc_id)):
                        global_idx += 1

                        tok = lambda s: self.tokenizer.convert_tokens_to_ids(s)
                        input_ids_list = inputs["input_ids"][0].cpu().tolist()

                        sample_data = {
                            "metadata": {
                                "task": task,
                                "doc_id": doc_id[i],
                                "split": split,
                                "input_ids": input_ids_list,
                                "video_grid_thw": inputs["video_grid_thw"].cpu().tolist() if "video_grid_thw" in inputs else None,
                                "special_tokens": {
                                    "im_start": tok("<|im_start|>"),
                                    "im_end": tok("<|im_end|>"),
                                    "vision_start": tok("<|vision_bos|>"),
                                    "vision_end": tok("<|vision_eos|>"),
                                    "audio_start": tok("<|audio_bos|>"),
                                    "audio_end": tok("<|audio_eos|>"),
                                    "video": tok("<|VIDEO|>"),
                                    "audio": tok("<|AUDIO|>"),
                                    "image": tok("<|IMAGE|>"),
                                },
                            },
                            "model_outputs": {},
                        }

                        all_samples_data.append(sample_data)

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

                    # Skip generation — return dummy answers
                    for context in contexts:
                        res.append("")
                        self.cache_hook.add_partial("generate_until", (context, gen_kwargs), "")
                        pbar.update(1)
                else:
          
                    cont = self.model.generate(
                            **inputs,
                            return_audio=False,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=pad_token_id,
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                            use_audio_in_video=current_use_audio,
                            thinker_do_sample=False,
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict_in_generate = True,
                        )
                        
                    cont.attentions = mean_attentions(cont.attentions)  # Reduce attention tensors to save space
                    safe_outputs = move_to_cpu(dict(cont))
                    safe_outputs.pop("past_key_values", None)
                    torch.cuda.empty_cache() # Optional: only if you are tight on VRAM

                    for i in range(len(doc_id)):
                        global_idx += 1
                        output_to_save = {}

                        if save_type == "attention":
                            output_to_save["attentions"] = safe_outputs["attentions"]
                        elif save_type == "hidden":
                            output_to_save["hidden_states"] = safe_outputs["hidden_states"]

                    
                        tok = lambda s: self.tokenizer.convert_tokens_to_ids(s)

                        im_start_id     = tok("<|im_start|>")
                        im_end_id       = tok("<|im_end|>")
                        vision_start_id = tok("<|vision_bos|>")
                        vision_end_id   = tok("<|vision_eos|>")
                        audio_start_id  = tok("<|audio_bos|>")
                        audio_end_id    = tok("<|audio_eos|>")
                        video_token_id  = tok("<|VIDEO|>")
                        audio_token_id  = tok("<|AUDIO|>")
                        image_token_id  = tok("<|IMAGE|>")


                        sample_data = {
                            "metadata": {
                                "task": task,
                                "doc_id": doc_id[i],
                                "split": split,
                                "input_ids": inputs["input_ids"][0].cpu().tolist(),
                                "video_grid_thw": inputs["video_grid_thw"].cpu().tolist() if "video_grid_thw" in inputs else None,
                                "special_tokens": {
                                    "im_start": im_start_id,
                                    "im_end": im_end_id,
                                    "vision_start": vision_start_id,
                                    "vision_end": vision_end_id,
                                    "audio_start": audio_start_id,
                                    "audio_end": audio_end_id,
                                    "video": video_token_id,
                                    "audio": audio_token_id,
                                    "image": image_token_id,
                                },
                            },
                            "model_outputs": output_to_save,
                        }
        
                        all_samples_data.append(sample_data)

                        # Add this batch's data to our master list
                        # all_samples_data.append(sample_data)
                        # # Add this batch's data to our master list
                        # all_samples_data.append(sample_data)

                        if len(all_samples_data) >= save_step:
                            save_path = base_save_path # os.path.join(base_save_path) #, f"{task}")
                            os.makedirs(save_path, exist_ok=True)
                            save_path = os.path.join(save_path, f"infer_batch_upto_{global_idx}.pkl")
                            with open(save_path, "wb") as f:
                                pickle.dump(all_samples_data, f)
                            
                            print(f"\n[SAVE] Exported {len(all_samples_data)} samples to {save_path} (Total: {global_idx})")
                  
                            all_samples_data.clear() # Completely empty the list to free RAM
                        if  global_idx == total_save:
                            exit()

                # print(doc_id, task, split)
                # print(cont.keys())
                # print(f"sequences shape: {cont.sequences.shape}")
                # print(f"num generation steps: {len(cont.attentions)}")
                # print(f"layers per step: {len(cont.attentions[0])}")
                # print(f"prefill attn shape: {cont.attentions[0][0].shape}")
                # # should be [batch, heads, seq_len, seq_len] for prefill
                # # or [batch, heads, 1, seq_len] for decoding steps
                # exit()
            # exit()
                
                        

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
