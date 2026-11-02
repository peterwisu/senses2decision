<div align="center">

  # 👁️ 👂 From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs [NeurIPS 2026]

  [![Conference](https://img.shields.io/badge/NeurIPS-2026-blue)](https://neurips.cc/virtual/2026/loc/sydney/poster/153955)
  [![arXiv](https://img.shields.io/badge/arXiv-2606.10147-b31b1b.svg)](https://arxiv.org/abs/2606.10147)
  [![Project Page](https://img.shields.io/badge/Project-Page-brightgreen)](https://peterwisu.github.io/projects/senses-to-decisions/)

  📄 [**Paper**](https://arxiv.org/abs/2606.10147) &nbsp;|&nbsp; 🌐 [**Project Page**](https://peterwisu.github.io/projects/senses-to-decisions/) &nbsp;|&nbsp; 🔗 [**NeurIPS 2026 Poster**](https://neurips.cc/virtual/2026/loc/sydney/poster/153955)
</div>

Official implementation of **From Senses to Decisions: The Information Flow of Auditory
and Visual Perception in Multimodal LLMs** (NeurIPS 2026). We trace how Audio-Visual
Large Language Models (AVLLMs) route and integrate audio and visual information, using
**attention knockout** across two input configurations: audio-visual video and multiple
interleaved audio-visual items. We also show that audio, visual and other tokens can be
**discarded** once their information has been transferred, with minimal impact on
predictions.

## 📋 Table of Contents

- [📦 Repository Structure](#-repository-structure)
- [🛠️ Installation](#️-installation)
- [🚀 Experiments](#-experiments)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 Citation](#-citation)

## 📦 Repository Structure

```
.
├── apptainer/                  # Container definitions
├── config/                     # Experiment configs (baseline and token discarding)
├── lmms-eval/                  # lmms-eval with our model wrappers and task filters
├── models/
│   ├── qwen2_5/                # Qwen2.5-Omni, plus the token-discarding variant
│   └── video-SALMONN-2/        # Video-SALMONN2 Plus
├── qwen_omni_utils/            # Video and audio preprocessing for Qwen2.5-Omni
├── scripts/
│   ├── knockout/               # Attention knockout
│   └── drop_token/             # Token discarding (Table 4)
└── utils/knockout/             # Knockout configs, masks and attention hooks
```

## 🛠️ Installation

We use [Apptainer](https://apptainer.org/). Qwen2.5-Omni and Video-SALMONN2 Plus need
separate containers, so build one for each:

```bash
apptainer build qwen_omni.sif apptainer/qwen_omni_flash.def
apptainer build vdo_salmon.sif apptainer/vdo_salmon.def
```

Run every script inside the matching container, from the repository root:

```bash
apptainer exec --nv qwen_omni.sif bash scripts/knockout/av_speakerbench/run_baseline.sh
apptainer exec --nv vdo_salmon.sif bash scripts/knockout/salmonn_av_speakerbench/run_baseline.sh
```

To use your own environment instead, install the package versions listed in
[apptainer/qwen_omni_flash.def](apptainer/qwen_omni_flash.def) and
[apptainer/vdo_salmon.def](apptainer/vdo_salmon.def).

Model weights and datasets are downloaded from Hugging Face on the first run.

## 🚀 Experiments

| Experiment | Model | Dataset | Scripts |
|------------|-------|---------|---------|
| Attention knockout | Qwen2.5-Omni | AV-SpeakerBench | [scripts/knockout/av_speakerbench](scripts/knockout/av_speakerbench) |
| Attention knockout | Qwen2.5-Omni | AV-Odyssey | [scripts/knockout/av_odyssey](scripts/knockout/av_odyssey) |
| Attention knockout | Qwen2.5-Omni | WorldSense | [scripts/knockout/worldsense](scripts/knockout/worldsense) |
| Attention knockout | Video-SALMONN2 Plus | AV-SpeakerBench | [scripts/knockout/salmonn_av_speakerbench](scripts/knockout/salmonn_av_speakerbench) |
| Attention knockout | Video-SALMONN2 Plus | WorldSense | [scripts/knockout/salmonn_worldsense](scripts/knockout/salmonn_worldsense) |
| Token discarding | Qwen2.5-Omni | AV-SpeakerBench | [scripts/drop_token/av_speakerbench](scripts/drop_token/av_speakerbench) |
| Token discarding | Qwen2.5-Omni | WorldSense | [scripts/drop_token/worldsense](scripts/drop_token/worldsense) |
| Token discarding | Qwen2.5-Omni | AV-Odyssey | [scripts/drop_token/av_odyssey](scripts/drop_token/av_odyssey) |

### Attention knockout

Run the baseline first, then a knockout config. Each run sweeps layer windows of size
`k = 7` across centers 0-35. The commands below show the scripts; run them inside the
container as in [Installation](#️-installation).

```bash
bash scripts/knockout/av_speakerbench/run_baseline.sh
bash scripts/knockout/av_speakerbench/run_knockout.sh 1         # config 1, all centers
bash scripts/knockout/av_speakerbench/run_knockout.sh 1 0-11    # config 1, centers 0-11
```

Knockout configs are defined in [utils/knockout/configs.py](utils/knockout/configs.py)
and, for AV-Odyssey, [utils/knockout/configs_interleaved.py](utils/knockout/configs_interleaved.py).
To change the window size, edit `WINDOW_K` at the top of `run_knockout.sh`.

### Token discarding

```bash
bash scripts/drop_token/av_speakerbench/run_baseline.sh
bash scripts/drop_token/av_speakerbench/run_drop.sh                     # discard all
bash scripts/drop_token/av_speakerbench/run_drop.sh tokendrop_video26   # discard one type
```

| Dataset | Discard | Layer | Method |
|---------|---------|-------|--------|
| AV-SpeakerBench, WorldSense | Video | 26 | `tokendrop_video26` |
| | Audio | 26 | `tokendrop_audio26` |
| | Question | 29 | `tokendrop_question29` |
| | All | 26, 29 | `tokendrop_video26_audio26_question29` |
| AV-Odyssey | Candidates | 25 | `tokendrop_targets25` |
| | Reference | 31 | `tokendrop_query31` |
| | Question | 29 | `tokendrop_question29` |
| | All | 25, 29, 31 | `tokendrop_targets25_query31_question_29` |

## 🙏 Acknowledgements

Our code is built on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), with
models from [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) and
[video-SALMONN-2](https://github.com/bytedance/video-SALMONN-2). We thank the authors
for releasing their code.

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{suharitdamrong2026senses,
  title={From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs},
  author={Suharitdamrong, Wish and Awais, Muhammad and Zhu, Xiatian and Atito, Sara},
  journal={arXiv preprint arXiv:2606.10147},
  year={2026}
}
```
