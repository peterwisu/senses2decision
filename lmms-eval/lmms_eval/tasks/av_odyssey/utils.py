import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

dir_name = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# AV-Odyssey Benchmark — Task & Input Layout Reference
# ============================================================================
#
# The dataset has 4555 samples across 26 tasks, grouped into 7 subfields and
# 5 logical input layouts (7 distinct data_type strings, 2 of which are typos
# of the others).
#
# Columns per sample: question_id, question_type_id, data_type, subfield,
#   question, options, answer, image_path, video_path, audio_path
#
# ----------------------------------------------------------------------------
# All 26 tasks (by question_type_id)
# ----------------------------------------------------------------------------
#  ID  Task Name                      Subfield        n    Input Layout(s)
#   1  Instrument Recognition         Timbre          200  multi-img+1-aud (98) + 1-img+multi-aud (102)
#   2  Singer Recognition             Timbre          200  multi-img+1-aud
#   3  Gunshot Recognition            Timbre          200  multi-img+1-aud (100) + 1-img+multi-aud (100)
#   4  Bird Recognition               Timbre          200  multi-img+1-aud (102) + 1-img+multi-aud (98)
#   5  Animal Recognition             Timbre          200  multi-img+1-aud (100) + 1-img+multi-aud (100)
#   6  Transportation Recognition     Timbre          200  multi-img+1-aud (105) + 1-img+multi-aud (95)
#   7  Material Recognition           Timbre          200  multi-img+1-aud
#   8  Scene Recognition              Timbre          200  multi-img+1-aud (103) + 1-img+multi-aud (97)
#   9  Hazard Recognition             Timbre          108  multi-img+1-aud
#  10  Action Recognition             Timbre          196  multi-img+1-aud
#  11  Eating Sound Recognition       Timbre          200  multi-img+1-aud (98) + 1-img+multi-aud (102)
#  12  Speech Sentiment Analysis      Tone            200  multi-img+1-aud (100) + 1-img+multi-aud (100, typo)
#  13  Meme Understanding             Tone             20  single video + 1-aud                           [VIDEO]
#  14  Music Sentiment Analysis       Melody          197  multi-img+1-aud
#  15  Music Genre Classification     Melody          200  multi-img+1-aud (103) + 1-img+multi-aud (97, typo)
#  16  Dance and Music Matching       Melody          200  single video + multi-aud                       [VIDEO]
#  17  Film and Music Matching        Melody          200  multi-img+1-aud
#  18  Music Score Matching           Melody          200  multi-img+1-aud
#  19  Audio 3D Angle Estimation      Space            20  1-img + 1-aud
#  20  Audio Distance Estimation      Space            20  1-img + 1-aud
#  21  Audio Time Estimation          Time            200  single video + 1-aud                           [VIDEO]
#  22  Audio-Visual Synchronization   Time            200  single video + multi-aud                       [VIDEO]
#  23  Action Sequencing              Time            200  multi-img+1-aud (dataset has trailing-comma typo)
#  24  Hallucination Evaluation       Hallucination   200  multi-img+1-aud
#  25  Action Prediction              Intricacy       199  1-img + 1-aud
#  26  Action Tracing                 Intricacy       195  1-img + 1-aud
# ----------------------------------------------------------------------------
# Distinct data_type strings in the dataset (note typos)
# ----------------------------------------------------------------------------
#   "multi images, single audio"          2410   main multi-image + 1-audio layout
#   "multi images, single audio,"          200   task 23 only (trailing-comma typo)
#   "single image, multi audios"           694   main 1-image + multi-audio layout
#   "single images, multi audios"          197   tasks 12, 15 (plural "images" typo)
#   "single image, single audio"           434   tasks 19, 20, 25, 26
#   "single video, single audio"           220   tasks 13, 21
#   "single video, multi audios"           400   tasks 16, 22
#
# Logical layouts (collapsing typos):
#   multi-img + 1-aud       2610 samples, tasks: many
#   1-img + multi-aud        891 samples, tasks: many
#   1-img + 1-aud            434 samples, tasks: 19, 20, 25, 26
#   1-video + 1-aud          220 samples, tasks: 13, 21
#   1-video + multi-aud      400 samples, tasks: 16, 22
#
# Video-containing tasks: {13, 16, 21, 22} — 620 samples total.
# Dropping them leaves 3935 samples across 22 tasks, 6 subfields.
# ============================================================================

TASK_ID_TO_NAME = {
    1:  "Instrument Recognition",
    2:  "Singer Recognition",
    3:  "Gunshot Recognition",
    4:  "Bird Recognition",
    5:  "Animal Recognition",
    6:  "Transportation Recognition",
    7:  "Material Recognition",
    8:  "Scene Recognition",
    9:  "Hazard Recognition",
    10: "Action Recognition",
    11: "Eating Sound Recognition",
    12: "Speech Sentiment Analysis",
    13: "Meme Understanding",
    14: "Music Sentiment Analysis",
    15: "Music Genre Classification",
    16: "Dance and Music Matching",
    17: "Film and Music Matching",
    18: "Music Score Matching",
    19: "Audio 3D Angle Estimation",
    20: "Audio Distance Estimation",
    21: "Audio Time Estimation",
    22: "Audio-Visual Synchronization",
    23: "Action Sequencing",
    24: "Hallucination Evaluation",
    25: "Action Prediction",
    26: "Action Tracing",
}


def _ci95(scores):
    """95% confidence interval half-width (in percentage points) for a
    binary accuracy metric, computed via normal approximation."""
    n = len(scores)
    if n == 0:
        return 0.0
    p = sum(scores) / n
    return 1.96 * math.sqrt(p * (1 - p) / n) * 100.0


INCLUDED_TASK_IDS = {
    1,    # Instrument Recognition         Timbre   200  MIXED
    4,    # Bird Recognition               Timbre   200  MIXED
    5,    # Animal Recognition             Timbre   200  MIXED
    6,    # Transportation Recognition     Timbre   200  MIXED
    7,    # Material Recognition           Timbre   200  multi-img+1-aud
    8,    # Scene Recognition              Timbre   200  MIXED
    9,    # Hazard Recognition             Timbre   108  multi-img+1-aud
    10,   # Action Recognition             Timbre   196  multi-img+1-aud
    12,   # Speech Sentiment Analysis      Tone     200  MIXED
    15,   # Music Genre Classification     Melody   200  MIXED
    17,   # Film and Music Matching        Melody   200  multi-img+1-aud

}
"""
Aggregate across 11 selected tasks · 2104 samples total
Input layouts (total)

multi images, single audio	1415 (67%)
single image, multi audios	492 (23%)
single images, multi audios	197 (9%)
Detected structures (total) (4 unique structures across 412 templates)

targets → question → query	1639 (78%)
question → query → targets	323 (15%)
query → targets → question	124 (6%)
query → question → targets	18 (1%)

"""
def filter_av_odyssey(dataset):
    """Keep only curated task IDs for evaluation.

    Includes 23 of 26 tasks. Excludes:
      - Task 13 (Meme Understanding,        20 samples, Tone)
      - Task 19 (Audio 3D Angle Estimation, 20 samples, Space)
      - Task 20 (Audio Distance Estimation, 20 samples, Space)
    All excluded for having too few samples for reliable per-task accuracy.

    Result: 4495 samples across 23 tasks covering 6 subfields
    (Timbre, Tone, Melody, Time, Hallucination, Intricacy).
    Space subfield is removed entirely.
    """
    return dataset.filter(lambda x: x["question_type_id"] in INCLUDED_TASK_IDS)

eval_type_dict = {
    "Subfield": [
        "Timbre",
        "Tone",
        "Melody",
        "Space",
        "Time",
        "Hallucination",
        "Intricacy",
    ],
}

with open(Path(__file__).parent / "av_odyssey.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

question_prompt = "Answer with the option's letter from the given choices directly."


def split_media_tags(content):
    pattern = r"\[(audio|video|img)(\d+)\]"

    matches = list(re.finditer(pattern, content))
    if not matches:
        return [content]

    result = []
    last_end = 0

    for match in matches:
        if match.start() > last_end:
            result.append(content[last_end : match.start()])

        media_type = match.group(1)
        media_num = int(match.group(2))
        result.append((media_type, media_num))

        last_end = match.end()

    if last_end < len(content):
        result.append(content[last_end:])

    return result


def av_odyssey_doc_to_visual(doc):
    audio_data = []
    image_data = []
    video_data = []
    result = []

    # image 类型数据
    if "image" in doc["data_type"]:
        for relative_path in doc["image_path"]:
            abs_path = os.path.join(cache_dir, relative_path)
            if os.path.exists(abs_path):
                image_data.append(abs_path)
            else:
                print(f"Image path does not exist: {abs_path}")

    # video 类型数据
    elif "video" in doc["data_type"]:
        for relative_path in doc["video_path"]:
            abs_path = os.path.join(cache_dir, relative_path)
            if os.path.exists(abs_path):
                video_data.append(abs_path)
            else:
                print(f"Video path does not exist: {abs_path}")

    # audio 类型数据
    for relative_path in doc["audio_path"]:
        abs_path = os.path.join(cache_dir, relative_path)
        if os.path.exists(abs_path):
            audio_data.append(abs_path)
        else:
            print(f"Audio path does not exist: {abs_path}")

    question = get_text(doc)
    for q in question:
        if isinstance(q, str):
            continue
        else:
            media_type, media_num = q
            media_num = media_num - 1
            if media_type == "audio":
                result.append(audio_data[media_num])
            elif media_type == "video":
                result.append(video_data[media_num])
            elif media_type == "img":
                result.append(image_data[media_num])

    return result


def get_text(doc):
    question = doc["question"]
    options = doc["options"]
    option_text = options[0] + "\n" + options[1] + "\n" + options[2] + "\n" + options[3] + "\n"
    text = question + "\n" + option_text + question_prompt
    return split_media_tags(text)


def av_odyssey_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    text = get_text(doc)
    id = 0
    result = []
    for t in text:
        if isinstance(t, str):
            result.append(t)
        else:
            result.append(f"<media_{id}>")
            id += 1
    return "".join(result)


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"{choice}" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = "A"
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def av_odyssey_process_results(doc, results):
    """Score one sample. Returns subfield, task_id, data_type for aggregation."""
    pred = results[0]
    options = doc["options"]
    option_list = {"A": options[0][3:], "B": options[1][3:], "C": options[2][3:], "D": options[3][3:]}
    answer = parse_multi_choice_response(pred, ["A", "B", "C", "D"], option_list)
    gt_answer = doc["answer"]
    assert answer in ["A", "B", "C", "D"]
    assert gt_answer in ["A", "B", "C", "D"]
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "av_odyssey_score": {
            "question_id": doc["question_id"],
            "category": doc["subfield"],
            "task_id": doc["question_type_id"],
            "data_type": doc["data_type"],
            "score": score,
        }
    }


def av_odyssey_aggregate_results(results):
    """Aggregate per-subfield, per-task, and per-task-per-layout accuracy
    with 95% confidence intervals for all groupings.
    """
    category2scores = defaultdict(list)
    task2scores = defaultdict(list)
    task_layout2scores = defaultdict(lambda: defaultdict(list))
    task2subfield = {}

    for result in results:
        category2scores[result["category"]].append(result["score"])
        task2scores[result["task_id"]].append(result["score"])
        task_layout2scores[result["task_id"]][result["data_type"]].append(result["score"])
        task2subfield[result["task_id"]] = result["category"]

    # Per-subfield
    print()
    print("=" * 100)
    print("Per-subfield accuracy:")
    print("-" * 100)
    print(f"{'Subfield':<18} {'n':>6} {'Acc':>10} {'±95% CI':>10}")
    print("-" * 100)
    for category in sorted(category2scores):
        scores = category2scores[category]
        avg = sum(scores) / len(scores) * 100.0
        ci = _ci95(scores)
        print(f"{category:<18} {len(scores):>6} {avg:>9.2f}% {ci:>9.2f}")

    # Per-task with per-layout sub-rows for mixed-layout tasks
    print()
    print("=" * 100)
    print("Per-task accuracy (with layout breakdown for mixed-layout tasks):")
    print("-" * 100)
    print(f"{'ID':>3}  {'Task Name':<32} {'Subfield':<14} {'n':>5} {'Acc':>8} {'±95% CI':>10}")
    print("-" * 100)
    for tid in sorted(task2scores):
        name = TASK_ID_TO_NAME.get(tid, f"task_{tid}")
        subfield = task2subfield[tid]
        scores = task2scores[tid]
        avg = sum(scores) / len(scores) * 100.0
        ci = _ci95(scores)
        print(f"{tid:>3}  {name:<32} {subfield:<14} {len(scores):>5} {avg:>7.2f}% {ci:>9.2f}")

        layouts = task_layout2scores[tid]
        if len(layouts) > 1:
            for dt in sorted(layouts):
                dt_scores = layouts[dt]
                dt_avg = sum(dt_scores) / len(dt_scores) * 100.0
                dt_ci = _ci95(dt_scores)
                print(f"     └─ {dt:<50} {len(dt_scores):>5} {dt_avg:>7.2f}% {dt_ci:>9.2f}")

    # Overall
    overall_scores = [r["score"] for r in results]
    overall = sum(overall_scores) / len(overall_scores) * 100.0
    overall_ci = _ci95(overall_scores)
    print("-" * 100)
    print(f"Overall: {len(results)} samples, {overall:.2f}% ± {overall_ci:.2f}")
    print("=" * 100)
    print()

    return overall
