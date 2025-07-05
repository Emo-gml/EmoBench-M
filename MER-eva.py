# Multi-Modal Emotion Recognition Evaluator
#This is a benchmark evaluation pipeline for multi-modal large models (e.g., Video-LLaMA2), covering various emotion recognition datasets across modalities like audio, video, and audio-video fusion.
## Supported Datasets
#- FGMSA
#- MELD
#- MOSI
#- MOSEI
#- SIMS
#- RAVDSS (speech & song)
#- MUSTARD
#- CH-SIMSv2S
#- Funny
#- MER2023
## Usage
#1. Modify `main()` in `evaluate.py` to specify:
#  - Model path
#  - Dataset JSON path
#  - Video base path
#2. Run:
#```bash
#python evaluate.py

import sys
import os
import json
import re  # 正则表达式库
import torch
import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import threading
import datetime
from typing import List, Dict
import cv2  # OpenCV 用于检查视频文件

# 设置使用的 GPU
# torch.cuda.set_device(7)  # 如果想使用其他 GPU，请修改为对应的数字
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 根据需要设置可见的 GPU

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# 超时处理类
class TimeoutException(Exception):
    pass

def handle_invalid_answer(e: Exception, sample: Dict, invalid_answers_count: int) -> int:
    """
    统一处理错误和统计无效答案。

    :param e: 捕获的异常
    :param sample: 当前处理的样本
    :param invalid_answers_count: 当前无效答案计数
    :return: 更新后的无效答案计数
    """
    print(f"错误: {e}, 样本: {sample.get('video', '未知')}")
    return invalid_answers_count + 1

def infer_with_timeout(modal_path: str, question: str, processor: Dict, model, tokenizer, modal: str, timeout: int) -> str:
    """
    带有超时机制的推理方法。

    :param modal_path: 视频路径
    :param question: 指令
    :param processor: 处理器
    :param model: 模型
    :param tokenizer: 分词器
    :param modal: 模态类型
    :param timeout: 超时时间（秒）
    :return: 模型预测的值
    """
    result = None

    def target():
        nonlocal result
        result = mm_infer(
            processor[modal](modal_path), 
            question, 
            model=model, 
            tokenizer=tokenizer, 
            do_sample=False, 
            modal=modal
        )

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutException("样本处理超时")
    return result

def check_video_validity(video_path):
    """
    检查视频文件是否可以正常读取。
    :param video_path: 视频文件路径
    :return: 如果可以正常读取，返回 True，否则返回 False
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception as e:
        print(f"无法读取视频 {video_path}: {e}")
        return False

def evaluate_model(
    json_file_path: str,
    model_path: str,
    video_base_path: str,
    output_jsonl_path: str,
    output_invalid_json_path: str,
    categories: List[str],
    timeout: int = 60,
    modal_type: str = "av"
) -> Dict[str, float]:
    """
    评估模型的准确率，并将结果保存到一个 JSON Lines 文件中。
    支持中断恢复功能，并记录已损坏的行。
    记录无法提取情感标签的样本到一个单独的 JSON 文件中。

    :param json_file_path: JSON 文件路径
    :param model_path: 模型路径
    :param video_base_path: 视频文件的基础路径
    :param output_jsonl_path: 输出 JSON Lines 文件路径
    :param output_invalid_json_path: 输出无效样本的 JSON 文件路径
    :param categories: 类别列表
    :param timeout: 每个样本的最大处理时间（秒）
    :param modal_type: 模态类型，可选 "a"（音频）、"v"（视频）、"av"（音频+视频）
    :return: 评估指标的字典
    """
    # 显示当前 GPU 信息
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用的 GPU: {current_device} - {gpu_name}")
    else:
        print("CUDA 不可用，使用 CPU。")

    # 初始化模型
    try:
        model, processor, tokenizer = model_init(model_path)
        print(f"模型已成功加载: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return {}

    # 根据 modal_type 设置模型的组件
    if modal_type == "a":
        model.model.vision_tower = None
    elif modal_type == "v":
        model.model.audio_tower = None
    elif modal_type == "av":
        pass
    else:
        raise NotImplementedError("Invalid modal_type!")

    # 加载 JSON 数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"已加载 JSON 数据: {json_file_path}，总样本数: {len(data)}")
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return {}

    # 初始化结果存储
    processed_videos = set()
    results = []
    corrupted_lines_count = 0
    corrupted_lines_details = []

    invalid_samples = []

    # 如果输出文件已存在，读取已处理的样本
    if os.path.exists(output_jsonl_path):
        print(f"发现已有的结果文件: {output_jsonl_path}，将从中断处继续处理。")
        try:
            with open(output_jsonl_path, 'r', encoding='utf-8') as outfile:
                for line_num, line in enumerate(outfile, 1):
                    try:
                        result = json.loads(line)
                        modal_path = result['modal_path']
                        # 计算相对路径，保证在跳过判断时一致
                        video_rel_path = os.path.relpath(modal_path, video_base_path).replace('\\', '/').strip().lower()
                        processed_videos.add(video_rel_path)
                        results.append(result)
                    except json.JSONDecodeError:
                        corrupted_lines_count += 1
                        corrupted_lines_details.append(line_num)
                        print(f"损坏的行已记录: 行号 {line_num}")
        except Exception as e:
            print(f"读取结果文件时出错: {e}")
    else:
        print(f"将创建新的结果文件: {output_jsonl_path}")

    if corrupted_lines_count > 0:
        print(f"总共有 {corrupted_lines_count} 行损坏，位于行号: {corrupted_lines_details}")

    # 计算被跳过的样本并输出信息
    all_video_filenames = set(sample.get("video", "").strip().lower() for sample in data)
    skipped_videos = processed_videos & all_video_filenames

    for video in skipped_videos:
        print(f"跳过样本: {video}，因为它已经在结果文件中存在。")

    # 计算剩余未处理的样本
    remaining_data = [
        sample for sample in data 
        if sample.get("video", "").strip().lower() not in processed_videos
    ]
    print(f"剩余未处理的样本数量: {len(remaining_data)}")

    # 初始化统计
    correct_predictions = sum(
        1 for r in results 
        if r.get('predicted_value') == r.get('expected_value')
    )
    invalid_answers_count = 0

    targets_list = [r.get('expected_value') for r in results]
    answers_list = [r.get('predicted_value') for r in results]

    # 打开输出文件以追加新结果
    try:
        with open(output_jsonl_path, 'a', encoding='utf-8') as outfile:
            for sample in tqdm(remaining_data, desc="Evaluating samples"):
                try:
                    # 拼接完整视频路径
                    video_path = sample.get("video", "").strip()
                    modal_path = os.path.join(video_base_path, video_path).replace('\\', '/')
                    if not os.path.exists(modal_path):
                        raise FileNotFoundError(f"视频文件不存在: {modal_path}")

                    if not check_video_validity(modal_path):
                        raise ValueError(f"视频文件无法正常读取: {modal_path}")

                    conversations = sample.get("conversations", [])
                    if len(conversations) < 2:
                        raise ValueError("对话内容不完整")

                    question = conversations[0].get("value", "").replace("<video>", "").strip()
                    instruct = question
                    expected_value = conversations[1].get("value", "").strip().lower()

                    # 设置 modal 为 'audio' 或 'video'
                    if modal_type == "a":
                        modal = 'audio'
                    else:
                        modal = 'video'

                    # 预处理输入数据
                    preprocess = processor['audio' if modal_type == "a" else "video"]
                    if modal_type == "a":
                        audio_video_tensor = preprocess(modal_path)
                    else:
                        audio_video_tensor = preprocess(modal_path, va=True if modal_type == "av" else False)

                    # 模型推理，带超时
                    raw_predicted_value = infer_with_timeout(
                        modal_path, 
                        instruct, 
                        processor, 
                        model, 
                        tokenizer, 
                        modal, 
                        timeout
                    )
                    print(f"原始输出: {raw_predicted_value}")

                    # 使用正则表达式提取情感标签，只要是categories中的标签就提取
                    pattern = r"\b(" + "|".join(map(re.escape, categories)) + r")\b"
                    match = re.search(pattern, raw_predicted_value.lower())
                    if match:
                        predicted_value = match.group(1).strip().lower()
                        print(f"Predicted: {predicted_value}, Expected: {expected_value}")

                        # 保存结果
                        result_entry = {
                            "modal_path": modal_path,
                            "instruct": instruct,
                            "expected_value": expected_value,
                            "predicted_value": predicted_value,
                            "raw_predicted_value": raw_predicted_value
                        }
                        outfile.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                        outfile.flush()

                        # 更新统计
                        targets_list.append(expected_value)
                        answers_list.append(predicted_value)
                        if predicted_value == expected_value:
                            correct_predictions += 1
                    else:
                        raise ValueError("无法提取情感标签")

                except (TimeoutException, FileNotFoundError, ValueError) as e:
                    invalid_answers_count = handle_invalid_answer(e, sample, invalid_answers_count)
                    invalid_samples.append({
                        "sample": sample,
                        "raw_predicted_value": raw_predicted_value if 'raw_predicted_value' in locals() else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    })
                    continue
                except Exception as e:
                    invalid_answers_count = handle_invalid_answer(e, sample, invalid_answers_count)
                    invalid_samples.append({
                        "sample": sample,
                        "raw_predicted_value": raw_predicted_value if 'raw_predicted_value' in locals() else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    })
                    continue
                finally:
                    # 清理内存
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"写入结果文件时出错: {e}")

    # 将无效样本写入 JSON 文件
    if invalid_samples:
        try:
            with open(output_invalid_json_path, 'w', encoding='utf-8') as invalid_file:
                json.dump(invalid_samples, invalid_file, ensure_ascii=False, indent=4)
            print(f"\n已将 {len(invalid_samples)} 个无效样本记录到: {output_invalid_json_path}")
        except Exception as e:
            print(f"无法写入无效样本文件: {e}")
    else:
        print("\n没有无效样本需要记录。")

    # 计算评估指标
    metrics = {}
    if targets_list and answers_list:
        accuracy_metric = accuracy_score(targets_list, answers_list)
        precision = precision_score(targets_list, answers_list, average='weighted', zero_division=0)
        recall = recall_score(targets_list, answers_list, average='weighted', zero_division=0)
        f1 = f1_score(targets_list, answers_list, average='weighted', zero_division=0)

        # 这里新增保存评估的文件名与模型信息
        metrics = {
            "json_file_used": os.path.basename(json_file_path),
            "model_used": os.path.basename(model_path),
            "accuracy": accuracy_metric,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"\n评估结果 for {os.path.basename(json_file_path)} (使用模型: {model_path}):")
        print(f"准确率 (sklearn): {accuracy_metric:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1 分数: {f1:.4f}")
    else:
        print("没有有效的预测结果来计算指标。")

    print(f"无效输出数量: {invalid_answers_count}")
    if corrupted_lines_count > 0:
        print(f"已记录 {corrupted_lines_count} 行损坏的数据，位于行号: {corrupted_lines_details}")

    return metrics

def main():
    """
    主函数：定义需要评估的数据集配置，并调用 evaluate_model 进行评估。
    最后将结果写入综合评估 JSON 文件。
    """
    # 定义数据集配置，每个配置包含 JSON 文件路径和对应的类别
    datasets = [
        {
            "name": "FGMSA",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/FGMSA_test_instructuin.json",
            "categories": ["weak negative", "strong negative", "neutral", "strong positive", "weak positive"]
        },
        {
            "name": "ch-simsv2s",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/ch-simsv2s_test_500.json",
            "categories": ["neutral", "negative", "positive"]
        },
        {
            "name": "MOSI",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/MOSI_test_500.json",
            "categories": ["neutral", "negative", "positive"]
        },
        {
            "name": "SIMS",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/SIMS_test_500.json",
            "categories": ["neutral", "negative", "positive"]
        },
        {
            "name": "funny",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/funny_test_instruction.json",
            "categories": ["true", "false"]
        },
        {
            "name": "MUSTARD",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/MUSTARD_500.json",
            "categories": ["true", "false"]
        },
        {
            "name": "MELD",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/MELD_test_instruction.json",
            "categories": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
        },
        {
            "name": "mer2023",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/mer2023_test1_instruction.json",
            "categories": ["happy", "sad", "neutral", "angry", "worried", "surprise"]
        },
        {
            "name": "RAVDSS-song",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/RAVDSS_song_500.json",
            "categories": ["neutral", "calm", "happy", "sad", "angry", "fearful"]
        },
        {
            "name": "RAVDSS-speech",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/RAVDSS_speech_500.json",
            "categories": ["neutral", "calm", "happy", "sad", "angry", "fearful", "surprised", "disgust"]
        },
        {
            "name": "MOSEI",
            "json_file_path": "/data/VideoLLaMA2-audio_visual/benchmark_json/MOSEI_test_500.json",
            "categories": ["neutral", "negative", "positive"]
        },
    ]

    # 视频基础路径
    video_base_path = "/data/VideoLLaMA2-audio_visual/datasets"

    # 模型路径
    model_path = "/data/video-llama2-av/av-weight/VideoLLaMA2.1-7B-AV"

    # 定义输出基础路径
    output_base_path = "/data/VideoLLaMA2-audio_visual/result/VideoLLaMA2.1-7B-AV/"
    os.makedirs(output_base_path, exist_ok=True)

    # 遍历每个数据集并进行评估
    all_metrics = {}
    for dataset in datasets:
        json_file_path = dataset["json_file_path"]
        categories = dataset["categories"]

        # 提取输入 JSON 文件名（不含扩展名）
        json_filename = os.path.splitext(os.path.basename(json_file_path))[0]

        # 构建输出 JSON Lines 文件路径
        output_jsonl_path = os.path.join(output_base_path, f"{json_filename}.jsonl")

        # 构建输出无效样本 JSON 文件路径
        output_invalid_json_path = os.path.join(output_base_path, f"{json_filename}_invalid_samples.json")

        print(f"\n开始评估数据集: {json_filename}")
        metrics = evaluate_model(
            json_file_path=json_file_path,
            model_path=model_path,
            video_base_path=video_base_path,
            output_jsonl_path=output_jsonl_path,
            output_invalid_json_path=output_invalid_json_path,
            categories=categories,
            timeout=60,  # 可根据需要调整超时时间
            modal_type="av"  # 根据需要调整模态类型："a"、"v" 或 "av"
        )
        if metrics:
            all_metrics[json_filename] = metrics

    # 保存所有评估结果到一个综合 JSON 文件（可选）
    comprehensive_result_path = os.path.join(output_base_path, "comprehensive_evaluation_results.json")
    try:
        with open(comprehensive_result_path, 'w', encoding='utf-8') as comp_file:
            json.dump(all_metrics, comp_file, ensure_ascii=False, indent=4)
        print(f"\n所有评估结果已保存到: {comprehensive_result_path}")
    except Exception as e:
        print(f"无法写入综合评估结果文件: {e}")

if __name__ == "__main__":
    main()


