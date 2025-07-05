
import sys
import os
import json
import re
import torch
import threading
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import List, Dict
import cv2  # OpenCV for video validation

# Ensure the CUDA device is set correctly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modify as needed

# Append current directory to system path
sys.path.append('./')

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass

def extract_emotion_first_match(text: str, emotion_categories: List[str]) -> str:
    """
    从 emotion_categories 中寻找文本中出现的第一个情感关键字并返回。
    若无匹配则返回 None。
    """
    text_lower = text.lower()
    for cat in emotion_categories:
        if cat in text_lower:
            return cat
    return None

def extract_intent_first_match(text: str, intent_categories: List[str]) -> str:
    """
    从 intent_categories 中寻找文本中出现的第一个意图关键字并返回。
    若无匹配则返回 None。
    """
    text_lower = text.lower()
    for cat in intent_categories:
        if cat in text_lower:
            return cat
    return None

def handle_invalid_answer(e: Exception, sample: Dict, invalid_answers_count: int) -> int:
    """
    Handles invalid answers by logging the error and incrementing the invalid count.
    """
    print(f"错误: {e}, 样本: {sample.get('video', '未知')}")
    return invalid_answers_count + 1

def infer_with_timeout(modal_path: str, question: str, processor: Dict, model, tokenizer, modal: str, timeout: int) -> str:
    """
    Performs model inference with a timeout mechanism.
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

def check_video_validity(video_path: str) -> bool:
    """
    Checks if the video file can be read properly.
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
    intent_categories: List[str],
    timeout: int = 60,
    modal_type: str = "av"
) -> Dict[str, float]:
    """
    Evaluates the model and logs outputs and errors.
    """
    # Display GPU information
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用的 GPU: {current_device} - {gpu_name}")
    else:
        print("CUDA 不可用，使用 CPU。")

    # Initialize the model
    try:
        model, processor, tokenizer = model_init(model_path)
        print(f"模型已成功加载: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return {}

    # Configure model based on modal_type
    if modal_type == "a":
        model.model.vision_tower = None
    elif modal_type == "v":
        model.model.audio_tower = None
    elif modal_type == "av":
        pass
    else:
        raise NotImplementedError("Invalid modal_type!")

    # Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"已加载 JSON 数据: {json_file_path}，总样本数: {len(data)}")
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return {}

    # Prepare output directories
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_invalid_json_path), exist_ok=True)

    # Initialize results and invalid samples
    results = []
    invalid_samples = []
    invalid_answers_count = 0

    # Check if output files already exist to resume processing
    processed_videos = set()
    if os.path.exists(output_jsonl_path):
        print(f"发现已有的结果文件: {output_jsonl_path}，将从中断处继续处理。")
        try:
            with open(output_jsonl_path, 'r', encoding='utf-8') as outfile:
                for line in outfile:
                    try:
                        result = json.loads(line)
                        video_rel_path = os.path.relpath(result['modal_path'], video_base_path).replace('\\', '/').strip().lower()
                        processed_videos.add(video_rel_path)
                        results.append(result)
                    except json.JSONDecodeError:
                        print("发现损坏的行，已跳过。")
        except Exception as e:
            print(f"读取结果文件时出错: {e}")

    # Identify remaining samples to process
    remaining_data = [sample for sample in data if sample.get("video", "").strip().lower() not in processed_videos]
    print(f"剩余未处理的样本数量: {len(remaining_data)}")

    # Open the JSONL output file in append mode
    try:
        with open(output_jsonl_path, 'a', encoding='utf-8') as outfile:
            for sample in tqdm(remaining_data, desc="Evaluating samples"):
                try:
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
                    expected_value = conversations[1].get("value", "").strip().lower()

                    # Determine modal type and preprocessing
                    if modal_type == "a":
                        modal = 'audio'
                        preprocess = processor['audio']
                        audio_video_tensor = preprocess(modal_path)
                    elif modal_type == "v":
                        modal = 'video'
                        preprocess = processor['video']
                        audio_video_tensor = preprocess(modal_path)
                    elif modal_type == "av":
                        # 用 video 处理器 + va=True 同时处理音视频
                        modal = 'video'
                        preprocess = processor['video']
                        audio_video_tensor = preprocess(modal_path, va=True)
                    else:
                        raise NotImplementedError("Invalid modal_type!")

                    # Perform inference with timeout
                    predicted_value = infer_with_timeout(modal_path, question, processor, model, tokenizer, modal, timeout)
                    raw_predicted_value = predicted_value.strip().lower()
                    print(f"原始输出: {raw_predicted_value}")

                    # ========== 优化的情感意图提取 ==========
                    predicted_emotion = extract_emotion_first_match(raw_predicted_value, categories)
                    predicted_intent = extract_intent_first_match(raw_predicted_value, intent_categories)

                    if not predicted_emotion or not predicted_intent:
                        raise ValueError("无法提取情感或意图标签")

                    # Extract true labels (ground truth)
                    emotion_true = extract_emotion_first_match(expected_value, categories)
                    intent_true = extract_intent_first_match(expected_value, intent_categories)

                    if not emotion_true or not intent_true:
                        raise ValueError("无法提取真实情感或意图标签")

                    print(f"Predicted Emotion: {predicted_emotion}, Expected Emotion: {emotion_true}")
                    print(f"Predicted Intent: {predicted_intent}, Expected Intent: {intent_true}")

                    # Prepare result entry
                    result_entry = {
                        "modal_path": modal_path,
                        "question": question,
                        "expected_emotion": emotion_true,
                        "expected_intent": intent_true,
                        "predicted_emotion": predicted_emotion,
                        "predicted_intent": predicted_intent,
                        "raw_predicted_value": raw_predicted_value
                    }

                    # Write result to JSONL
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    outfile.flush()

                    # Append to results
                    results.append(result_entry)

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
                    # Clear CUDA cache to free memory
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"写入结果文件时出错: {e}")

    # Save invalid samples to a separate JSON file
    if invalid_samples:
        try:
            with open(output_invalid_json_path, 'w', encoding='utf-8') as invalid_file:
                json.dump(invalid_samples, invalid_file, ensure_ascii=False, indent=4)
            print(f"\n已将 {len(invalid_samples)} 个无效样本记录到: {output_invalid_json_path}")
        except Exception as e:
            print(f"无法写入无效样本文件: {e}")
    else:
        print("\n没有无效样本需要记录。")

    # --------------------------------------------------
    # 评估指标（情感、意图、二者联合共 12 项）
    # --------------------------------------------------
    if results:
        # 1. 提取真值和预测值
        true_emotions = [r['expected_emotion'] for r in results]
        pred_emotions = [r['predicted_emotion'] for r in results]
        true_intents = [r['expected_intent'] for r in results]
        pred_intents = [r['predicted_intent'] for r in results]

        # 2. 分别计算情感和意图的 Accuracy
        accuracy_emotion = accuracy_score(true_emotions, pred_emotions)
        accuracy_intent = accuracy_score(true_intents, pred_intents)

        # 3. 分别计算情感和意图的 Precision, Recall, F1
        precision_emotion = precision_score(true_emotions, pred_emotions, average='weighted', zero_division=0)
        recall_emotion = recall_score(true_emotions, pred_emotions, average='weighted', zero_division=0)
        f1_emotion = f1_score(true_emotions, pred_emotions, average='weighted', zero_division=0)

        precision_intent = precision_score(true_intents, pred_intents, average='weighted', zero_division=0)
        recall_intent = recall_score(true_intents, pred_intents, average='weighted', zero_division=0)
        f1_intent = f1_score(true_intents, pred_intents, average='weighted', zero_division=0)

        # 4. (情感 + 意图) 将 (emotion, intent) 组合视为一个多分类标签
        true_joint = [f"{e}_{i}" for e, i in zip(true_emotions, true_intents)]
        pred_joint = [f"{e}_{i}" for e, i in zip(pred_emotions, pred_intents)]

        accuracy_joint   = accuracy_score(true_joint, pred_joint)
        precision_joint  = precision_score(true_joint, pred_joint, average='weighted', zero_division=0)
        recall_joint     = recall_score(true_joint, pred_joint, average='weighted', zero_division=0)
        f1_joint         = f1_score(true_joint, pred_joint, average='weighted', zero_division=0)

        # 5. 汇总所有指标（共 12 项）
        metrics = {
            # ---- 情感 ----
            "emotion_accuracy": accuracy_emotion,
            "emotion_precision": precision_emotion,
            "emotion_recall": recall_emotion,
            "emotion_f1": f1_emotion,
            # ---- 意图 ----
            "intent_accuracy": accuracy_intent,
            "intent_precision": precision_intent,
            "intent_recall": recall_intent,
            "intent_f1": f1_intent,
            # ---- 联合 ----
            "joint_accuracy": accuracy_joint,
            "joint_precision": precision_joint,
            "joint_recall": recall_joint,
            "joint_f1": f1_joint
        }

        # 6. 控制台打印
        print("\n评估结果:")
        print(f"Emotion Accuracy:   {accuracy_emotion:.4f}")
        print(f"Emotion Precision:  {precision_emotion:.4f}")
        print(f"Emotion Recall:     {recall_emotion:.4f}")
        print(f"Emotion F1:         {f1_emotion:.4f}")
        print(f"Intent Accuracy:    {accuracy_intent:.4f}")
        print(f"Intent Precision:   {precision_intent:.4f}")
        print(f"Intent Recall:      {recall_intent:.4f}")
        print(f"Intent F1:          {f1_intent:.4f}")
        print(f"Joint Accuracy:     {accuracy_joint:.4f}")
        print(f"Joint Precision:    {precision_joint:.4f}")
        print(f"Joint Recall:       {recall_joint:.4f}")
        print(f"Joint F1:           {f1_joint:.4f}")
    else:
        metrics = {}
        print("没有有效的预测结果来计算指标。")

    print(f"无效输出数量: {invalid_answers_count}")
    return metrics

def main():
    # Define the evaluation parameters
    json_file_path = "/data/VideoLLaMA2-audio_visual/benchmark_json/MC-EIU-test_500.json"  # Replace with actual path
    video_base_path = "/data/VideoLLaMA2-audio_visual/datasets"  # Replace with actual path
    model_path = "/data/video-llama2-av/av-weight/VideoLLaMA2.1-7B-AV"
    modal_type = "av"  # Options: "a", "v", "av"

    # Define emotion and intent categories
    categories = ["happy", "surprise", "sad", "disgust", "anger", "fear", "neutral"]
    intent_categories = ["questioning", "agreeing", "acknowledging", "encouraging", "consoling", "suggesting", "wishing", "neutral"]

    # Define output paths
    output_base_path = "/data/VideoLLaMA2-audio_visual/result/VideoLLaMA2.1-7B-AV"  # Replace with desired output directory
    os.makedirs(output_base_path, exist_ok=True)

    # Define output file paths
    json_filename = os.path.splitext(os.path.basename(json_file_path))[0]
    output_jsonl_path = os.path.join(output_base_path, f"{json_filename}.jsonl")
    output_invalid_json_path = os.path.join(output_base_path, f"{json_filename}_invalid_samples.json")

    # Evaluate the model
    metrics = evaluate_model(
        json_file_path=json_file_path,
        model_path=model_path,
        video_base_path=video_base_path,
        output_jsonl_path=output_jsonl_path,
        output_invalid_json_path=output_invalid_json_path,
        categories=categories,
        intent_categories=intent_categories,
        timeout=60,  # Adjust as needed
        modal_type=modal_type
    )

    # Optionally, save metrics to a comprehensive JSON file
    if metrics:
        comprehensive_result_path = os.path.join(output_base_path, "evaluation_metrics.json")
        try:
            with open(comprehensive_result_path, 'w', encoding='utf-8') as comp_file:
                json.dump({"MC-EIU-test_500": metrics}, comp_file, ensure_ascii=False, indent=4)
            print(f"\n评估指标已保存到: {comprehensive_result_path}")
        except Exception as e:
            print(f"无法写入评估指标文件: {e}")

if __name__ == "__main__":
    main()

