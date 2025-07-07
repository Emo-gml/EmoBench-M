import os
import json
import re
import datetime
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def evaluate_predictions(
    json_file_path: str,
    output_invalid_json_path: str,
    categories: List[str]
) -> Dict[str, float]:
    """
    评估模型预测准确率和其他指标。输入 JSON 文件应包含:
    - video: 视频文件名
    - expected_value: 真实标签
    - predicted_value: 模型预测的标签

    :param json_file_path: 输入 JSON 文件路径
    :param output_invalid_json_path: 输出无效样本的 JSON 文件路径
    :param categories: 支持的标签类别
    :return: 评估指标字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return {}

    targets_list = []
    answers_list = []
    invalid_samples = []

    for sample in tqdm(data, desc="Evaluating"):
        expected = sample.get("expected_value", "").strip().lower()
        predicted = sample.get("predicted_value", "").strip().lower()

        if expected in categories and predicted in categories:
            targets_list.append(expected)
            answers_list.append(predicted)
        else:
            invalid_samples.append({
                "sample": sample,
                "error": "标签无效",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            })

    if invalid_samples:
        try:
            with open(output_invalid_json_path, 'w', encoding='utf-8') as f:
                json.dump(invalid_samples, f, ensure_ascii=False, indent=4)
            print(f"\n无效样本已记录到: {output_invalid_json_path}")
        except Exception as e:
            print(f"无法写入无效样本: {e}")

    if targets_list and answers_list:
        accuracy = accuracy_score(targets_list, answers_list)
        precision = precision_score(targets_list, answers_list, average='weighted', zero_division=0)
        recall = recall_score(targets_list, answers_list, average='weighted', zero_division=0)
        f1 = f1_score(targets_list, answers_list, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print("\n评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1 分数: {f1:.4f}")
        return metrics
    else:
        print("无有效样本用于评估。")
        return {}

# 用法示例
# result = evaluate_predictions(
#     json_file_path="example_results.json",
#     output_invalid_json_path="invalid_samples.json",
#     categories=["positive", "negative", "neutral"]
# )

# 支持的 .json 文件结构如下：
#[
#  {
#    "video": "xxx.mp4",
#    "expected_value": "positive",
#    "predicted_value": "positive"
#  },
#  {
#    "video": "yyy.mp4",
#    "expected_value": "negative",
#    "predicted_value": "neutral"
#  },
#  ...
#]

