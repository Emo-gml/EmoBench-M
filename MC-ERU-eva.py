import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List

def evaluate_emotion_intent_from_jsonl(jsonl_path: str, emotion_categories: List[str], intent_categories: List[str]) -> dict:
    true_emotions = []
    pred_emotions = []
    true_intents = []
    pred_intents = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                sample = json.loads(line)
                expected_emotion = sample.get("expected_emotion", "").strip().lower()
                predicted_emotion = sample.get("predicted_emotion", "").strip().lower()
                expected_intent = sample.get("expected_intent", "").strip().lower()
                predicted_intent = sample.get("predicted_intent", "").strip().lower()

                if (expected_emotion in emotion_categories and predicted_emotion in emotion_categories and
                    expected_intent in intent_categories and predicted_intent in intent_categories):
                    true_emotions.append(expected_emotion)
                    pred_emotions.append(predicted_emotion)
                    true_intents.append(expected_intent)
                    pred_intents.append(predicted_intent)
            except json.JSONDecodeError:
                continue

    if not true_emotions or not true_intents:
        return {"error": "No valid predictions found."}

    true_joint = [f"{e}_{i}" for e, i in zip(true_emotions, true_intents)]
    pred_joint = [f"{e}_{i}" for e, i in zip(pred_emotions, pred_intents)]

    return {
        "joint_accuracy": accuracy_score(true_joint, pred_joint),
        "joint_precision": precision_score(true_joint, pred_joint, average='weighted', zero_division=0),
        "joint_recall": recall_score(true_joint, pred_joint, average='weighted', zero_division=0),
        "joint_f1": f1_score(true_joint, pred_joint, average='weighted', zero_division=0),
        "total_samples": len(true_joint)
    }

# 示例调用（用户需要自己替换路径和分类列表）
# metrics = evaluate_emotion_intent_from_jsonl(
#     jsonl_path="your_output.jsonl",
#     emotion_categories=["happy", "surprise", "sad", "disgust", "anger", "fear", "neutral"],
#     intent_categories=["questioning", "agreeing", "acknowledging", "encouraging", "consoling", "suggesting", "wishing", "neutral"]
# )
# print(metrics)
#.jsonl 输入格式要求
#每一行是一个 JSON 对象，示例如下：
#{
#  "modal_path": "/path/to/video.mp4",
#  "question": "What is the user expressing?",
#  "expected_emotion": "happy",
#  "expected_intent": "encouraging",
#  "predicted_emotion": "happy",
#  "predicted_intent": "encouraging",
#  "raw_predicted_value": "The speaker seems happy and is encouraging the listener."
#}
