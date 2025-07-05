import sys
import os
import json
from tqdm import tqdm
import re  # 正则表达式库
import torch
import cv2  # OpenCV 用于检查视频文件
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score
import pandas as pd  # 用于生成 DataFrame

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

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

def evaluate_model(json_file_path, model_path, video_base_path, output_file_path, modal_type="av"):
    """
    推理并将输出保存到文件，同时计算 BLEU-4、METEOR、ROUGE-L 和 BERTScore。
    :param json_file_path: JSON 文件路径
    :param model_path: 模型路径
    :param video_base_path: 视频文件的基础路径
    :param output_file_path: 推理结果保存的文件路径
    :param modal_type: 模态类型，可选 "a"（音频）、"v"（视频）、"av"（音频+视频）  
    """
    # 显示当前使用的 GPU 信息
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前使用的 GPU: {current_device} - {gpu_name}")

    # 初始化模型
    model, processor, tokenizer = model_init(model_path)

    if modal_type == "a":
        model.model.vision_tower = None
    elif modal_type == "v":
        model.model.audio_tower = None
    elif modal_type == "av":
        pass
    else:
        raise NotImplementedError("Invalid modal_type!")

    # 加载 JSON 数据
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 记录推理结果
    results = []
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    bert_scores = []

    rouge_evaluator = Rouge()

    for sample in tqdm(data, desc="Evaluating samples"):
        video_path = os.path.join(video_base_path, sample["video"])

        if not check_video_validity(video_path):
            print(f"无法读取视频文件: {video_path}")
            results.append({
                'video': sample['video'],
                'prediction': "无法读取视频",
                'bleu': None,
                'rouge': None,
                'bert': None,
            })
            continue

        question = sample["conversations"][0]["value"].replace("<video>", "").strip()
        expected_value = sample["conversations"][1]["value"].strip()

        # 预处理输入数据
        preprocess = processor['audio' if modal_type == "a" else "video"]
        if modal_type == "a":
            audio_video_tensor = preprocess(video_path)
        else:
            audio_video_tensor = preprocess(video_path, va=True if modal_type == "av" else False)

        # 模型推理
        try:
            predicted_value = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal='audio' if modal_type == "a" else "video",
                do_sample=False,
            ).strip()
            print(predicted_value)

            # 计算 BLEU-4
            bleu = sentence_bleu([expected_value.split()], predicted_value.split())
            bleu_scores.append(bleu)

            # 计算 ROUGE-L
            rouge_score = rouge_evaluator.get_scores(predicted_value, expected_value)[0]['rouge-l']['f']
            rouge_scores.append(rouge_score)

            # 计算 BERTScore
            P, R, F1 = bert_score([predicted_value], [expected_value], lang="en")
            bert_scores.append(F1.mean().item())

            # 保存推理结果
            results.append({
                'video': sample['video'],
                'prediction': predicted_value,
                'bleu': bleu,
                'rouge': rouge_score,
                'bert': F1.mean().item(),
            })

        except Exception as e:
            print(f"模型推理失败: {e}")
            results.append({
                'video': sample['video'],
                'prediction': f"推理失败: {e}",
                'bleu': None,
                'rouge': None,
                'bert': None,
            })
            continue

    # 将结果保存到 CSV 文件
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file_path, index=False, sep='\t')

    # 输出平均指标
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0

    print(f"平均 BLEU-4: {avg_bleu:.4f}")
    print(f"平均 ROUGE-L: {avg_rouge:.4f}")
    print(f"平均 BERTScore: {avg_bert:.4f}")

    return avg_bleu, avg_rouge, avg_bert

if __name__ == "__main__":
    json_file_path = "/data/VideoLLaMA2-audio_visual/benchmark_json/smile_test_data.json"  # 替换为实际 JSON 文件路径
    video_base_path = "/data/VideoLLaMA2-audio_visual/datasets"  # 替换为视频文件的基础路径
    model_path = "/data/video-llama2-av/av-weight/VideoLLaMA2.1-7B-AV"  # 模型路径
    modal_type = "av"  # 模态类型
    output_file_path = "/data/VideoLLaMA2-audio_visual/Video-llama2_smile.csv"  # 输出路径将生成的输出到一个csv文件后边展示

    evaluate_model(json_file_path, model_path, video_base_path, output_file_path, modal_type)


