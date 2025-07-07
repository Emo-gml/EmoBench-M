#!/usr/bin/env python3
"""
eval.py: Unified evaluation script supporting three modes:
  1. classification — accuracy/precision/recall/F1 over one or multiple JSON files
  2. joint — emotion+intent joint evaluation from a JSONL file
  3. generation — BLEU-4, ROUGE-L, BERTScore for generated text (outputs metrics JSON and detailed CSV)

Usage examples:
  python eval.py classification --json results1.json results2.json \
      --categories positive negative neutral \
      --output class_metrics.json --invalid invalid_samples.json

  python eval.py joint --jsonl emotions_intents.jsonl \
      --emotion-cats happy sad neutral \
      --intent-cats questioning agreeing \
      --output joint_metrics.json

  python eval.py generation --json gen_results.json --output gen_metrics.json
"""
import os
import json
import argparse
import datetime
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd


def evaluate_classification(json_paths: List[str], categories: List[str], output_metrics: str, output_invalid: str = None) -> None:
    targets, preds = [], []
    invalid_samples = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sample in data:
            exp = sample.get('expected_value', '').strip().lower()
            pre = sample.get('predicted_value', '').strip().lower()
            if exp in categories and pre in categories:
                targets.append(exp)
                preds.append(pre)
            else:
                invalid_samples.append({
                    'file': path,
                    'sample': sample,
                    'error': 'invalid label',
                    'time': datetime.datetime.utcnow().isoformat() + 'Z'
                })
    if invalid_samples and output_invalid:
        with open(output_invalid, 'w', encoding='utf-8') as f:
            json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
        print(f"Invalid samples written to: {output_invalid}")
    if targets:
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_score': f1_score(targets, preds, average='weighted', zero_division=0)
        }
        with open(output_metrics, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Classification metrics saved to: {output_metrics}")
    else:
        print("No valid samples for classification evaluation.")


def evaluate_joint(jsonl_path: str, emotion_cats: List[str], intent_cats: List[str], output_metrics: str) -> None:
    true_e, pred_e, true_i, pred_i = [], [], [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                s = json.loads(line)
                te = s.get('expected_emotion', '').strip().lower()
                pe = s.get('predicted_emotion', '').strip().lower()
                ti = s.get('expected_intent', '').strip().lower()
                pi = s.get('predicted_intent', '').strip().lower()
                if te in emotion_cats and pe in emotion_cats and ti in intent_cats and pi in intent_cats:
                    true_e.append(te)
                    pred_e.append(pe)
                    true_i.append(ti)
                    pred_i.append(pi)
            except json.JSONDecodeError:
                continue
    joint_true = [f"{e}_{i}" for e, i in zip(true_e, true_i)]
    joint_pred = [f"{e}_{i}" for e, i in zip(pred_e, pred_i)]
    if joint_true:
        metrics = {
            'joint_accuracy': accuracy_score(joint_true, joint_pred),
            'joint_precision': precision_score(joint_true, joint_pred, average='weighted', zero_division=0),
            'joint_recall': recall_score(joint_true, joint_pred, average='weighted', zero_division=0),
            'joint_f1': f1_score(joint_true, joint_pred, average='weighted', zero_division=0),
            'total': len(joint_true)
        }
        with open(output_metrics, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Joint emotion-intent metrics saved to: {output_metrics}")
    else:
        print("No valid samples for joint evaluation.")


def evaluate_generation(json_file: str, output_metrics: str) -> None:
    data = json.load(open(json_file, 'r', encoding='utf-8'))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rows, bleu_list, rouge_list, bert_list = [], [], [], []
    for s in data:
        pred = s.get('prediction', '').strip()
        ref = s.get('reference', '').strip()
        vid = s.get('video', '')
        if not pred or not ref:
            continue
        b = sentence_bleu([ref.split()], pred.split())
        r = scorer.score(ref, pred)['rougeL'].fmeasure
        P, R, F = bert_score([pred], [ref], lang='en')
        bf = F.mean().item()
        rows.append({'video': vid, 'bleu': b, 'rouge': r, 'bert': bf})
        bleu_list.append(b); rouge_list.append(r); bert_list.append(bf)
    # Save detailed CSV alongside metrics
    csv_path = os.path.splitext(output_metrics)[0] + '_generation.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metrics = {
        'avg_bleu': sum(bleu_list) / len(bleu_list) if bleu_list else 0,
        'avg_rouge': sum(rouge_list) / len(rouge_list) if rouge_list else 0,
        'avg_bert': sum(bert_list) / len(bert_list) if bert_list else 0,
        'total': len(rows)
    }
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Generation metrics saved to: {output_metrics}")
    print(f"Detailed CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation')
    sub = parser.add_subparsers(dest='mode', required=True)
    # classification
    pc = sub.add_parser('classification')
    pc.add_argument('--json', nargs='+', required=True)
    pc.add_argument('--categories', nargs='+', required=True)
    pc.add_argument('--output', required=True)
    pc.add_argument('--invalid', default=None)
    # joint
    pj = sub.add_parser('joint')
    pj.add_argument('--jsonl', required=True)
    pj.add_argument('--emotion-cats', nargs='+', required=True)
    pj.add_argument('--intent-cats', nargs='+', required=True)
    pj.add_argument('--output', required=True)
    # generation
    pg = sub.add_parser('generation')
    pg.add_argument('--json', required=True)
    pg.add_argument('--output', required=True)

    args = parser.parse_args()
    if args.mode == 'classification':
        evaluate_classification(args.json, args.categories, args.output, args.invalid)
    elif args.mode == 'joint':
        evaluate_joint(args.jsonl, args.emotion_cats, args.intent_cats, args.output)
    elif args.mode == 'generation':
        evaluate_generation(args.json, args.output)

if __name__ == '__main__':
    main()

# === 示例文件结构 & 运行示例 ===
# 1. Classification JSON（results.json）:
# [
#   {"video":"a.mp4","expected_value":"positive","predicted_value":"neutral"},
#   {"video":"b.mp4","expected_value":"negative","predicted_value":"negative"}
# ]
# 运行:
# python eval.py classification --json results.json --categories positive negative neutral \
#     --output class_metrics.json --invalid invalid_samples.json

# 2. Joint JSONL（emotions.jsonl）:
# {"modal_path":"/p/a.mp4","expected_emotion":"happy","expected_intent":"encouraging","predicted_emotion":"happy","predicted_intent":"encouraging"}
# {"modal_path":"/p/b.mp4","expected_emotion":"sad","expected_intent":"questioning","predicted_emotion":"sad","predicted_intent":"neutral"}
# 运行:
# python eval.py joint --jsonl emotions.jsonl --emotion-cats happy sad neutral \
#     --intent-cats questioning agreeing --output joint_metrics.json

# 3. Generation JSON（gen.json）:
# [
#   {"video":"a.mp4","prediction":"I am happy","reference":"I feel happy"},
#   {"video":"b.mp4","prediction":"He looks sad","reference":"He seems sad"}
# ]
# 运行:
# python eval.py generation --json gen.json --output gen_metrics.json
