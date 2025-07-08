#!/usr/bin/env python3
"""
eval.py: Unified evaluation script supporting four modes:
  1. classification — compute accuracy, precision, recall, F1 across JSON files
  2. joint — evaluate emotion and intent together from a JSONL file
  3. generation — compute BLEU-4, ROUGE-L, and BERTScore for generated text
  4. all — run all three evaluations in one command

Usage examples:
  python eval.py classification --json results1.json results2.json --output class_metrics.json
  python eval.py joint --jsonl emotions.jsonl --output joint_metrics.json
  python eval.py generation --json gen.json --output gen_metrics.json

  # All-in-one evaluation
  python eval.py all \\
    --classification-json results1.json results2.json \\
    --joint-jsonl emotions.jsonl \\
    --generation-json gen.json \\
    --output-dir results/
"""

import os
import json
import argparse
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_classification(json_paths: List[str], output_metrics: str) -> None:
    targets, preds = [], []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sample in data:
            exp = sample.get('expected_value', '').strip().lower()
            pre = sample.get('predicted_value', '').strip().lower()
            if exp and pre:
                targets.append(exp)
                preds.append(pre)
    if targets:
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_score': f1_score(targets, preds, average='weighted', zero_division=0)
        }
        with open(output_metrics, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[classification] Metrics saved to: {output_metrics}")
    else:
        print("[classification] No valid samples.")

def evaluate_joint(jsonl_path: str, output_metrics: str) -> None:
    true_e, pred_e, true_i, pred_i = [], [], [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                s = json.loads(line)
                te = s.get('expected_emotion', '').strip().lower()
                pe = s.get('predicted_emotion', '').strip().lower()
                ti = s.get('expected_intent', '').strip().lower()
                pi = s.get('predicted_intent', '').strip().lower()
                if te and pe and ti and pi:
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
        print(f"[joint] Metrics saved to: {output_metrics}")
    else:
        print("[joint] No valid samples.")

def evaluate_generation(json_file: str, output_metrics: str) -> None:
    data = json.load(open(json_file, 'r', encoding='utf-8'))
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_list, rouge_list, bert_list = [], [], []
    for s in data:
        pred = s.get('prediction', '').strip()
        ref = s.get('reference', '').strip()
        if not pred or not ref:
            continue
        bleu_val = sentence_bleu([ref.split()], pred.split())
        rouge_val = scorer.score(ref, pred)['rougeL'].fmeasure
        _, _, F = bert_score([pred], [ref], lang='en')
        bert_val = F.mean().item()
        bleu_list.append(bleu_val)
        rouge_list.append(rouge_val)
        bert_list.append(bert_val)
    metrics = {
        'avg_bleu': sum(bleu_list) / len(bleu_list) if bleu_list else 0,
        'avg_rouge': sum(rouge_list) / len(rouge_list) if rouge_list else 0,
        'avg_bert': sum(bert_list) / len(bert_list) if bert_list else 0,
        'total': len(bleu_list)
    }
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[generation] Metrics saved to: {output_metrics}")

def main():
    parser = argparse.ArgumentParser(description='Unified evaluation')
    sub = parser.add_subparsers(dest='mode', required=True)

    pc = sub.add_parser('classification')
    pc.add_argument('--json', nargs='+', required=True)
    pc.add_argument('--output', required=True)

    pj = sub.add_parser('joint')
    pj.add_argument('--jsonl', required=True)
    pj.add_argument('--output', required=True)

    pg = sub.add_parser('generation')
    pg.add_argument('--json', required=True)
    pg.add_argument('--output', required=True)

    pa = sub.add_parser('all')
    pa.add_argument('--classification-json', nargs='+')
    pa.add_argument('--joint-jsonl')
    pa.add_argument('--generation-json')
    pa.add_argument('--output-dir', required=True)

    args = parser.parse_args()
    if args.mode == 'classification':
        evaluate_classification(args.json, args.output)
    elif args.mode == 'joint':
        evaluate_joint(args.jsonl, args.output)
    elif args.mode == 'generation':
        evaluate_generation(args.json, args.output)
    elif args.mode == 'all':
        os.makedirs(args.output_dir, exist_ok=True)
        if args.classification_json:
            evaluate_classification(args.classification_json, os.path.join(args.output_dir, 'classification.json'))
        if args.joint_jsonl:
            evaluate_joint(args.joint_jsonl, os.path.join(args.output_dir, 'joint.json'))
        if args.generation_json:
            evaluate_generation(args.generation_json, os.path.join(args.output_dir, 'generation.json'))

if __name__ == '__main__':
    main()


# === Example file structure & running example ===
# 1. Classification JSON (results.json):
# [
# {"video":"a.mp4","expected_value":"positive","predicted_value":"neutral"},
# {"video":"b.mp4","expected_value":"negative","predicted_value":"negative"}
# ]
# Run:
# python eval.py classification --json results.json --categories positive negative neutral \
# --output class_metrics.json --invalid invalid_samples.json

# 2. Joint JSONL (emotions.jsonl):
# {"modal_path":"/p/a.mp4","expected_emotion":"happy","expected_intent":"encouraging","predicted_emotion":"happy","predicted_intent":"encouraging"}
# {"modal_path":"/p/b.mp4","expected_emotion":"sad","expected_intent":"questioning","predicted_emotion":"sad","predicted_intent":"neutral"}
# Run:
# python eval.py joint --jsonl emotions.jsonl --emotion-cats happy sad neutral \
# --intent-cats questioning agreeing acknowledging \(\emotion_categories = ["happy", "surprise", "sad", "disgust", "anger", "fear", "neutral"]\)
# --output joint_metrics.json

# 3. Generation JSON (gen.json):
# [
# {"video":"a.mp4","prediction":"I am happy","reference":"I feel happy"},
# {"video":"b.mp4","prediction":"He looks sad","reference":"He seems sad"}
# ]
# Run:
# python eval.py generation --json gen.json --output gen_metrics.json

# === Example categories ===
# Save the following as categories to centrally manage dataset and MC-EIU settings:
# {
#   "datasets": [
#     {"name": "FGMSA",       "categories": ["weak negative","strong negative","neutral","strong positive","weak positive"]},
#     {"name": "ch-simsv2s", "categories": ["neutral","negative","positive"]},
#     {"name": "MOSI",        "categories": ["neutral","negative","positive"]},
#     {"name": "SIMS",        "categories": ["neutral","negative","positive"]},
#     {"name": "funny",       "categories": ["true","false"]},
#     {"name": "MUSTARD",     "categories": ["true","false"]},
#     {"name": "MELD",        "categories": ["neutral","surprise","fear","sadness","joy","disgust","anger"]},
#     {"name": "mer2023",     "categories": ["happy","sad","neutral","angry","worried","surprise"]},
#     {"name": "RAVDSS-song", "categories": ["neutral","calm","happy","sad","angry","fearful"]},
#     {"name": "RAVDSS-speech","categories": ["neutral","calm","happy","sad","angry","fearful","surprised","disgust"]},
#     {"name": "MOSEI",       "categories": ["neutral","negative","positive"]}
#   ],
#   "MC-EIU": {
#     "emotion_categories": ["happy","surprise","sad","disgust","anger","fear","neutral"],
#     "intent_categories":  ["questioning","agreeing","acknowledging","encouraging","consoling","suggesting","wishing","neutral"]
#   }
# }
