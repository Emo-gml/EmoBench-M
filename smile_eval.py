
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score

def evaluate_generation_from_json(json_file_path: str, output_file_path: str) -> dict:
    """
    Evaluate generation quality using BLEU-4, ROUGE-L, and BERTScore.
    :param json_file_path: Path to the input JSON file (with predictions and references).
    :param output_file_path: Path to save the evaluation results as CSV.
    :return: Dictionary of average metrics.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    bleu_scores = []
    rouge_scores = []
    bert_scores = []

    rouge_evaluator = Rouge()

    for sample in data:
        predicted = sample.get("prediction", "").strip()
        reference = sample.get("reference", "").strip()
        video = sample.get("video", "unknown")

        if not predicted or not reference:
            continue

        try:
            bleu = sentence_bleu([reference.split()], predicted.split())
            rouge_score = rouge_evaluator.get_scores(predicted, reference)[0]['rouge-l']['f']
            P, R, F1 = bert_score([predicted], [reference], lang="en")
            bert_f1 = F1.mean().item()

            bleu_scores.append(bleu)
            rouge_scores.append(rouge_score)
            bert_scores.append(bert_f1)

            results.append({
                'video': video,
                'prediction': predicted,
                'reference': reference,
                'bleu': bleu,
                'rouge': rouge_score,
                'bert': bert_f1
            })
        except Exception as e:
            print(f"Error on sample {video}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False, sep='\t')

    return {
        "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        "avg_rouge": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
        "avg_bert": sum(bert_scores) / len(bert_scores) if bert_scores else 0,
        "total_evaluated": len(results)
    }

# 示例调用（需替换路径）
# metrics = evaluate_generation_from_json(
#     json_file_path="your_data.json",
#     output_file_path="evaluated_results.csv"
# )
# print(metrics)

#.json 文件结构要求：
#[
#  {
#    "video": "example1.mp4",
#    "prediction": "the speaker looks happy",
#    "reference": "the person seems to be smiling and happy"
#  },
#  {
#    "video": "example2.mp4",
#    "prediction": "he appears to be angry",
#    "reference": "the speaker expresses anger"
#  }
#]
