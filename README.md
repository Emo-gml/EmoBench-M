# 🎯 Unified Evaluation Tool

A unified evaluation script supporting three NLP and multimodal evaluation tasks:

1. **Classification** — Compute accuracy, precision, recall, F1 from JSON files.
2. **Joint Evaluation** — Evaluate *emotion* and *intent* jointly from a JSONL file.
3. **Generation Evaluation** — Compute BLEU-4, ROUGE-L, and BERTScore from generated text.

---

## 📦 Features

- 🧠 Label classification evaluation (e.g., emotion/sentiment).
- 🧩 Joint label evaluation (e.g., emotion + intent).
- 📝 Text generation evaluation using BLEU, ROUGE, and BERTScore.
- 📤 Invalid sample detection and optional export.
- 📊 JSON and CSV result outputs for easy analysis and plotting.

---

## 🛠️ Installation

### Python Dependencies

Install required Python packages:

```bash
pip install scikit-learn nltk rouge-score bert-score pandas

## 🚀 Usage
```bash
python eval.py [classification | joint | generation] <arguments>

#EmoBench-M
#谷歌云盘链接
https://drive.google.com/file/d/16MAChQR2ASjL_gk24bGVnBxlV3ukoVoh/view
