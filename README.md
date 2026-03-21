# Causal-BERT: A Causal Intervention Framework for Chinese LLM Safety Guardrails

[![Paper](https://img.shields.io/badge/Paper-ACM%20TOPS-blue)](https://dl.acm.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Request%20Required-orange)](DATA_REQUEST.md)

Official implementation of **"Beyond Spurious Correlations: A Causal Intervention Framework for Compositional Security in Chinese Large Language Models"** (ACM Transactions on Privacy and Security, 2025).

## 📋 Overview

This repository contains the implementation of **Causal-BERT**, a lightweight discriminative model for Chinese content safety detection that achieves robust compositional generalization through causal intervention.

### Key Results

| Metric | Causal-BERT | Vanilla BERT | DeBERTa-v3 |
|--------|-------------|--------------|------------|
| **OOD FPR** | **0.0%** | 13.3% | 6.7% |
| **Adversarial Acc** | **100%** | 80% | 91% |
| **Latency** | **12.5ms** | 12.5ms | 15.2ms |

## 🔧 Installation

```bash
git clone https://github.com/Qiujianmin/causal-guardrail.git
cd causal-guardrail
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training

```bash
python train.py \
    --model_name bert-base-chinese \
    --data_dir ./data \
    --output_dir ./outputs/causal-bert \
    --num_counterfactual_pairs 240 \
    --margin_gamma 1.0 \
    --alpha 0.5 \
    --beta 0.3 \
    --epochs 10 \
    --batch_size 32
```

### Inference

```python
from model import CausalBERTClassifier

classifier = CausalBERTClassifier.from_pretrained("./outputs/causal-bert")
result = classifier.predict("西藏是中国的一个自治区")
print(f"Label: {result.label}, Confidence: {result.confidence:.4f}")
```

## 📁 Repository Structure

```
causal-guardrail/
├── model/
│   ├── causal_bert.py          # Causal-BERT model implementation
│   ├── contrastive_loss.py     # Counterfactual Contrastive Learning loss
│   └── trainer.py              # Training pipeline
├── data/
│   ├── data_processor.py       # Data preprocessing
│   ├── counterfactual_generator.py  # Counterfactual pair generation
│   └── README.md               # Data format documentation
├── scripts/
│   ├── train.sh                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── run_ood_test.py         # OOD evaluation
├── configs/
│   └── default.yaml            # Default hyperparameters
├── requirements.txt
└── README.md
```

## 📊 Dataset: CCSD v2.0

The **Chinese Compositional Safety Dataset (CCSD) v2.0** contains 49,016 samples designed for evaluating compositional generalization in Chinese safety detection.

### Dataset Statistics

| Split | Total | Illicit | Benign | Counterfactual Pairs |
|-------|-------|---------|--------|---------------------|
| Train | 34,310 | 29,056 | 5,254 | 240 |
| Val | 7,352 | 6,232 | 1,120 | 0 |
| Test | 7,354 | 6,234 | 1,120 | 0 |

### Data Format

```json
{
  "id": "train_00001",
  "text": "西藏是中国的一个自治区",
  "label": 0,
  "entity": "西藏",
  "relation_type": "benign_administrative",
  "difficulty": "easy",
  "is_counterfactual": false
}
```

### ⚠️ Dataset Access

**Due to the sensitive nature of content moderation data involving geopolitical topics, the CCSD v2.0 dataset is available upon request for academic research purposes only.**

To request access:
1. Read the [Data Use Agreement](DATA_USE_AGREEMENT.md)
2. Fill out the [Data Request Form](https://forms.gle/YOUR-FORM-LINK)
3. Wait for approval (typically 3-5 business days)

## 🔬 Repproducing Results

### Main Results (Table 2)

```bash
# Train Causal-BERT
bash scripts/train.sh

# Evaluate on test set
python scripts/evaluate.py --model_path ./outputs/causal-bert
```

### OOD Evaluation (Table 4)

```bash
python scripts/run_ood_test.py --model_path ./outputs/causal-bert
```

### Adversarial Evaluation (Table 5)

```bash
python scripts/run_adversarial_test.py --model_path ./outputs/causal-bert
```

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{qiu2025causal,
  title={Beyond Spurious Correlations: A Causal Intervention Framework for Compositional Security in Chinese Large Language Models},
  author={Qiu, Jianmin and Han, Jinguang},
  journal={ACM Transactions on Privacy and Security},
  year={2025}
}
```

## 📄 License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Dataset**: CC BY-NC-SA 4.0 (for approved researchers only)

## 📧 Contact

For questions about the code or dataset access:
- **Qiu Jianmin**: 230239771@seu.edu.cn
- **Institution**: Southeast University, China

## 🙏 Acknowledgments

This work was supported by [funding sources if applicable].
