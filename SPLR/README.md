# SPLR: Single-Point and Length Representation Model for Nested NER in Historical Texts

## Project Overview

This repository implements **SPLR** (Single-Point and Length Representation), a novel model for nested and polysemous named entity recognition (NER) in Chinese historical texts. It is especially suitable for ancient literature and domain-specific corpora. SPLR introduces a left-boundary plus length representation, dramatically improving the automatic extraction of complex entities and supporting external knowledge augmentation for large language models (LLMs).

---

## Background

The construction of historical knowledge bases has undergone three main stages:

1. **Data-Driven Information Extraction**  
   Using deep learning to extract core entities (e.g., person names, place names, official titles).

2. **Ontology-Guided Semantic Modeling**  
   Incorporating expert knowledge and ontology structures to model complex semantic relations.

3. **LLM-Enabled Automatic Knowledge Generation**  
   Using large language models for scalable, automated knowledge graph construction.

Despite this progress, most methods still struggle with nested/polysemous entity recognition and deep attribute extraction, often requiring extensive manual annotation. SPLR directly addresses these challenges with an innovative model design.

---

## Key Features & Innovations

- **Unified Boundary and Length Representation:**  
  Each entity is uniquely determined by its left boundary and length, enabling natural handling of nested and polysemous entities.

- **Convolutional Fusion of Length Information:**  
  CNN layers allow the model to distinguish entities of different lengths/spans.

- **Joint Decoding Strategy:**  
  Predicts entity category and boundary simultaneously for higher efficiency and accuracy.

- **Cost-Sensitive Learning for Imbalanced Data:**  
  Adaptive loss weighting improves learning for rare entity types and long-tail distributions.

---

## Directory Structure
```text
SPLR/
├── SPLR_nested_ner/                     #
│   ├── __init__.py
│   ├── dataset.py                 
│   ├── eval.py              
│   ├── model.py          
│   ├── SPLR_data.py       
│   └── utils.py              # Utility functions
│
├── main.py # Main entry for training and evaluation
├── .gitignore 
├── config.yaml
├── LICENSE
│
├── data
│   ├── SPLRtext
│   ├── 科举_train.jsonl                #
│   ├── 科举_test.jsonl 
│
└── README.md 
```


---
## Configuration

All file paths and major hyperparameters are set in `config.yaml`:

```yaml
data:
  train_path: data/科举_train.jsonl
  val_path: data/科举_val.jsonl


model:
  backbone: SIKU-BERT/sikubert

train:
  epochs: 30
  batch_size: 1
  lr: 0.0001
  seed: 42
  device: auto

eval:
  metrics: [f1, precision, recall]
  eval_interval: 1

logging:
  log_level: INFO
  log_dir: ./logs
```


## Quick Start

### 1. Install Dependencies

Python 3.8+ is recommended. Install required libraries:
### 2. Prepare Data

Ensure your training and validation datasets are in .jsonl format and referenced in config.yaml (default path: SPLR/data/).
### 3. Train & Evaluate
Run the main script:
```bash
python SPLR/main.py
```
Model checkpoints and logs will be saved as specified in the config file.

## Module Overview
| Module         | Description                                            |
| -------------- | ------------------------------------------------------ |
| `main.py`      | Main entry; orchestrates loading, training, evaluation |
| `SPLR_data.py` | Data reading & type-index mapping                      |
| `dataset.py`   | Custom PyTorch Dataset & batch preparation             |
| `model.py`     | SPLR model architecture, BERT backbone                 |
| `eval.py`      | Evaluation and metrics (e.g., F1, precision)           |
| `utils.py`     | Helper functions for training & evaluation             |

## Example Workflow
1. Prepare your data files and edit config.yaml as needed.
2. Install dependencies.
3. Run:
```bash
python SPLR/main.py
```
4. The model will train, evaluate each epoch, and save results.
## Citation
If you use this project in your research, please cite our related papers or this repository.
``` bibtex
@misc{splr_ner_2025,
  author = {Hangqin Lu},
  title = {SPLR: Single-Point and Length Representation Model for Nested NER },
  year = {2025},
  howpublished = {\url{https://github.com/HangqinLu/splr-ner}}
}
```
## License

``` markdown
MIT License, see LICENSE for details.

Copyright (c) 2025 Hangqin Lu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---





