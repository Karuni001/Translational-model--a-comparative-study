# Translational-model--a-comparative-study
# Translation-model--Comparitive-study
# 🌐 Hindi-to-English Poetry Translation – Comparative Study Models

This project presents a comparative analysis of four advanced neural machine translation (NMT) models for translating **Hindi poetry into English**. The goal is to explore the effectiveness and performance of multiple architectures in capturing linguistic nuances and poetic expressions.

## 🧠 Models Implemented

1. **Seq2Seq with Attention**
2. **mBART (Multilingual BART)**
3. **Standard NMT Model (Vanilla Encoder-Decoder)**
4. **Hybrid GPT-LSTM Model**

Each model was trained, evaluated, and compared based on translation quality, BLEU score, accuracy, and confusion matrix results.

---

## 📁 Project Structure

<pre> ``` Translation_project/ 
  ├── data_preprocessing.py 
  ├── seq2seq_model.py 
  ├── mbart_model.py 
  ├── nmt_model.py 
  ├── gpt_lstm_model.py
  ├── evaluation_metrics.py 
  ├── requirements.txt 
  ├── README.md 
  └── results_comparison.ipynb ``` </pre>


---

## 📊 Evaluation Metrics

| Model                | BLEU Score | Accuracy | Observations                          |
|---------------------|------------|----------|---------------------------------------|
| Seq2Seq + Attention | XX.XX      | XX.XX%   | Strong baseline                       |
| mBART                | XX.XX      | XX.XX%   | Best performance on poetic fluency    |
| Vanilla NMT          | XX.XX      | XX.XX%   | Decent performance, fast to train     |
| GPT + LSTM          | XX.XX      | XX.XX%   | Balanced generative and contextual    |

📌 *Exact values filled after model runs and results logging.*

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

Includes:

TensorFlow

PyTorch

Transformers (HuggingFace)

NumPy, Pandas, Matplotlib

scikit-learn

sacrebleu

```
## 🚀 How to Run
**1. Preprocess the dataset:**

```bash-

python data_preprocessing.py
```
**2. Train each model:**

```bash

python seq2seq_model.py

python mbart_model.py

python nmt_model.py

python gpt_lstm_model.py
```
**3. Evaluate and compare results:**

```bash

python evaluation_metrics.py

Visual comparison (Jupyter):

Open results_comparison.ipynb for detailed metrics, charts, and insights.
```

## ✍️ Author
**Karuna Gupta**
**Final Year B.Tech (AIML), Galgotias University**
**Email: karuna.gupta1103@gmail.com**
**GitHub: Karuni001**

## 📌 Note
This project is part of a comparative research study and intended for educational, research, and exploratory purposes only.
