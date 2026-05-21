# 🇻🇳 Vietnamese News Summarization Using Natural Language Processing

## 📌 Project Overview

This project builds an **Automatic Vietnamese News Summarization System** using modern **Natural Language Processing (NLP)** techniques and Transformer-based deep learning models.

The system takes a Vietnamese news article as input and generates a concise and meaningful summary automatically.

---

## 🎯 Objectives

* Apply NLP techniques to Vietnamese text processing
* Use pretrained Transformer models for text summarization
* Build an end-to-end summarization pipeline
* Deploy a runnable summarization application

---

## 🧠 Technologies Used

* Python 3.x
* Transformers
* PyTorch
* HuggingFace Transformers
* SentencePiece Tokenizer

---

## 🤖 Model

This project uses pretrained Transformer models:

* **mT5 (Multilingual Text-to-Text Transfer Transformer)**
* **ViT5 Vietnamese Summarization Model**

⚠️ Model weights are **not included** in this repository due to GitHub file size limitations.

Models are automatically downloaded from HuggingFace when running the application.

---

## 📂 Project Structure

```
bigproject/
│
├── app.py                 # Main application
├── README.md
├── requirements.txt
├── .gitignore
└── vit5_finance_summary_final/   # (ignored model directory)
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/anvip243123456-prog/VIETNAMESE-NEWS-SUMMARIZATION-USING-NATURAL-LANGUAGE-PROCESSING.git
cd VIETNAMESE-NEWS-SUMMARIZATION-USING-NATURAL-LANGUAGE-PROCESSING
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python app.py
```

The model will automatically download during the first run.

---

## 🧪 Example

**Input**

```
Việt Nam đạt tăng trưởng kinh tế ấn tượng trong quý đầu năm...
```

**Output**

```
Kinh tế Việt Nam tăng trưởng mạnh trong quý đầu năm.
```

---

## 🚀 Features

* Vietnamese text preprocessing
* Transformer-based summarization
* Automatic model loading
* Easy local deployment

---

## ⚠️ Notes

* Large model files (`.bin`, `.pt`, `.safetensors`) are excluded using `.gitignore`.
* First execution may take time because pretrained models are downloaded automatically.

---

## 📚 Future Improvements

* Web interface (Flask / Streamlit)
* REST API deployment
* Model fine-tuning with larger datasets
* Performance optimization

---

## 👨‍💻 Author

**Hải An**
**Quốc Thịn**
**Sang**

Natural Language Processing Project
University Coursework

---

## ⭐ License

This project is for educational and research purposes.
