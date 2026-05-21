import streamlit as st
import torch
import re

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Vietnamese News Summarization",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Vietnamese News Summarization")
st.write("Tóm tắt bài báo tiếng Việt bằng mô hình mT5")

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    model_path = "mt5_finance_summary_final"

    tokenizer = T5Tokenizer.from_pretrained(model_path)

    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    )

    return tokenizer, model


try:
    tokenizer, model = load_model()
    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Cannot load model: {e}")
    st.stop()

# ======================================================
# DEVICE
# ======================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

st.info(f"Running on: {device}")

# ======================================================
# CLEAN OUTPUT (FIX <extra_id_0>)
# ======================================================

def clean_summary(text):

    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# SUMMARIZATION FUNCTION
# ======================================================

def summarize_text(text):

    # ---------- SPLIT LONG ARTICLE ----------
    words = text.split()
    chunk_size = 350

    chunks = [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    partial_summaries = []

    # ---------- FIRST PASS ----------
    for chunk in chunks:

        prompt = "Tóm tắt bài báo tài chính sau:\n" + chunk

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            min_length=40,
            num_beams=6,
            length_penalty=1.5,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        summary = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        summary = clean_summary(summary)

        if len(summary) > 0:
            summary = summary[0].upper() + summary[1:]

        partial_summaries.append(summary)

    # ---------- SECOND PASS (VERY IMPORTANT) ----------
    combined_text = " ".join(partial_summaries)

    final_prompt = "Tóm tắt ngắn gọn nội dung sau:\n" + combined_text

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        min_length=50,
        num_beams=8,
        length_penalty=1.7,
        repetition_penalty=2.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    final_summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    final_summary = clean_summary(final_summary)

    if len(final_summary) > 0:
        final_summary = final_summary[0].upper() + final_summary[1:]

    return final_summary


# ======================================================
# INPUT AREA
# ======================================================

input_text = st.text_area(
    "📄 Nhập nội dung bài báo",
    height=300,
    placeholder="Dán nội dung bài báo tiếng Việt..."
)

# ======================================================
# BUTTON
# ======================================================

if st.button("🚀 Tóm tắt bài báo"):

    if len(input_text.strip()) == 0:

        st.warning("⚠️ Vui lòng nhập nội dung")

    else:

        with st.spinner("⏳ Đang tạo summary..."):

            summary = summarize_text(input_text)

        # ================= OUTPUT =================

        st.subheader("📝 Summary")
        st.write(summary)

        # ================= STATISTICS =================

        st.subheader("📊 Statistics")

        original_len = len(input_text.split())
        summary_len = len(summary.split())

        compression = 100 - (summary_len / original_len * 100)

        col1, col2, col3 = st.columns(3)

        col1.metric("Original Words", original_len)
        col2.metric("Summary Words", summary_len)
        col3.metric("Compression Rate", f"{compression:.1f}%")