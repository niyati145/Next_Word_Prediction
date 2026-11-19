import os
import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- File names (change here if yours are different) ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "lstm_model.h5")
MAX_LEN_PATH = str(BASE_DIR / "max_len.pkl")
TOKENIZER_PATH = str(BASE_DIR / "tokenizer.pkl")

st.set_page_config(page_title="Next Word Prediction", page_icon="🔮", layout="centered")


# ---------- Load everything once ----------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, int(max_len)


def clean_text(text):
    """Same cleaning used in training: lowercase, no punctuation."""
    import string
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def get_probs(model, tokenizer, text, max_len):
    seq = tokenizer.texts_to_sequences([clean_text(text)])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre", truncating="pre")
    probs = model.predict(seq, verbose=0)[0]
    probs[0] = 0.0  # index 0 is padding, never a real word
    return probs


def top_k_words(probs, tokenizer, k=5):
    idxs = np.argsort(probs)[::-1][:k]
    results = []
    for i in idxs:
        word = tokenizer.index_word.get(int(i))
        if word:
            results.append((word, float(probs[i])))
    return results


def generate(model, tokenizer, text, max_len, n_words, temperature):
    out = text.strip()
    for _ in range(n_words):
        probs = get_probs(model, tokenizer, out, max_len)
        if temperature <= 0.05:
            idx = int(np.argmax(probs))
        else:
            p = np.log(probs + 1e-9) / temperature
            p = np.exp(p - np.max(p))
            p /= p.sum()
            idx = int(np.random.choice(len(p), p=p))
        word = tokenizer.index_word.get(idx)
        if not word:
            break
        out += " " + word
    return out


# ---------- UI ----------
st.title("🔮 Next Word Prediction(LSTM)")
st.caption("LSTM model trained on a quotes dataset")

missing = [p for p in (MODEL_PATH, MAX_LEN_PATH, TOKENIZER_PATH) if not os.path.exists(p)]
if missing:
    st.error("These files were not found:")
    for p in missing:
        st.code(p)
    st.info("Files actually in the app folder:")
    st.code("\n".join(sorted(os.listdir(BASE_DIR))))
    st.write("Rename your files to match, or change the names at the top of `app.py`.")
    st.stop()

try:
    model, tokenizer, max_len = load_assets()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(f"Make sure `{MODEL_PATH}`, `{MAX_LEN_PATH}` and `{TOKENIZER_PATH}` "
            "are in the same folder as app.py.")
    st.stop()

tab1, tab2 = st.tabs(["Predict next word", "Generate text"])

with tab1:
    text = st.text_input("Type some words:", value="life is", key="predict_input")
    k = st.slider("Number of suggestions", 1, 10, 5)

    if st.button("Predict", type="primary", key="predict_btn"):
        if not text.strip():
            st.warning("Please type something first.")
        else:
            probs = get_probs(model, tokenizer, text, max_len)
            results = top_k_words(probs, tokenizer, k)
            if results:
                st.success(f"Most likely next word: **{results[0][0]}**")
                st.write("Top suggestions:")
                for word, p in results:
                    st.progress(min(p, 1.0), text=f"{word}  ({p:.1%})")
            else:
                st.warning("No prediction found for this input.")

with tab2:
    seed = st.text_input("Starting text:", value="life is", key="gen_input")
    n_words = st.slider("Words to generate", 1, 30, 10)
    temperature = st.slider(
        "Creativity (0 = most likely word, higher = more random)", 0.0, 1.5, 0.7, 0.1
    )

    if st.button("Generate", type="primary", key="gen_btn"):
        if not seed.strip():
            st.warning("Please type some starting text.")
        else:
            with st.spinner("Generating..."):
                result = generate(model, tokenizer, seed, max_len, n_words, temperature)
            st.write(result)

st.divider()
st.caption(f"Model input length: {max_len} words")