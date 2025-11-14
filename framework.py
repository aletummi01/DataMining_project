import streamlit as st
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functions import clean_text

def transform_input(title, text, vectorizer_text, vectorizer_title, svd_text, svd_title, scaler_text=None, scaler_title=None):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_t = clean_text(text)
    clean_title = clean_text(title)
    
    tokens_text = [lemmatizer.lemmatize(w) for w in word_tokenize(clean_t) if w not in stop_words]
    tokens_title = [lemmatizer.lemmatize(w) for w in word_tokenize(clean_title) if w not in stop_words]
    
    final_text = " ".join(tokens_text)
    final_title = " ".join(tokens_title)
    
    X_text_vec = vectorizer_text.transform([final_text])
    X_title_vec = vectorizer_title.transform([final_title])
    
    X_text_svd = svd_text.transform(X_text_vec)
    X_title_svd = svd_title.transform(X_title_vec)
    
    if scaler_text and scaler_title:
        X_text_scaled = np.clip(scaler_text.transform(X_text_svd), -3, 3)
        X_title_scaled = np.clip(scaler_title.transform(X_title_svd), -3, 3)
    else:
        X_text_scaled = X_text_svd
        X_title_scaled = X_title_svd
    
    X_input = np.hstack([X_text_scaled, X_title_scaled])
    return X_input

@st.cache_resource
def load_artifacts(path="xgboost_artifacts.pkl"):
    try:
        artifacts = joblib.load(path)
        return artifacts
    except FileNotFoundError:
        st.error(f"File artifacts non trovato: {path}. Il modello deve essere addestrato prima.")
        return None

def main():
    artifacts = load_artifacts()
    if artifacts is None:
        return  

    model = artifacts["model"]
    vectorizer_text = artifacts["vectorizer_text"]
    vectorizer_title = artifacts["vectorizer_title"]
    svd_text = artifacts["svd_text"]
    svd_title = artifacts["svd_title"]
    scaler_text = artifacts.get("scaler_text", None)
    scaler_title = artifacts.get("scaler_title", None)

    st.title("Classificatore Articoli (lingua inglese)")
    st.write("Inserisci titolo e testo dell'articolo che vuoi sapere se riporta una notizia reale o fake.")

    title_input = st.text_input("Titolo dell'articolo")
    text_input = st.text_area("Testo dell'articolo")

    if st.button("Classifica articolo"):
        if title_input.strip() == "" or text_input.strip() == "":
            st.warning("Inserisci sia il titolo sia il testo.")
        else:
            X_input = transform_input(title_input, text_input,
                                      vectorizer_text, vectorizer_title,
                                      svd_text, svd_title,
                                      scaler_text, scaler_title)
            
            pred = model.predict(X_input)[0]
            prob = np.max(model.predict_proba(X_input))
            
            if pred == 1:
                st.success(f"L'articolo è classificato come **REAL** (con probabilità: {prob:.2f})")
            else:
                st.error(f"L'articolo è classificato come **FAKE** (con probabilità: {prob:.2f})")

if __name__ == "__main__":
    main()
