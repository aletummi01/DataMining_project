from functions import preprocessing, top_terms_per_component,explain_with_lime
from models import random_forest,xgboost,distilbert
import pandas as pd

def main():
    # Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, df_clean, vectorizer_text, vectorizer_title, svd_text, svd_title,scaler_text,scaler_title = preprocessing()
    
    # Costruzione feature names dai top terms SVD
    text_top_terms = top_terms_per_component(svd_text, vectorizer_text)
    title_top_terms = top_terms_per_component(svd_title, vectorizer_title)
    text_features = [f"TEXT_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(text_top_terms)]
    title_features = [f"TITLE_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(title_top_terms)]
    feature_names = text_features + title_features
    best_model = random_forest(X_train, y_train, X_val, y_val,X_test,y_test)
    #best_model = xgboost(X_train, y_train, X_val, y_val,X_test,y_test,vectorizer_text,vectorizer_title,svd_text,svd_title,scaler_text,scaler_title)
    # Carica dataset pulito (deve avere 'title', 'text', 'label')
    df = pd.read_csv("dataset.csv")
    model_path="rfr_finale"
    # Suddivisione in train, validation, test
    #df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Class"])
    #df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp["Class"])

    #best_model = distilbert(model_path=model_path)  # df passato alla funzione di training

if __name__ == "__main__":
    main()