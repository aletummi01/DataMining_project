import os
import re
import joblib
import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from lime.lime_tabular import LimeTabularExplainer
from wordcloud import WordCloud
import warnings

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def top_terms_per_component(svd, vectorizer, n_terms=5):
    terms = vectorizer.get_feature_names_out()
    components = svd.components_
    top_terms = []
    for comp in components:
        top_indices = comp.argsort()[-n_terms:][::-1]
        top_terms.append([terms[idx] for idx in top_indices])
    return top_terms

def preprocessing(file_id="1JuANqhW7-YJ90_yO8vA9hMk569onLu3X", dataset_filename="dataset.csv",
                  test_size=0.2, val_size=0.2, random_state=42):
    # Scarica dataset se non esiste
    if not os.path.exists(dataset_filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Download dataset...")
        gdown.download(url, dataset_filename, quiet=False)
    
    df = pd.read_csv(dataset_filename)
    df = df.dropna(subset=['text', 'title', 'Class']).copy()
    
    # Pulizia testi
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_text"] = df["text"].apply(clean_text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenizzazione e lemmatizzazione
    df["text_tokens"] = df["clean_text"].apply(word_tokenize)
    df["text_tokens"] = df["text_tokens"].apply(lambda x: [lemmatizer.lemmatize(w) for w in x if w not in stop_words])
    df["final_text"] = df["text_tokens"].apply(lambda x: " ".join(x))
    
    df["title_tokens"] = df["clean_title"].apply(word_tokenize)
    df["title_tokens"] = df["title_tokens"].apply(lambda x: [lemmatizer.lemmatize(w) for w in x if w not in stop_words])
    df["final_title"] = df["title_tokens"].apply(lambda x: " ".join(x))
    
    # Classi 0/1
    df["Class"] = df["Class"].astype(str).str.lower().map({"fake":0, "true":1})
    
    # Train/Test split
    X_train_full_text, X_test_text, X_train_full_title, X_test_title, y_train_full, y_test = train_test_split(
        df["final_text"], df["final_title"], df["Class"], test_size=test_size, random_state=random_state, stratify=df["Class"]
    )
    
    # Validation split
    X_train_text, X_val_text, X_train_title, X_val_title, y_train, y_val = train_test_split(
        X_train_full_text, X_train_full_title, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )
    
    # TF-IDF
    vectorizer_text = TfidfVectorizer(max_features=20000)
    X_train_text_vec = vectorizer_text.fit_transform(X_train_text)
    X_val_text_vec = vectorizer_text.transform(X_val_text)
    X_test_text_vec = vectorizer_text.transform(X_test_text)
    
    vectorizer_title = TfidfVectorizer(max_features=5000)
    X_train_title_vec = vectorizer_title.fit_transform(X_train_title)
    X_val_title_vec = vectorizer_title.transform(X_val_title)
    X_test_title_vec = vectorizer_title.transform(X_test_title)
    
    # SVD
    svd_text = TruncatedSVD(n_components=min(100, X_train_text_vec.shape[1]-1), random_state=random_state)
    X_train_text_svd = svd_text.fit_transform(X_train_text_vec)
    X_val_text_svd = svd_text.transform(X_val_text_vec)
    X_test_text_svd = svd_text.transform(X_test_text_vec)
    
    svd_title = TruncatedSVD(n_components=min(20, X_train_title_vec.shape[1]-1), random_state=random_state)
    X_train_title_svd = svd_title.fit_transform(X_train_title_vec)
    X_val_title_svd = svd_title.transform(X_val_title_vec)
    X_test_title_svd = svd_title.transform(X_test_title_vec)
    
    # Scaling e clipping
    scaler_text = StandardScaler()
    X_train_text_scaled = np.clip(scaler_text.fit_transform(X_train_text_svd), -3, 3)
    X_val_text_scaled = np.clip(scaler_text.transform(X_val_text_svd), -3, 3)
    X_test_text_scaled = np.clip(scaler_text.transform(X_test_text_svd), -3, 3)
    
    scaler_title = StandardScaler()
    X_train_title_scaled = np.clip(scaler_title.fit_transform(X_train_title_svd), -3, 3)
    X_val_title_scaled = np.clip(scaler_title.transform(X_val_title_svd), -3, 3)
    X_test_title_scaled = np.clip(scaler_title.transform(X_test_title_svd), -3, 3)
    
    # Stack finale
    X_train = np.hstack([X_train_text_scaled, X_train_title_scaled])
    X_val = np.hstack([X_val_text_scaled, X_val_title_scaled])
    X_test = np.hstack([X_test_text_scaled, X_test_title_scaled])
    
    df_clean = df[["final_title","final_text","Class"]].reset_index(drop=True)
    
    return X_train, X_val, X_test, y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True), df_clean, vectorizer_text, vectorizer_title, svd_text, svd_title

def perform_eda(df, text_column="final_text", class_column="Class", n_top_words=20, n_top_ngrams=10):
    plt.figure(figsize=(6,4))
    sns.countplot(x=class_column,data=df)
    plt.title("Distribuzione delle classi")
    plt.show()
    
    print("\nConteggio per classe:")
    print(df[class_column].value_counts())
    
    df['num_words'] = df[text_column].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8,5))
    sns.histplot(df,x='num_words',hue=class_column,bins=50,kde=True)
    plt.title("Distribuzione lunghezza articoli (numero parole)")
    plt.show()
    
    print("\nStatistiche lunghezza articoli per classe:")
    print(df.groupby(class_column)['num_words'].describe())
    
    # Top parole per classe
    classes = sorted(df[class_column].unique())
    for cls in classes:
        words = " ".join(df[df[class_column]==cls][text_column]).split()
        most_common = Counter(words).most_common(n_top_words)
        print(f"\nTop {n_top_words} parole per classe {cls}:")
        print(most_common)
    
    # Wordcloud
    plt.figure(figsize=(15,6))
    for i, cls in enumerate(classes):
        wc = WordCloud(width=800, height=400, background_color='white').generate(
            " ".join(df[df[class_column]==cls][text_column])
        )
        plt.subplot(1,len(classes),i+1)
        plt.imshow(wc,interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Wordcloud {cls}")
    plt.show()


def explain_with_lime(model, X_train, X_test, y_train, feature_names, sample_idx=0, output_file="lime_explanation.html"):
    X_train_dense = np.array(X_train)
    X_test_dense = np.array(X_test)
    
    explainer = LimeTabularExplainer(
        training_data=X_train_dense[:1000],
        feature_names=feature_names,
        class_names=[str(c) for c in np.unique(y_train)],
        discretize_continuous=True
    )
    
    exp = explainer.explain_instance(
        data_row=X_test_dense[int(sample_idx)],
        predict_fn=model.predict_proba,
        num_features=20
    )
    exp.save_to_file(output_file)
    print(f"Spiegazione LIME salvata in '{output_file}'")

def train_random_forest(X_train, y_train, X_val=None, y_val=None,
                        X_test=None, y_test=None, model_path="random_forest_finale.pkl"):
    if os.path.exists(model_path):
        print("Carico modello salvato...")
        best_model = joblib.load(model_path)
    else:
        print("Addestramento modello Random Forest...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators':[300],
            'max_depth':[13],
            'min_samples_split':[7],
            'min_samples_leaf':[3],
            'max_features':[None]
        }
        grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        joblib.dump(best_model, model_path)
        print(f"Modello salvato in {model_path}")
    
    # Accuracy train
    y_train_pred = best_model.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy TRAIN: {acc_train:.4f}")
    
    # Accuracy validation
    if X_val is not None and y_val is not None:
        y_val_pred = best_model.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)
        print(f"Accuracy VALIDATION: {acc_val:.4f}")
    
    # Accuracy test
    if X_test is not None and y_test is not None:
        y_test_pred = best_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        print(f"Accuracy TEST: {acc_test:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    cm_labels = ["Fake", "True"]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()


    return best_model

def xgboost(X_train, y_train, X_val, y_val, X_test, y_test, model_path="xgboost_gpu_finale.pkl"):
    warnings.filterwarnings("ignore", category=UserWarning)
    
    if os.path.exists(model_path):
        print("Carico modello XGBoost salvato...")
        best_model = joblib.load(model_path)
    else:
        print("Addestramento XGBoost su GPU con early stopping...")

        best_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist',    
            device='cuda',             
            n_estimators=300,          # numero di alberi
            max_depth=3,               # profondità moderata per ridurre overfitting
            learning_rate=0.02,        # step di apprendimento
            subsample=0.8,             # riduce overfitting
            colsample_bytree=0.7,      # riduce overfitting
            reg_alpha=1.5,             
            reg_lambda=3.0,
            random_state=42               
        )

        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        joblib.dump(best_model, model_path)
        print(f"Modello XGBoost GPU salvato in {model_path}")

    # Predizioni
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Accuracy
    print("\nRisultati Finali:")
    print(f"Accuracy TRAIN: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Accuracy VAL:   {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Accuracy TEST:  {accuracy_score(y_test, y_test_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - XGBoost GPU")
    plt.show()

    return best_model