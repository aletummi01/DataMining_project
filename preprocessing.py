import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import gdown
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()                                # metto tutti in minuscole
    text = re.sub(r'\d+', '', text)                    # rimuovi numeri
    text = re.sub(r'[^\w\s]', '', text)                # rimuovi punteggiatura
    text = re.sub(r'\s+', ' ', text).strip()           # spazi multipli
    return text



def preprocessing():
    file_id = "1JuANqhW7-YJ90_yO8vA9hMk569onLu3X"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset"
    gdown.download(url, output, quiet=False)
    
    df = pd.read_csv(output)
    
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_text"] = df["text"].apply(clean_text)
    
    df["text_tokens"] = df["clean_text"].apply(word_tokenize)
    df = df.dropna(subset=['text', 'title', 'Class'])
    
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    
    df["text_tokens"] = df["text_tokens"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x if word not in stop_words]
    )
    df["final_text"] = df["text_tokens"].apply(lambda x: " ".join(x))
    
    df["title_tokens"] = df["clean_title"].apply(word_tokenize)
    df["title_tokens"] = df["title_tokens"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x if word not in stop_words]
    )
    df["final_title"] = df["title_tokens"].apply(lambda x: " ".join(x))
    df["Class"] = df["Class"].str.lower().map({"fake": 0, "true": 1})
    train_text, test_text, train_title, test_title, y_train, y_test = train_test_split(
        df["final_text"], df["final_title"], df["Class"],
        test_size=0.2, random_state=42, stratify=df["Class"]
    )
    
    vectorizer_text = TfidfVectorizer()
    X_train_text = vectorizer_text.fit_transform(train_text)
    X_test_text = vectorizer_text.transform(test_text)
    
    vectorizer_title = TfidfVectorizer()
    X_train_title = vectorizer_title.fit_transform(train_title)
    X_test_title = vectorizer_title.transform(test_title)
    
    svd_text = TruncatedSVD(n_components=100, random_state=42)
    X_train_text_svd = svd_text.fit_transform(X_train_text)
    X_test_text_svd = svd_text.transform(X_test_text)
    
    svd_title = TruncatedSVD(n_components=20, random_state=42)
    X_train_title_svd = svd_title.fit_transform(X_train_title)
    X_test_title_svd = svd_title.transform(X_test_title)
    
    scaler_text = StandardScaler()
    X_train_text_svd_scaled = scaler_text.fit_transform(X_train_text_svd)
    X_test_text_svd_scaled = scaler_text.transform(X_test_text_svd)
    
    scaler_title = StandardScaler()
    X_train_title_svd_scaled = scaler_title.fit_transform(X_train_title_svd)
    X_test_title_svd_scaled = scaler_title.transform(X_test_title_svd)
    
    X_train_text_svd_scaled = np.clip(X_train_text_svd_scaled, -3, 3)
    X_test_text_svd_scaled = np.clip(X_test_text_svd_scaled, -3, 3)
    X_train_title_svd_scaled = np.clip(X_train_title_svd_scaled, -3, 3)
    X_test_title_svd_scaled = np.clip(X_test_title_svd_scaled, -3, 3)
    
    X_train = np.hstack([X_train_text_svd_scaled, X_train_title_svd_scaled])
    X_test = np.hstack([X_test_text_svd_scaled, X_test_title_svd_scaled])
    
    return X_train, X_test, y_train, y_test, df[["final_title", "final_text", "Class"]], vectorizer_text, vectorizer_title, svd_text, svd_title


def perform_eda(df, text_column="final_text", class_column="Class", n_top_words=20, n_top_ngrams=10):    
    plt.figure(figsize=(6,4))
    sns.countplot(x=class_column, data=df)
    plt.title("Distribuzione delle classi")
    plt.show()
    #conto quanti dati ho per ogni classe
    print("\nConteggio per classe:")
    print(df[class_column].value_counts())
    df['num_words'] = df[text_column].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8,5))
    sns.histplot(df, x='num_words', hue=class_column, bins=50, kde=True)
    plt.title("Distribuzione lunghezza articoli (numero parole)")
    plt.show()
    
    print("\nStatistiche lunghezza articoli per classe:")
    print(df.groupby(class_column)['num_words'].describe())
    
    #Top parole per classe
    classes = df[class_column].unique()
    for cls in classes:
        words = " ".join(df[df[class_column]==cls][text_column]).split()
        most_common = Counter(words).most_common(n_top_words)
        print(f"\nTop {n_top_words} parole per classe {cls}:")
        print(most_common)
    
    #Wordcloud per classe
    plt.figure(figsize=(15,6))
    for i, cls in enumerate(classes):
        wc = WordCloud(width=800, height=400, background_color='white').generate(
            " ".join(df[df[class_column]==cls][text_column])
        )
        plt.subplot(1, len(classes), i+1)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Wordcloud {cls}")
    plt.show()
    
    #Top bigram/trigram
    for n in [2, 3]:
        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=1000)
        X = vectorizer.fit_transform(df[text_column])
        total_counts = X.toarray().sum(axis=0)
        ngram_counts = list(zip(vectorizer.get_feature_names_out(), total_counts))
        ngram_counts = sorted(ngram_counts, key=lambda x: x[1], reverse=True)
        print(f"\nTop {n_top_ngrams} {n}-gram nel dataset:")
        for ng, count in ngram_counts[:n_top_ngrams]:
            print(f"{ng}: {count}")
        ngram_df = pd.DataFrame(ngram_counts[:n_top_ngrams], columns=[f"{n}-gram", "Frequenza"])
        plt.figure(figsize=(10,6))
        sns.barplot(data=ngram_df, x="Frequenza", y=f"{n}-gram", palette="mako")
        plt.title(f"Top {n_top_ngrams} {n}-gram più frequenti", fontsize=14)
        plt.xlabel("Frequenza")
        plt.ylabel(f"{n}-gram")
        plt.tight_layout()
        plt.show()

def top_terms_per_component(svd, vectorizer, n_terms=5):
    terms = vectorizer.get_feature_names_out()
    components = svd.components_
    top_terms = []
    for i, comp in enumerate(components):
        top_indices = comp.argsort()[-n_terms:][::-1]
        top_terms.append([terms[idx] for idx in top_indices])
    return top_terms

