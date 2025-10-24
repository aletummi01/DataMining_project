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

name_file='/Users/aletummi/Desktop/fake_news_dataset/Dataset.csv'
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

def preprocessing():#funzione di preprocessing del dataset
    name_file='/Users/aletummi/Desktop/fake_news_dataset/Dataset.csv'
    df=pd.read_csv(name_file)
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_tokens"] = df["clean_text"].apply(word_tokenize)
    df = df.dropna(subset=['text', 'title', 'Class'])
    stop_words = set(stopwords.words('english')) 
    df["text_tokens"] = df["text_tokens"].apply(lambda x: [word for word in x if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    df["text_tokens"] = df["text_tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    df["final_text"] = df["text_tokens"].apply(lambda x: " ".join(x))
    df["title_tokens"] = df["clean_title"].apply(word_tokenize)
    df["title_tokens"] = df["title_tokens"].apply(lambda x: [word for word in x if word not in stop_words])
    df["title_tokens"] = df["title_tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    df["final_title"] = df["title_tokens"].apply(lambda x: " ".join(x))
    vectorizer_text = TfidfVectorizer(max_features=15000)
    vectorizer_title = TfidfVectorizer(max_features=3000) 
    X_textfidf = vectorizer_text.fit_transform(df["final_text"])
    X_textfidf = normalize(X_textfidf)
    X_titlefidf = vectorizer_title.fit_transform(df["final_title"])
    X_titlefidf = normalize(X_titlefidf)
    y = df["Class"].values if "Class" in df.columns else None
    return X_textfidf, X_titlefidf, y, df[["final_title", "final_text", "Class"]], vectorizer_text, vectorizer_title


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

