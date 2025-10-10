import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
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

def preprocessing():
    name_file='/Users/aletummi/Desktop/fake_news_dataset/Dataset.csv'
    df=pd.read_csv(name_file)
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(word_tokenize)
    df = df.dropna(subset=['text', 'title', 'Class'])
    stop_words = set(stopwords.words('english')) 
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    df["tokens"] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    # ricompone i token in una stringa pulita (opzionale)
    df["final_text"] = df["tokens"].apply(lambda x: " ".join(x))
    vectorizer = TfidfVectorizer(max_features=45000)
    X_tfidf = vectorizer.fit_transform(df["final_text"]).toarray()
    X_tfidf = normalize(X_tfidf)
    y = df["Class"].values if "Class" in df.columns else None
    return X_tfidf, y, df[["title", "final_text", "Class"]], vectorizer


def perform_eda(df, text_column="final_text", class_column="Class", n_top_words=20, n_top_ngrams=10):    
    plt.figure(figsize=(6,4))
    sns.countplot(x=class_column, data=df)
    plt.title("Distribuzione delle classi")
    plt.show()

    print("\nConteggio per classe:")
    print(df[class_column].value_counts())
    
    # --- 2. Lunghezza articoli ---
    df['num_words'] = df[text_column].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8,5))
    sns.histplot(df, x='num_words', hue=class_column, bins=50, kde=True)
    plt.title("Distribuzione lunghezza articoli (numero parole)")
    plt.show()
    
    print("\nStatistiche lunghezza articoli per classe:")
    print(df.groupby(class_column)['num_words'].describe())
    
    # --- 3. Top parole per classe ---
    classes = df[class_column].unique()
    for cls in classes:
        words = " ".join(df[df[class_column]==cls][text_column]).split()
        most_common = Counter(words).most_common(n_top_words)
        print(f"\nTop {n_top_words} parole per classe {cls}:")
        print(most_common)
    
    # --- 4. Wordcloud per classe ---
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
    
    # --- 5. Top bigram/trigram ---
    for n in [2, 3]:
        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=1000)
        X = vectorizer.fit_transform(df[text_column])
        total_counts = X.toarray().sum(axis=0)
        ngram_counts = list(zip(vectorizer.get_feature_names_out(), total_counts))
        ngram_counts = sorted(ngram_counts, key=lambda x: x[1], reverse=True)
        
        # stampa testuale
        print(f"\nTop {n_top_ngrams} {n}-gram nel dataset:")
        for ng, count in ngram_counts[:n_top_ngrams]:
            print(f"{ng}: {count}")
        
        # --- creazione dataframe per il grafico ---
        ngram_df = pd.DataFrame(ngram_counts[:n_top_ngrams], columns=[f"{n}-gram", "Frequenza"])
        
        # --- barplot ---
        plt.figure(figsize=(10,6))
        sns.barplot(data=ngram_df, x="Frequenza", y=f"{n}-gram", palette="mako")
        plt.title(f"Top {n_top_ngrams} {n}-gram più frequenti", fontsize=14)
        plt.xlabel("Frequenza")
        plt.ylabel(f"{n}-gram")
        plt.tight_layout()
        plt.show()

X,y,df_clean,vectorizer = preprocessing()
perform_eda(df_clean)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)


