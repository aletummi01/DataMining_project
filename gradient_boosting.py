import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from lime.lime_tabular import LimeTabularExplainer
from preprocessing import preprocessing, top_terms_per_component


MODEL_PATH = "gradient_boosting_finale.pkl"
last_y_test, last_y_pred, last_best_model = None, None, None

# --- Caricamento / Addestramento modello ---
if os.path.exists(MODEL_PATH):
    print(" Carico il modello salvato...")
    last_best_model = joblib.load(MODEL_PATH)
    model_loaded = True
else:
    print(" Nessun modello salvato trovato: avvio addestramento...")
    model_loaded = False

X_train, X_test, y_train, y_test,df_clean, vectorizer_text, vectorizer_title,svd_text, svd_title = preprocessing()

text_top_terms = top_terms_per_component(svd_text, vectorizer_text)
title_top_terms = top_terms_per_component(svd_title, vectorizer_title)

text_feature_names = [f"TEXT_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(text_top_terms)]
title_feature_names = [f"TITLE_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(title_top_terms)]
feature_names = text_feature_names + title_feature_names

print("\nEsempio di nomi di feature:")
print(feature_names[:10])

if not model_loaded:
    print("\n Addestro il modello Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    gb.fit(X_train, y_train)
    joblib.dump(gb, MODEL_PATH)
    print(f"\n Modello salvato in {MODEL_PATH}")

else:
    gb = last_best_model

y_pred = gb.predict(X_test)
acc_train = accuracy_score(y_train, gb.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print(f"\nAccuracy su TRAIN: {acc_train:.4f}")
print(f"Accuracy su TEST : {acc_test:.4f}\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fake", "True"],
            yticklabels=["Fake", "True"])
plt.title("Matrice di Confusione - Gradient Boosting")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.tight_layout()
plt.show()

print("\n Genero spiegazioni LIME...")

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=np.unique(y_train).astype(str),
    mode="classification"
)

sample_idx = 0
sample = X_test[sample_idx].reshape(1, -1)
predizione = gb.predict(sample)[0]
print(f"\nPredizione per il campione {sample_idx}: {predizione}")

exp = explainer.explain_instance(
    data_row=X_test[sample_idx],
    predict_fn=gb.predict_proba,
    num_features=10
)

print("\nSpiegazione LIME:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.3f}")

exp.save_to_file("lime_explanation_gradient_boosting.html")
print("\nSpiegazione LIME salvata in 'lime_explanation_gradient_boosting.html'")
