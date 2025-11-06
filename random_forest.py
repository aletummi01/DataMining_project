import os
from sklearn.tree import export_graphviz
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing
from preprocessing import top_terms_per_component
from lime.lime_tabular import LimeTabularExplainer


MODEL_PATH = "random_forest_finale.pkl"
last_y_test, last_y_pred, last_best_model = None, None, None

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

if not model_loaded:
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': [None]
    }

    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    acc_train = accuracy_score(y_train, best_model.predict(X_train))
    acc_test = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy su TRAIN: {acc_train:.4f}")
    print(f"Accuracy su TEST: {acc_test:.4f}")

    last_y_test, last_y_pred, last_best_model = y_test, y_pred, best_model

    joblib.dump(last_best_model, MODEL_PATH)
    print(f"\n Modello salvato in {MODEL_PATH}")

else:
    best_model = last_best_model

print("\n Genero spiegazioni LIME")

if hasattr(X_test, "toarray"):
    X_test_dense = X_test.toarray()
else:
    X_test_dense = np.array(X_test)

X_sample = X_test_dense[:200]
y_sample = y_test[:200]

explainer = LimeTabularExplainer(
    training_data=X_sample,
    feature_names=feature_names,
    class_names=np.unique(y_train).astype(str),
    discretize_continuous=True
)

i = 0
exp = explainer.explain_instance(
    data_row=X_sample[i],
    predict_fn=best_model.predict_proba,
    num_features=20
)

exp.save_to_file("lime_explanation.html")
print("Spiegazione LIME salvata in 'lime_explanation.html'")
