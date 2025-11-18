import os
import warnings
import torch
import pandas as pd
from transformers import RobertaTokenizer,Trainer,TrainingArguments, EarlyStoppingCallback,DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from scipy.stats import ttest_1samp 
from xgboost import XGBClassifier, callback as xgb_callback
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from functions import _evaluate_model_metrics
import numpy as np

def random_forest(X_train, y_train, X_val=None, y_val=None,X_test=None, y_test=None, model_path="random_forest_finale.pkl",nested_cv=True):

    param_grid = {
        'n_estimators': [100,200,300],
        'max_depth': [20,30,40],
        'min_samples_split': [2,3],
        'min_samples_leaf': [3,5],
        'max_features': [None] 
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    if nested_cv:
        print("Eseguo Nested Cross-Validation (Stima della Generalizzazione)")
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = 5

        grid_search = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        nested_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\nAccuratezze nested CV: {nested_scores}")
        print(f"Media nested CV: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    
        t_stat, p_value = ttest_1samp(nested_scores, 0.5)
        print(f"Test t (baseline = 0.5): t={t_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.05:
            print("Risultato: Differenza statisticamente significativa.")
        else:
            print("Risultato: Nessuna differenza significativa.")
    
    if os.path.exists(model_path):
        print("Carico modello salvato...")
        best_model = joblib.load(model_path)
    else:
        print("Addestramento e ottimizzazione iperparametri Random Forest (Grid Search)...")
        grid = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2 
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        print(f"\nIperparametri ottimali: {grid.best_params_}")
        joblib.dump(best_model, model_path)
        print(f"Modello salvato in {model_path}")
    _evaluate_model_metrics(best_model, X_train, y_train, "TRAIN (Ottimistico)")

    if X_val is not None and y_val is not None:
        _evaluate_model_metrics(best_model, X_val, y_val, "VALIDATION")

    if X_test is not None and y_test is not None:
        y_test_pred = _evaluate_model_metrics(best_model, X_test, y_test, "TEST (Finale)")
        cm = confusion_matrix(y_test, y_test_pred)
        cm_labels = ["Fake", "True"]
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Random Forest (TEST Set)")
        plt.show()

    return best_model

def xgboost(X_train, y_train, X_val, y_val, X_test, y_test,vectorizer_text=None, vectorizer_title=None,svd_text=None, svd_title=None,scaler_text=None, scaler_title=None,model_path="xgboost_gpu_finale.pkl",artifacts_path="xgboost_artifacts.pkl",nested_cv=True):
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    base_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    param_grid = {
        "max_depth": [3,4,5],
        "learning_rate": [0.02,0.05],
        "n_estimators": [200,300,400],
        "subsample": [0.8],
        "colsample_bytree": [0.7],
        "reg_alpha": [1.5],
        "reg_lambda": [3.0]
    }
    
    if nested_cv:
        print("Eseguo Nested Cross-Validation (Stima della Generalizzazione)")
        
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1
        )
        nested_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring="accuracy", n_jobs=-1)
        
        mean_acc = nested_scores.mean()
        std_acc = nested_scores.std()
        print(f"\nAccuratezze nested CV: {nested_scores}")
        print(f"Media nested CV: {mean_acc:.4f} ± {std_acc:.4f}")

        t_stat, p_value = ttest_1samp(nested_scores, 0.5)
        print(f"Test t (baseline = 0.5): t={t_stat:.4f}, p={p_value:.6f}")
        print("Risultato:", "Differenza significativa" if p_value < 0.05 else "Differenza NON significativa")
    
    if os.path.exists(model_path):
        print("\nCarico modello XGBoost salvato")
        best_model = joblib.load(model_path)
    else:
        print("\nAddestramento e ottimizzazione iperparametri XGBoost (Grid Search)")
        grid = GridSearchCV(base_model, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"\nIperparametri ottimali: {grid.best_params_}")
        joblib.dump(best_model, model_path)
        print(f"Modello XGBoost GPU salvato in {model_path}")
    
    print("Valutazione finale del modello XGBoost")
    _evaluate_model_metrics(best_model, X_train, y_train, "TRAIN (Ottimistico)")
    if X_val is not None and y_val is not None:
        _evaluate_model_metrics(best_model, X_val, y_val, "VALIDATION")
    if X_test is not None and y_test is not None:
        _evaluate_model_metrics(best_model, X_test, y_test, "TEST (Finale)")
    
    artifacts = {
        "model": best_model,
        "vectorizer_text": vectorizer_text,
        "vectorizer_title": vectorizer_title,
        "svd_text": svd_text,
        "svd_title": svd_title,
        "scaler_text": scaler_text,
        "scaler_title": scaler_title
    }
    
    joblib.dump(artifacts, artifacts_path)
    print(f"Tutti gli artifacts salvati in {artifacts_path}")
    
    return best_model

def compute_metrics_distilbert(eval_pred):
    logits, labels = eval_pred 
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='binary', zero_division=0)
    rec = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)

    return {
        'eval_accuracy': acc,
        'eval_precision': prec,
        'eval_recall': rec,
        'eval_f1': f1,
    }

def distilbert(model_path="distilbert_finale", n_splits_outer=2, n_splits_inner=2):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    if device == "cuda":
        torch.cuda.empty_cache()

    df = pd.read_csv("dataset.csv").dropna(subset=["title", "text", "Class"])
    df["label"] = df["Class"].str.lower().map({"fake": 0, "true": 1})
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    df["full_text"] = df["title"].astype(str) + " " + tokenizer.sep_token + " " + df["text"].astype(str)
    
    full_dataset = Dataset.from_pandas(df[["full_text", "label"]])
    
    def tokenize_function(examples):
        return tokenizer(examples["full_text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizzazione del dataset completo (DistilBERT)...")
    tokenized_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=["full_text"]) 
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    indices = np.arange(len(tokenized_dataset))
    labels = df["label"].values
    accuracies_nested_cv = []

    print(f"\nEseguo Nested Cross-Validation ({n_splits_outer}x{n_splits_inner})...")
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    learning_rates = [5e-6, 1e-5, 1.5e-5]
    best_lr_overall = None 
    best_acc_overall = 0

    for fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(indices, labels), 1):
        print(f"\n--- Outer Fold {fold}/{n_splits_outer} ---")
        train_val_indices = indices[train_val_idx]
        train_val_labels = labels[train_val_idx]
        
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=fold)
        best_acc_fold = -1
        best_lr_fold = None

        print("Inizio Tuning Iperparametri")
        for lr in learning_rates:
            inner_accs = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(train_val_indices, train_val_labels):
                inner_train_ds = tokenized_dataset.select(train_val_indices[inner_train_idx])
                inner_val_ds = tokenized_dataset.select(train_val_indices[inner_val_idx])

                model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

                training_args = TrainingArguments(
                    output_dir=f"./results_temp/fold_{fold}_lr_{lr}",
                    evaluation_strategy="epoch",
                    weight_decay=0.1,
                    save_strategy="epoch",
                    num_train_epochs=3, 
                    per_device_train_batch_size=16, 
                    per_device_eval_batch_size=16,
                    learning_rate=lr,
                    disable_tqdm=True,
                    load_best_model_at_end=True,
                    logging_steps=100,
                    max_grad_norm=1.0,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=inner_train_ds,
                    eval_dataset=inner_val_ds,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics_distilbert,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] 
                )

                trainer.train()
                eval_metrics = trainer.evaluate()
                inner_accs.append(eval_metrics["eval_accuracy"])
            
                del model 
                del trainer
                if device == "cuda":
                    torch.cuda.empty_cache()

            mean_acc = np.mean(inner_accs)
            print(f"LR {lr:.1e} - Media Inner Acc: {mean_acc:.4f}")
            
            if mean_acc > best_acc_fold:
                best_acc_fold = mean_acc
                best_lr_fold = lr
                
        print(f"Miglior LR per Fold {fold}: {best_lr_fold} (Acc: {best_acc_fold:.4f})")
        
        if best_acc_fold > best_acc_overall:
            best_acc_overall = best_acc_fold
            best_lr_overall = best_lr_fold

        final_train_ds = tokenized_dataset.select(train_val_indices)
        model_final_fold = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

        training_args_final_fold = TrainingArguments(
            output_dir=f"./final_fold_{fold}",
            evaluation_strategy="no",
            save_strategy="no",
            learning_rate=best_lr_fold,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            disable_tqdm=True,
            max_grad_norm=1.0,
            weight_decay=0.1
        )

        trainer_final_fold = Trainer(
            model=model_final_fold,
            args=training_args_final_fold,
            train_dataset=final_train_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_distilbert,
        )

        trainer_final_fold.train()
        
        test_ds = tokenized_dataset.select(test_idx)
        preds = trainer_final_fold.predict(test_ds)
        y_pred = np.argmax(preds.predictions, axis=-1)
        y_true = preds.label_ids

        acc = accuracy_score(y_true, y_pred)
        accuracies_nested_cv.append(acc)
        print(f"Accuracy TEST Fold {fold}: {acc:.4f}")
        
        del model_final_fold
        del trainer_final_fold
        if device == "cuda":
            torch.cuda.empty_cache()
            
    print("Risultati Nested Cross-Validation (NCV)")
 
    print("Accuratezze Nested CV:", np.round(accuracies_nested_cv, 4))
    print(f"Media Nested CV: {np.mean(accuracies_nested_cv):.4f} ± {np.std(accuracies_nested_cv):.4f}")

    t_stat, p_value = ttest_1samp(accuracies_nested_cv, 0.5)
    print(f"\nTest t (baseline = 0.5): t={t_stat:.4f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("Risultato: Differenza statisticamente significativa.")
    else:
        print("Risultato: Nessuna differenza significativa.")
        
    print("Addestramento finale sull'intero dataset (Train/Val Split)")
 

    print(f"Miglior Learning Rate (Media NCV): {best_lr_overall:.1e}")
    
    df_train_val, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

    df_train, df_val = train_test_split(df_train_val, test_size=0.15/(1-0.15), random_state=42, stratify=df_train_val["label"])

    train_ds_final = tokenized_dataset.select(df_train.index.values)
    val_ds_final = tokenized_dataset.select(df_val.index.values)
    test_ds_final = tokenized_dataset.select(df_test.index.values)

    model_final = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    training_args_final = TrainingArguments(
        output_dir="./distilbert_final_train",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_lr_overall if best_lr_overall else 2e-5, # Usa il miglior LR della NCV
        num_train_epochs=5, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.1,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=1,
        disable_tqdm=True,
        max_grad_norm=1.0
    )

    trainer_final = Trainer(
        model=model_final,
        args=training_args_final,
        train_dataset=train_ds_final,
        eval_dataset=val_ds_final,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_distilbert,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )

    trainer_final.train()
    trainer_final.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\nModello finale salvato in {model_path}")

    preds = trainer_final.predict(test_ds_final)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    print("\n" + "-"*40)
    print("Metriche finali su TEST (Modello migliore):")
    print("-"*40)
    target_names = ["Fake (0)", "True (1)"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "True"]).plot(cmap='Blues')
    plt.title("Confusion Matrix - DistilBERT Finale")
    plt.show()

    return model_final

