from functions import preprocessing, top_terms_per_component,train_random_forest,explain_with_lime,xgboost

def main():
    # Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, df_clean, vectorizer_text, vectorizer_title, svd_text, svd_title = preprocessing()
    
    # Costruzione feature names dai top terms SVD
    text_top_terms = top_terms_per_component(svd_text, vectorizer_text)
    title_top_terms = top_terms_per_component(svd_title, vectorizer_title)
    text_features = [f"TEXT_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(text_top_terms)]
    title_features = [f"TITLE_SVD{i+1}_({' '.join(words)})" for i, words in enumerate(title_top_terms)]
    feature_names = text_features + title_features
    
    best_model = xgboost(X_train, y_train, X_val, y_val,X_test,y_test)

    explain_with_lime(best_model,X_train,X_test,y_train,feature_names,output_file="lime_explanation_xgboost.html")

if __name__ == "__main__":
    main()