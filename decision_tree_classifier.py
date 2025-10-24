from preprocessing import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import matplotlib.pyplot as plt

X_title,X_text, y, df_clean, vectorizer_text, vectorizer_title = preprocessing()
X = hstack([X_title, X_text])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42, criterion='entropy')

param_grid = {
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print("Migliori parametri trovati:", grid_search.best_params_)

y_pred = best_clf.predict(X_test)

print("Accuracy sul test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=best_clf.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matrice di confusione")
plt.show()

plt.figure(figsize=(20,10))
feature_names = list(vectorizer_title.get_feature_names_out()) + list(vectorizer_text.get_feature_names_out())

plot_tree(
    best_clf,
    max_depth=3,
    feature_names=feature_names,
    class_names=best_clf.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Ottimizzato")
plt.show()



