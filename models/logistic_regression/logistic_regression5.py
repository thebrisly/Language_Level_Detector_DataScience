from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_logistic_regression_model(X_train, y_train):
    model = LogisticRegression()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_logistic_regression_model(model, X_test, y_test):
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    accuracy = accuracy_score(y_test, preds)
    
    print("Logistic Regression Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Accuracy: {accuracy}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def apply_logistic_regression_model(model, tfidf_vectorizer, unlabelled_data, label_encoder):
    X_unlabelled = tfidf_vectorizer.transform(unlabelled_data['sentence'])
    predictions = model.predict(X_unlabelled)
    difficulty_labels = label_encoder.inverse_transform(predictions)
    unlabelled_data['difficulty'] = difficulty_labels
    return unlabelled_data[['id', 'difficulty']]

def main():
    # Reading the dataset
    training_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv")

    # 'difficulty' is the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(training_data['difficulty'])

    # Using TF-IDF for text vectorization
    tfidf_vectorizer = CountVectorizer()
    X = tfidf_vectorizer.fit_transform(training_data['sentence'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    logreg_model = train_logistic_regression_model(X_train, y_train)

    # Evaluate logistic regression model
    evaluate_logistic_regression_model(logreg_model, X_test, y_test)

    # Reading the unlabelled test data
    unlabelled_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv")

    # Apply the logistic regression model to unlabelled data
    result_data = apply_logistic_regression_model(logreg_model, tfidf_vectorizer, unlabelled_data, label_encoder)

    # Save the result in the desired output format
    result_data.to_csv("logistic_regression5_results.csv", index=False)

if __name__ == "__main__":
    main()
