# logistic_regression_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder


### Functions 

def train_logistic_regression_model(X_train, y_train):
    """
    Trains a logistic regression model on the given training data.

    Parameters:
    - X_train: Features for training
    - y_train: Target variable for training

    Returns:
    - Trained logistic regression model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_logistic_regression_model(model, X_test, y_test):
    """
    Evaluates a logistic regression model on the given test data and prints performance metrics.

    Parameters:
    - model: Trained logistic regression model
    - X_test: Features for testing
    - y_test: Target variable for testing
    """
    # Make predictions
    preds = model.predict(X_test)

    # Evaluate performance metrics
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    accuracy = accuracy_score(y_test, preds)

    # Print performance metrics
    print("Logistic Regression Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Accuracy: {accuracy}")

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


def apply_logistic_regression_model(model, tfidf_vectorizer, unlabelled_data, label_encoder):
    """
    Applies the pretrained logistic regression model to unlabelled data and saves the results.

    Parameters:
    - model: Trained logistic regression model
    - tfidf_vectorizer: TF-IDF vectorizer used for training
    - unlabelled_data: Unlabelled test data
    - label_encoder: LabelEncoder used for encoding 'difficulty'

    Returns:
    - DataFrame with predictions
    """
    # Use the same vectorizer to transform the unlabelled data
    X_unlabelled = tfidf_vectorizer.transform(unlabelled_data['sentence'])

    # Make predictions
    predictions = model.predict(X_unlabelled)

    # Decode numerical labels back to original string labels
    difficulty_labels = label_encoder.inverse_transform(predictions)

    # Add predictions to the unlabelled data
    unlabelled_data['difficulty'] = difficulty_labels

    return unlabelled_data[['id', 'difficulty']]


### Main code

def main():
    # Reading the dataset
    training_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv")

    # 'difficulty' is the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(training_data['difficulty'])

    # Using TF-IDF for text vectorization
    tfidf_vectorizer = TfidfVectorizer()
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
    result_data.to_csv("logistic_regression1_results.csv", index=False)


if __name__ == "__main__":
    main()
