# Import necessary libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# Reading the dataset :
training_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv")

tfidf_vectorizer = TfidfVectorizer()

# Assuming you have a dataset X (features) and y (target variable)
features = ["id", "sentence"]
target = ["difficulty"]

X = tfidf_vectorizer.fit_transform(training_data[features])

# Assuming 'difficulty' is the target variable
y = training_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
logreg_model = LogisticRegression()
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

# Train models
logreg_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
logreg_preds = logreg_model.predict(X_test)
knn_preds = knn_model.predict(X_test)
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluate performance metrics
logreg_precision = precision_score(y_test, logreg_preds)
logreg_recall = recall_score(y_test, logreg_preds)
logreg_f1 = f1_score(y_test, logreg_preds)
logreg_accuracy = accuracy_score(y_test, logreg_preds)

knn_precision = precision_score(y_test, knn_preds)
knn_recall = recall_score(y_test, knn_preds)
knn_f1 = f1_score(y_test, knn_preds)
knn_accuracy = accuracy_score(y_test, knn_preds)

dt_precision = precision_score(y_test, dt_preds)
dt_recall = recall_score(y_test, dt_preds)
dt_f1 = f1_score(y_test, dt_preds)
dt_accuracy = accuracy_score(y_test, dt_preds)

rf_precision = precision_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)

# Print performance metrics
print("Logistic Regression Metrics:")
print(f"Precision: {logreg_precision}")
print(f"Recall: {logreg_recall}")
print(f"F1-score: {logreg_f1}")
print(f"Accuracy: {logreg_accuracy}")
print()

print("k-Nearest Neighbors Metrics:")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")
print(f"F1-score: {knn_f1}")
print(f"Accuracy: {knn_accuracy}")
print()

print("Decision Tree Metrics:")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1-score: {dt_f1}")
print(f"Accuracy: {dt_accuracy}")
print()

print("Random Forest Metrics:")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1-score: {rf_f1}")
print(f"Accuracy: {rf_accuracy}")
print()

# Confusion Matrix
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, rf_preds))

# Examples of Erroneous Predictions
misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_test, rf_preds)) if true != pred]

print("\nExamples of Erroneous Predictions:")
for idx in misclassified_indices[:min(5, len(misclassified_indices))]:
    print(f"True: {y_test[idx]}, Predicted: {rf_preds[idx]}")

# Additional Analysis
# Add any additional analysis you'd like to perform to better understand your model's behavior.
