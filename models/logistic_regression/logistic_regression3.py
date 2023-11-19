import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the test data
df_test = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv")

# Extract the labels from the test data
y_test = df_test["difficulty"]

# Load the training data
df_train = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv")

# Separate the training data into features and labels
X_train = df_train["sentence"]
y_train = df_train["difficulty"]

# Transform the training data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Separate the test data into features
X_test = df_test["sentence"]

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test data
y_pred = model.predict(vectorizer.transform(X_test))

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
