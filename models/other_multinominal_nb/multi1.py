import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# Charger les données d'entraînement
url = "https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv"
data = pd.read_csv(url)

# Séparation des données en ensemble d'entraînement et ensemble de test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extraction des caractéristiques du texte avec TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['sentence'])
X_test = vectorizer.transform(test_data['sentence'])

# Encodage des niveaux de difficulté
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['difficulty'])
y_test = label_encoder.transform(test_data['difficulty'])

# Construire le modèle de classification
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_data['sentence'], y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(test_data['sentence'])

# Calculer l'accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Afficher le rapport de classification
print(classification_report(y_test, predictions))

# Appliquer le modèle sur les données de test non étiquetées
unlabeled_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv")
unlabeled_predictions = model.predict(unlabeled_data['sentence'])
unlabeled_data['difficulty'] = label_encoder.inverse_transform(unlabeled_predictions)

# Sauvegarder les prédictions dans un fichier CSV
unlabeled_data[['id', 'difficulty']].to_csv("predictions.csv", index=False)
