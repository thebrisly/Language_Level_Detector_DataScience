import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

# Assurez-vous d'avoir téléchargé la liste des stop words en français
nltk.download('stopwords')

# Chargement des données d'entraînement
train_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv")

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(train_data['sentence'], train_data['difficulty'], test_size=0.2, random_state=42)

# Obtention de la liste des stop words en français
stop_words = set(stopwords.words('french'))

# Extraction des caractéristiques avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialisation et entraînement du modèle Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Prédiction sur l'ensemble de test
y_pred = rf_classifier.predict(X_test_tfidf)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Chargement des données de test non étiquetées
unlabeled_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv")

# Extraction des caractéristiques avec TF-IDF
unlabeled_tfidf = tfidf_vectorizer.transform(unlabeled_data['sentence'])

# Prédiction sur les données de test non étiquetées
unlabeled_pred = rf_classifier.predict(unlabeled_tfidf)

# Création du fichier CSV avec les prédictions
output_df = pd.DataFrame({'id': unlabeled_data['id'], 'difficulty': unlabeled_pred})
output_df.to_csv('random_forest2_results.csv', index=False)
