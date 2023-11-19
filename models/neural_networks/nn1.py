import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Charger les données d'entraînement
url = "https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv"
data = pd.read_csv(url)

# Séparation des données en ensemble d'entraînement et ensemble de test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Encodage des niveaux de difficulté
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['difficulty'])
y_test = label_encoder.transform(test_data['difficulty'])

# Tokenisation des phrases
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['sentence'])
X_train = tokenizer.texts_to_sequences(train_data['sentence'])
X_test = tokenizer.texts_to_sequences(test_data['sentence'])

# Ajout de padding pour que toutes les séquences aient la même longueur
max_len = 50
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Création du modèle
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Définition des callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Chargement du meilleur modèle
model.load_weights('best_model.h5')

# Évaluation sur l'ensemble de test
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy: {accuracy}")

# Appliquer le modèle sur les données de test non étiquetées
unlabeled_data = pd.read_csv("https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv")
X_unlabeled = tokenizer.texts_to_sequences(unlabeled_data['sentence'])
X_unlabeled = pad_sequences(X_unlabeled, maxlen=max_len, padding='post')
unlabeled_predictions = model.predict(X_unlabeled)
unlabeled_data['difficulty'] = label_encoder.inverse_transform(np.round(unlabeled_predictions).astype(int))

# Sauvegarder les prédictions dans un fichier CSV
unlabeled_data[['id', 'difficulty']].to_csv("nn1_results.csv", index=False)
