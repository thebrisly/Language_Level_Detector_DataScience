import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset

# Charger les données d'entraînement
train_data_url = "https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv"
train_data = pd.read_csv(train_data_url)

# Charger les données de test non étiquetées
test_data_url = "https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/unlabelled_test_data.csv"
test_data = pd.read_csv(test_data_url)

# Créer un modèle BERT pré-entraîné pour la classification de séquence
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_data['difficulty'].unique()))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokeniser les phrases
train_inputs = tokenizer(train_data['sentence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
test_inputs = tokenizer(test_data['sentence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

# Créer les ensembles de données PyTorch
train_labels = torch.tensor(pd.factorize(train_data['difficulty'])[0])
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)

# Diviser les données en ensembles d'entraînement et de validation
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

# Créer les DataLoader pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Définir les paramètres d'entraînement
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

# Entraîner le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = {key: val.to(device) for key, val in zip(['input_ids', 'attention_mask', 'labels'], batch)}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Évaluer le modèle sur l'ensemble de validation
model.eval()
val_preds, val_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {key: val.to(device) for key, val in zip(['input_ids', 'attention_mask', 'labels'], batch)}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        val_preds.extend(preds)
        val_labels.extend(labels)

# Calculer l'accuracy sur l'ensemble de validation
accuracy = accuracy_score(val_labels, val_preds)
print(f'Accuracy on validation set: {accuracy}')

# Faire des prédictions sur l'ensemble de test non étiqueté
model.eval()
test_preds = []

with torch.no_grad():
    for i in range(0, len(test_inputs['input_ids']), 8):
        batch = {key: val[i:i+8].to(device) for key, val in test_inputs.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)

# Ajouter les prédictions au DataFrame de test
test_data['difficulty'] = pd.Categorical(test_preds).rename_categories(range(len(train_data['difficulty'].unique())))
test_data[['id', 'difficulty']].to_csv('bert_results.csv', index=False)
