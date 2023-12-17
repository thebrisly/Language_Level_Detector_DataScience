import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Définir batch_size, learning_rate, num_epochs
batch_size = 8  # Choisissez la taille de lot appropriée en fonction de la mémoire de votre GPU
learning_rate = 2e-5  # Vous pouvez ajuster cette valeur en fonction de votre expérience ou utiliser des techniques d'optimisation
num_epochs = 3  # Vous pouvez ajuster ce nombre en fonction de vos besoins

# Charger les données
url = "https://raw.githubusercontent.com/thebrisly/UNIL_Geneva_DSML/main/data/training_data.csv"
data = pd.read_csv(url)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Définir le modèle BERT
num_classes = 6
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokeniser les phrases
train_inputs = tokenizer(train_data['sentence'].tolist(), padding=True, truncation=True, return_tensors='pt')
test_inputs = tokenizer(test_data['sentence'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Définir les labels
difficulty_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
train_labels = torch.tensor([difficulty_mapping[label] for label in train_data['difficulty']])
test_labels = torch.tensor([difficulty_mapping[label] for label in test_data['difficulty']])

# Créer un DataLoader pour les données d'entraînement et de test
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définir l'optimiseur et la fonction de perte
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Entraîner le modèle
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Évaluer le modèle
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculer la précision
accuracy = accuracy_score(all_labels, all_predictions)
print(f'Accuracy: {accuracy}')
