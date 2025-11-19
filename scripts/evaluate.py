import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

with open('models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/processed/test.csv')

X_test = df.drop('target', axis=1)
y_test = df['target']

y_pred = model.predict(X_test)

# TODO: Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# TODO: Create metrics dictionary and save to JSON
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1)
}

Path('metrics').mkdir(exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Evaluation metrics: {metrics}")

