import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# TODO: For Part 2 (pipeline) load hyperparameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# TODO: Fill in hyperparameters
n_estimators = params['train']['n_estimators']
max_depth = params['train']['max_depth']
random_state = params['train']['random_state']

# TODO: Set the data path (different for Part 1 vs Part 2)
data_path = 'data/processed/train.csv'

df = pd.read_csv(data_path)

X_train = df.drop('target', axis=1)
y_train = df['target']

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=random_state
)

model.fit(X_train, y_train)

Path('models').mkdir(exist_ok=True)
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Trained model with n_estimators={n_estimators}, max_depth={max_depth}")
