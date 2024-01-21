import os
import warnings
import numpy as np
import pandas as pd
import os.path as op
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Training with batch processing
batch_size = 100
num_epochs = 10 

# setup
root = "C:/Users/jason/github/personal/healthy-brain-activity-classification/data/"
eeg_path = op.join(root, 'train_eegs')

# read metadata and get ids
meta = pd.read_csv(op.join(root, "train.csv"))
ids = np.unique(meta.eeg_id)

def load_data(eeg_path, idx, meta):
  
  y_cols = [
    'expert_consensus'
    # 'seizure_vote', 
    # 'lpd_vote', 
    # 'gpd_vote', 
    # 'lrda_vote', 
    # 'grda_vote', 
    # 'other_vote'
  ]
  
  meta_id = meta[meta.eeg_id == idx]

  labels = meta_id.loc[:, y_cols]
  if not all(labels.nunique() == 1):
    raise Exception(f'Not all votes are the same')

  raw = pd.read_parquet(op.join(eeg_path, str(idx) + '.parquet'))
  raw.insert(0, 'id', idx)
  data = process_raw(raw)

  # repeat y_labs for n_rows in raw
  labels = np.array([[str(y)] * data.shape[0] for y in labels.iloc[0,:]]).T
  
  return data, labels

# process function
def process_raw(raw):
  raw = raw[::2]
  raw = np.array(raw)
  if np.any(np.isnan(raw)):
    raise Exception('Some values are missing')
  return raw

load_data(eeg_path, ids[0], meta)

# data generator for batch processing
def data_generator(ids, batch_size):
  while True:
    for i in range(0, len(ids), batch_size):
      batch_ids = ids[i:i+batch_size]
      list_X = []
      list_labels = []
      for idx in batch_ids:
        try:
          X, labels = load_data(eeg_path, idx, meta)
          list_X.append(X)
          list_labels.append(labels)
        except Exception as e:
          print(f'Could not load {idx}: {e}, skipping...')
          continue
        X = np.vstack(list_X)
        labels = np.ravel(np.vstack(list_labels))
      yield X, labels

scaler = StandardScaler()
svm_model = SVC(kernel='linear', C=1.0)

# Training loop
for epoch in range(num_epochs):
  train_generator = data_generator(ids, batch_size)
  for X_batch, y_batch in train_generator:
    X_batch_scaled = scaler.fit_transform(X_batch)
    svm_model.fit(X_batch_scaled, y_batch) 
