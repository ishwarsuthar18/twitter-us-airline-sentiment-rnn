import numpy as np 
#import pandas as pd 
from requests import post
from tensorflow.keras.preprocessing.sequence import pad_sequences

path="data/processed/X_sequences.npy"
save_path="data/processed/X_padded.npy"

max_length=40
padding="post"
d_type="post"

print("loading sequences...")
X = np.load(path,
            allow_pickle=True)
print(f"total sequences:{len(X)}")
X_padded = pad_sequences(X,
                          maxlen=max_length,
                          padding=padding, 
                          truncating=d_type)
np.save(save_path, X_padded)

print(f"padded sequences saved to {save_path}")

print("padding completed successfully!")
print("before padding:")
print(f"sequences shape: {X.shape}")
print("after padding:")
print(f"padded sequences shape: {X_padded.shape}")