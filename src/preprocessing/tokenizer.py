import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. config
DATA_PATH = "data/processed/cleaned_tweets.csv"
SAVE_DIR = "saved_models"

VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"

# 2. load data
print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape : {df.shape}")

# 3. remove missing values
df = df.dropna(subset=["text", "airline_sentiment"])
df["text"] = df["text"].astype(str)

print(f"Dataset Shape After Cleaning : {df.shape}")

# 4. split features & labels
X = df["text"]
y = df["airline_sentiment"]

# 5. encode labels
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

y = y.map(label_map)

# 6. remove invalid labels
valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows].astype(int)

# 7. create tokenizer
print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token=OOV_TOKEN
)

tokenizer.fit_on_texts(X)

# 8. convert text to sequences
print("Converting text into sequences...")

X_sequences = tokenizer.texts_to_sequences(X)

# 9. create save folder
os.makedirs(SAVE_DIR, exist_ok=True)

# 10. save tokenizer
tokenizer_path = os.path.join(
    SAVE_DIR,
    "tokenizer.pkl"
)

with open(tokenizer_path, "wb") as file:
    pickle.dump(tokenizer, file)

print("Tokenizer saved successfully.")

# 11. save sequences
np.save(
    "data/processed/X_sequences.npy",
    np.array(X_sequences, dtype=object)
)

np.save(
    "data/processed/y.npy",
    y.to_numpy()
)

# 12. show information
print("\nVocabulary Size :", len(tokenizer.word_index))

print("\nSample Vocabulary:")

for word, index in list(tokenizer.word_index.items())[:15]:
    print(f"{word:<15} -> {index}")

print("\nFirst Tweet")
print(X.iloc[0])

print("\nSequence")
print(X_sequences[0])

print("\nTokenizer preprocessing completed successfully.")