import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

from model.rnn_model import build_model

#configuration
X_train_path="data/preprocessed/X_train.npy"
y_train_path="data/preprocessed/y_train.npy"

x_test_path="data/preprocessed/X_test.npy"
y_test_path="data/preprocessed/y_test.npy"


model_save_path="data/model/rnn_model.keras"

batch_size=32
epochs=10
learning_rate=0.001


#save directory
os.makedirs("saved_models", exist_ok=True)

#load data
print("loading dataset.....")
X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_test = np.load(x_test_path)
y_test = np.load(y_test_path)
#shape
print(f"X_train shape: {X_train.shape}")
print(f"X_tesst shape:{X_test.shape}")


#build and compile model

model=build_model()

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='')