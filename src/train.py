import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from model.rnn_model import build_model

#configuration
X_train_path="data/processed/X_train.npy"
y_train_path="data/processed/y_train.npy"

x_test_path="data/processed/X_test.npy"
y_test_path="data/processed/y_test.npy"


model_save_path="saved_models/rnn_model.keras"

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
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

#callbacks
early_stopping=EarlyStopping(monitor='val_loss',
                             patience=3,
                             restore_best_weights=True)

checkpoint=ModelCheckpoint(filepath=model_save_path,
                             monitor='val_accuracy',
                             save_best_only=True)



#train model
print("\ntraining model......")
history=model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping,checkpoint],
    verbose=1
)
#save model
model.save(model_save_path)
print(f"\n model saved succesfully to {model_save_path}")

#evaluate model
loss,accuracy=model.evaluate(X_test,y_test,verbose=0)
print("\n final test results ")
print(f"loss:{loss:.4f}")
print(f"accuracy:{accuracy:.4f}")

#save training history 
np.save("outputs/train_accuracy.npy",history.history['accuracy'])
np.save("outputs/val_accuracy.npy",history.history['val_accuracy'])
np.save("outputs/train_loss.npy",history.history['loss']) 
#np.save()  
np.save(
    "outputs/val_loss.npy",
    history.history["val_loss"]
)

print("\nTraining history saved successfully.")      