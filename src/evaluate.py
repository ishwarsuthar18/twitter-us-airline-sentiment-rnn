import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
# Configuration

MODEL_PATH = "saved_models/rnn_model.keras"

X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"

OUTPUT_DIR = "outputs"

# Create Output Folder

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load Model

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)


# Load Test Data

print("Loading test data...")
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
print(f"X_test Shape : {X_test.shape}")
print(f"y_test Shape : {y_test.shape}")


# Predict

print("\nMaking predictions...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)


# Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)
print("\n========== Evaluation ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# Classification Report

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Negative",
        "Neutral",
        "Positive"
    ]
)
print("\nClassification Report\n")
print(report)
with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:
    file.write(report)


# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Neutral",
        "Positive"
    ]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )
)

plt.show()
print("\nConfusion matrix saved.")
print("\nEvaluation completed successfully.")