import numpy as np
from sklearn.model_selection import train_test_split
# Configuration
X_PATH = "data/processed/X_padded.npy"
Y_PATH = "data/processed/y.npy"
TEST_SIZE = 0.2
RANDOM_STATE = 42#jaisa ki hmesha rakhte hai 
# Load Data
print("Loading padded data...")
X = np.load(X_PATH)
y = np.load(Y_PATH)
print(f"Total Samples : {len(X)}")
# Train-Test Split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
# Save Data
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
# Information
print("\nDataset Split Completed Successfully!")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"\nX_train Shape : {X_train.shape}")
print(f"X_test Shape  : {X_test.shape}")
print(f"\ny_train Shape : {y_train.shape}")
print(f"y_test Shape  : {y_test.shape}")