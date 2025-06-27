# Import only the necessary functions and libraries
from process_for_ml_func import load_and_prepare_data_lightweight
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

if __name__ == '__main__':
    # Define your data source and output paths
    DATA_SOURCE_DIR = '/Users/lukebray/School/Research/SDR/RadioClassification/data/transient'
    OUTPUT_DATA_FILE = '/Users/lukebray/School/Research/SDR/RadioClassification/data/training/random_forest/ml_ready_data.npz'
    OUTPUT_SCALER_FILE = '/Users/lukebray/School/Research/SDR/RadioClassification/data/training/random_forest/scaler.joblib'

    print("--- Step 1: Loading and preprocessing data (Lightweight) ---")
    # We no longer need to calculate FIXED_LENGTH
    X, y = load_and_prepare_data_lightweight(DATA_SOURCE_DIR)

    if X is not None:
        # The new shape will be much smaller, around (725, 11)
        print(f"\nOriginal lightweight feature matrix shape: {X.shape}")

        print("\n--- Step 2: Scaling features ---")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Features have been scaled.")

        print(f"\n--- Step 3: Saving processed lightweight data and scaler ---")
        np.savez_compressed(OUTPUT_DATA_FILE, X=X_scaled, y=y)
        print(f"Saved scaled lightweight data to '{OUTPUT_DATA_FILE}'")

        joblib.dump(scaler, OUTPUT_SCALER_FILE)
        print(f"Saved lightweight scaler object to '{OUTPUT_SCALER_FILE}'")

        print("\nDone. Lightweight data is ready for training.")

