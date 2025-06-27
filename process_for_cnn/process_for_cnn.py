from process_for_cnn_func import *

if __name__ == '__main__':
    DATA_SOURCE_DIR = '/Users/lukebray/School/Research/SDR/RadioClassification/data/transient'
    OUTPUT_FILE = '/Users/lukebray/School/Research/SDR/RadioClassification/data/training/cnn/cnn_ready_data.npz'

    print("--- Step 1: Determining optimal fixed length for samples ---")
    FIXED_LENGTH = get_target_length(DATA_SOURCE_DIR, percentile=50)

    if FIXED_LENGTH is None:
        print("Could not determine target length. Exiting.")
    else:
        print("\n--- Step 2: Loading and preprocessing all data (Magnitude + Spectral) ---")
        X, y = load_and_prepare_data(DATA_SOURCE_DIR, FIXED_LENGTH)

        if X is not None:
            print("\n--- Step 3: Splitting data into training, validation, and test sets ---")
            x_train, x_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            print(f"Training set shape:   {x_train.shape}, Labels shape: {y_train.shape}")
            print(f"Validation set shape: {x_val.shape}, Labels shape: {y_val.shape}")
            print(f"Test set shape:       {x_test.shape}, Labels shape: {y_test.shape}")
            print("\n--- Step 4: Saving processed data to '{OUTPUT_FILE}' ---")
            np.savez_compressed(
                OUTPUT_FILE,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test
            )
            print("Done. You are now ready to train your CNN!")