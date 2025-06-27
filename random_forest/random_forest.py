from random_forest_func import *

if __name__ == '__main__':
    # Define the path to the lightweight dataset you created
    DATA_FILE = '/Users/lukebray/School/Research/SDR/RadioClassification/data/training/random_forest/ml_ready_data.npz'

    # Load the data
    X, y = load_data(DATA_FILE)

    if X is not None:
        # --- DEFINE THE CORRECT FEATURE NAMES ---
        # This list MUST match the features you decided to keep.
        # This assumes you kept energy, envelope shape, and spectral bins (1+3+7=11 features)
        if X.shape[1] == 11:
            feature_names = (
                    ['energy'] +
                    ['env_std', 'env_skew', 'env_kurt'] +
                    [f'spec_{i}' for i in range(7)]
            )
        # This assumes you kept only energy and spectral bins (1+7=8 features)
        elif X.shape[1] == 8:
            feature_names = ['energy'] + [f'spec_{i}' for i in range(7)]
        else:
            raise ValueError(f"Unrecognized feature set size: {X.shape[1]}. Please define feature_names manually.")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate and visualize results
        evaluate_and_visualize(model, X_test, y_test, feature_names)

