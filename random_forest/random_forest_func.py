import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(file_path):
    """Loads the processed feature matrix (X) and labels (y) from a .npz file."""
    try:
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        print(f"Data loaded successfully from '{file_path}'.")
        print(f"Feature matrix shape: {X.shape}")
        return X, y
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'.")
        return None, None


def train_model(X_train, y_train):
    """Initializes and trains a RandomForestClassifier."""
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_and_visualize(model, X_test, y_test, feature_names):
    """Evaluates the model and visualizes the results."""
    class_names = ['SDR 0', 'SDR 1', 'SDR 2']

    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Lightweight Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # --- Feature Importance Plot ---
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\n--- Feature Importances (Lightweight Model) ---")
    print(feature_importance_df)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importances (Lightweight Model)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
