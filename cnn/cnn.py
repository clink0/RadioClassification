import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_prepared_data(file_path='cnn_ready_data.npz'):
    """
    Loads the pre-split and preprocessed data from the .npz file.
    """
    try:
        with np.load(file_path) as data:
            return (data['x_train'], data['y_train'],
                    data['x_val'], data['y_val'],
                    data['x_test'], data['y_test'])
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please run the 'prepare_cnn_data.py' script first.")
        return None


def build_cnn_model(input_shape, num_classes):
    """
    Builds the Keras CNN model architecture. (REWORKED)
    This version uses a wider, but shallower architecture.
    """
    model = Sequential([
        # --- REWORKED BLOCK 1 ---
        # Using more filters (128 vs 64) and a smaller kernel (5 vs 7)
        Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),  # Added dropout for regularization within the conv block

        # --- REWORKED BLOCK 2 ---
        # Increased filters to 256, kept kernel size at 3
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4), # Increased dropout as we go deeper

        # --- REWORKED DENSE HEAD ---
        # The third convolutional block was removed for a shallower design.
        # The dense layers were simplified to a single Dense(128) layer.
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # A 50% dropout is still a good practice before the output layer

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    # --- CHANGED OPTIMIZER ---
    # Switched from 'adam' to 'rmsprop' to explore different optimization dynamics.
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates and plots a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', # Changed color map for variety
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_reworked.png')
    plt.show()


def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig('training_history_reworked.png')
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    DATA_FILE = '/Users/lukebray/School/Research/SDR/RadioClassification/data/training/cnn_ready_data.npz'

    # --- Main Workflow ---

    # 1. Load the preprocessed data
    print("--- Step 1: Loading preprocessed data ---")
    prepared_data = load_prepared_data(DATA_FILE)

    if prepared_data:
        x_train, y_train, x_val, y_val, x_test, y_test = prepared_data

        input_shape = x_train.shape[1:]
        num_classes = len(np.unique(y_train))
        class_names = [f'SDR {i}' for i in range(num_classes)]

        print(f"Input shape for model: {input_shape}")
        print(f"Number of classes: {num_classes}")

        # 2. Build the CNN model
        print("\n--- Step 2: Building the REWORKED CNN model ---")
        model = build_cnn_model(input_shape, num_classes)
        model.summary()

        # 3. Train the model
        print("\n--- Step 3: Training the model with reworked parameters ---")
        # --- REWORKED: Adjusted training parameters ---
        EPOCHS = 50          # Increased max epochs; EarlyStopping will find the best one
        BATCH_SIZE = 32

        # Define the EarlyStopping callback
        # Increased patience to allow the model more epochs to improve before stopping.
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,             # <-- CHANGED: Increased patience from 3 to 5
            restore_best_weights=True,
            verbose=1               # <-- ADDED: Will print a message when stopping occurs
        )

        # Pass the callback to the model's fit method
        history = model.fit(x_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping])

        # 4. Evaluate the model on the test set
        print("\n--- Step 4: Evaluating the model on test data ---")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")

        # 5. Generate predictions and plot the confusion matrix
        print("\n--- Step 5: Generating confusion matrix ---")
        y_pred_probs = model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        plot_confusion_matrix(y_test, y_pred, class_names)
        plot_training_history(history)