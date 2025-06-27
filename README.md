# RF Transmitter Location Classification using ML

This project explores the feasibility of determining a radio transmitter's physical location based on the subtle characteristics of its signal transients. By capturing a repeating signal from different known locations, we extract features from the signal's "turn-on" events and use them to train machine learning and deep learning models to classify the transmitter's position.

The core hypothesis is that multipath fading, reflections, and other environmental factors will imprint a unique, location-specific fingerprint onto the signal's transient ramp-up.

* **Receiver:** BladeRF Micro 2.0 xa4
* **Transmitter:** HackRF One
* **Signal:** A repeating on-off keying (OOK) cosine wave.
* **Environment:** A 3'x5' desk, with the transmitter placed at three distinct locations:
    1.  Adjacent to the receiver.
    2.  On the desk corner nearest to the receiver.
    3.  On the desk corner furthest from the receiver.

## Project Structure

The repository is organized into directories for data, processing scripts, and models. The main scripts often have a corresponding `_func.py` file which contains the core logic, separating configuration from implementation.

```
.
├── cnn
│   ├── cnn.py
│   └── cnn_func.py
├── data
│   ├── training
│   │   ├── cnn
│   │   └── random_forest
│   └── transient
├── docs
│   └── Ramp Detection Algorithm.pdf
├── plots
│   ├── amplitude
│   └── validation
├── process_for_cnn
│   ├── process_for_cnn.py
│   └── process_for_cnn_func.py
├── process_for_ml
│   ├── process_for_ml.py
│   └── process_for_ml_func.py
├── process_raw
│   ├── filter_transients.py
│   ├── filter_transients_func.py
│   ├── plot_amplitude.py
│   └── truncate_cf32.py
└── random_forest
    ├── random_forest.py
    └── random_forest_func.py
```

## The Data Processing Pipeline

The process is broken down into several key stages, moving from a raw signal capture to a location prediction.

### 1. Raw Data Processing (`/process_raw`)

This stage involves cleaning and preparing the initial, large `.cf32` signal captures.

* **`plot_amplitude.py`**: Before processing, it's crucial to visualize the raw data. This script reads a large `.cf32` file, splits it into four quarters, and plots the amplitude of each. This helps identify the approximate sample number where the first signal transmission begins, which is needed for the next step.
* **`truncate_cf32.py`**: Raw captures often start with millions of samples of noise. To speed up processing, this script removes a specified number of samples from the beginning of the file. **Warning:** This script permanently modifies the file.
* **`filter_transients.py` / `filter_transients_func.py`**: This is the most critical step. The script scans the truncated file to automatically detect and isolate every "turn-on" event (transient).
    * **Methodology**: It uses a 3-state machine (`searching_for_start`, `searching_for_plateau`, `waiting_for_signal_off`) and calculates the slope of the signal's amplitude within a sliding window.
    * **Tuning**: This script requires significant manual tuning of its parameters (e.g., `START_SLOPE_THRESHOLD`, `PLATEAU_SLOPE_THRESHOLD`) to accurately capture transients without including noise.
    * **Output**: The script saves all detected transients (as raw IQ samples) into a single, compressed `detected_transients_[id].npz` file for each location. It also generates verification plots for the first few detections.

### 2. Feature Engineering (`/process_for_ml` & `/process_for_cnn`)

Once transients are isolated, we extract meaningful features from them to create datasets for our models.

* **`process_for_ml.py`**: Prepares data for the Random Forest model. It loads the `.npz` files of transients and for each one, it calculates a feature vector containing:
    * **Signal Energy**: The sum of the squared magnitude.
    * **Central Spectral Bins**: The magnitudes of the 7 central frequency bins from an FFT, which captures the signal's shape in the frequency domain.
    * The final data is scaled using `StandardScaler` and saved as `ml_ready_data.npz`, with the scaler saved to `scaler.joblib`.
* **`process_for_cnn.py`**: Prepares data for the Convolutional Neural Network. It performs the same feature extraction as the ML script but also includes:
    * **Envelope Shape Features**: Standard deviation, skewness, and kurtosis of the magnitude envelope.
    * The final feature vector (11 features) is reshaped to be suitable for a 1D CNN `(num_samples, 11, 1)` and split into training, validation, and test sets. The final datasets are saved to `cnn_ready_data.npz`.

### 3. Model Training & Evaluation (`/random_forest` & `/cnn`)

With the prepared datasets, we can now train and evaluate the classifiers.

* **`random_forest.py`**: Loads the data prepared by `process_for_ml.py`, trains a `RandomForestClassifier`, and evaluates its performance. It outputs:
    * A classification report with precision, recall, and F1-score.
    * A confusion matrix plot.
    * A feature importance plot, showing which features were most influential.
* **`cnn.py`**: Loads the pre-split data from `process_for_cnn.py` and builds a 1D Convolutional Neural Network using TensorFlow/Keras. It trains the model using an early stopping callback to prevent overfitting and evaluates its performance on the test set. It outputs:
    * Test accuracy and loss.
    * A confusion matrix plot.
    * A plot of training/validation accuracy and loss over epochs.

## How to Run This Project

### Prerequisites

You will need Python 3 and the following libraries. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow pandas seaborn
```

### Step-by-Step Guide

1.  **Place Raw Data**: Place your raw `.cf32` files (one for each transmitter location) in a working directory.

2.  **Initial Analysis**: Configure `plot_amplitude.py` to inspect your file and note the sample number where the signal begins.

    ```python
    # In plot_amplitude.py
    file_to_analyze = '/path/to/your/capture.cf32'
    ```bash
    python process_raw/plot_amplitude.py
    ```

3.  **Truncate File**: Configure `truncate_cf32.py` with the path and the number of samples to remove.

    ```python
    # In truncate_cf32.py
    file_to_modify = '/path/to/your/capture.cf32'
    SAMPLES_TO_REMOVE = 13000000 # Example value
    ```bash
    python process_raw/truncate_cf32.py
    ```

4.  **Extract Transients (Iterative Step)**: This is the most important manual step.
    * Configure `filter_transients.py` with the path to your truncated file.
    * **Tune the `MANUAL Detector Parameters`** (`CHUNK_SIZE`, `WINDOW_SIZE`, `START_SLOPE_THRESHOLD`, etc.)
    * Run the script and inspect the `transient_verification_plots.png` it produces.
    * Repeat tuning until you are capturing a high number of clean transients.
    * Rename the output file (e.g., `detected_transients_sdr_pos1.npz`) and repeat for each location's data file.
    ```bash
    python process_raw/filter_transients.py
    ```

5.  **Prepare Datasets**:
    * Move all your `detected_transients_*.npz` files into the `data/transient/` directory.
    * Update the directory paths in `process_for_ml.py` and `process_for_cnn.py`.
    * Run both scripts to generate the final datasets.
    ```bash
    python process_for_ml/process_for_ml.py
    python process_for_cnn/process_for_cnn.py
    ```

6.  **Train and Evaluate Models**:
    * Update the data file paths in `random_forest.py` and `cnn.py`.
    * Run the scripts to train the models and see the results.
    ```bash
    python random_forest/random_forest.py
    python cnn/cnn.py
    ```
