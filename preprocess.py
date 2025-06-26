import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_central_spectral_bins(iq_data, num_bins=7):
    """
    Performs an FFT on IQ data and returns the normalized central frequency bins.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        num_bins (int): The number of central bins to extract.

    Returns:
        np.ndarray: A 1D NumPy array of the normalized magnitudes of the central bins.
    """
    # Ensure there's enough data for an FFT and to extract the bins
    if iq_data.size < num_bins:
        return np.zeros(num_bins, dtype=np.float32)

    # Apply a Hanning window to reduce spectral leakage at the edges of the sample
    windowed_data = iq_data * np.hanning(len(iq_data))

    # Perform the Fast Fourier Transform
    fft_result = np.fft.fft(windowed_data)
    fft_shifted = np.fft.fftshift(fft_result) # Shift zero-frequency component to the center
    fft_magnitude = np.abs(fft_shifted)

    # Normalize the entire spectrum's magnitude to be between 0 and 1
    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude /= max_fft_mag

    # Calculate the indices for the central bins
    center_index = len(fft_magnitude) // 2
    start_index = center_index - (num_bins // 2)
    end_index = start_index + num_bins

    selected_magnitudes = fft_magnitude[start_index:end_index]

    return selected_magnitudes.astype(np.float32)


def load_and_prepare_data(data_directory, fixed_length, file_pattern='detected_transients_*.npz'):
    """
    Loads all .npz files, creates features from both magnitude and spectral
    data, and formats them for a CNN.

    Args:
        data_directory (str): The path to the folder containing the .npz files.
        fixed_length (int): The target length for the magnitude part of the sample.
        file_pattern (str): The glob pattern to find the data files.

    Returns:
        tuple: A tuple containing two NumPy arrays: (all_processed_samples, all_labels).
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching the pattern '{search_path}'")
        return None, None

    all_samples = []
    all_labels = []

    sdr_labels = {}
    current_label = 0

    print(f"Found {len(file_paths)} files to process...")

    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        try:
            sdr_id = [part for part in filename.split('_') if 'sdr' in part][0]
        except IndexError:
            print(f"Warning: Could not determine SDR ID from filename '{filename}'. Skipping.")
            continue

        if sdr_id not in sdr_labels:
            print(f"Found new source: '{sdr_id}'. Assigning label: {current_label}")
            sdr_labels[sdr_id] = current_label
            current_label += 1
        label = sdr_labels[sdr_id]

        with np.load(file_path) as data:
            for key in data.keys():
                iq_data = data[key]

                # --- Feature 1: Magnitude (Time Domain) ---
                magnitude_data = np.abs(iq_data)
                max_val = np.max(magnitude_data)
                if max_val > 0:
                    magnitude_data /= max_val

                current_length = len(magnitude_data)
                if current_length > fixed_length:
                    start = (current_length - fixed_length) // 2
                    processed_magnitude = magnitude_data[start: start + fixed_length]
                else:
                    pad_width = fixed_length - current_length
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    processed_magnitude = np.pad(magnitude_data, (pad_left, pad_right), 'constant')

                # --- Feature 2: Spectral Bins (Frequency Domain) ---
                spectral_bins = get_central_spectral_bins(iq_data, num_bins=7)

                # --- Combine Features ---
                combined_features = np.concatenate([processed_magnitude, spectral_bins])

                # --- Format for CNN ---
                cnn_formatted_sample = np.expand_dims(combined_features, axis=-1)
                all_samples.append(cnn_formatted_sample)
                all_labels.append(label)

    print(f"\nFinished processing. Found {len(all_samples)} total transients.")
    print("Label mapping:", sdr_labels)
    return np.array(all_samples, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def get_target_length(data_directory, percentile=50, file_pattern='detected_transients_*.npz'):
    """
    Calculates a target length for transients by analyzing the distribution of lengths.
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)
    if not file_paths:
        return None

    # --- FIXED: Reverted to a for-loop to correctly handle file loading ---
    lengths = []
    for file_path in file_paths:
        with np.load(file_path) as data:
            for key in data.keys():
                lengths.append(len(data[key]))

    if not lengths:
        return None
    target_len = int(np.percentile(lengths, percentile))
    print(
        f"Analyzed {len(lengths)} transients. Recommended fixed length ({percentile}th percentile): {target_len} samples.")
    return target_len


def plot_spectral_bins_from_array(iq_data, transient_name="Transient"):
    """
    Performs an FFT on a given IQ data array and plots the 7 central
    frequency bins. Provided for visualization/utility.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return
    try:
        windowed_data = iq_data * np.hanning(len(iq_data))
        fft_result = np.fft.fft(windowed_data)
        fft_shifted = np.fft.fftshift(fft_result)
        fft_magnitude = np.abs(fft_shifted)
        center_index = len(fft_magnitude) // 2
        if len(fft_magnitude) < 7:
            print(f"Error: FFT size ({len(fft_magnitude)}) is too small to plot 7 bins.")
            return
        start_index = center_index - 3
        end_index = center_index + 4
        selected_magnitudes = fft_magnitude[start_index:end_index]
        bin_labels = ['-3', '-2', '-1', '0', '1', '2', '3']
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(bin_labels, selected_magnitudes)
        ax.set_title(f'Central Frequency Bins for {transient_name}', fontsize=16)
        ax.set_xlabel('Frequency Bin (Relative to Center)', fontsize=12)
        ax.set_ylabel('Spectral Magnitude', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        textstr = f'FFT Size: {len(iq_data)} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(f'{transient_name}_spectral_bins.png')
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == '__main__':
    DATA_SOURCE_DIR = '.'
    OUTPUT_FILE = 'cnn_ready_data.npz'

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
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_central_spectral_bins(iq_data, num_bins=7):
    """
    Performs an FFT on IQ data and returns the normalized central frequency bins.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        num_bins (int): The number of central bins to extract.

    Returns:
        np.ndarray: A 1D NumPy array of the normalized magnitudes of the central bins.
    """
    # Ensure there's enough data for an FFT and to extract the bins
    if iq_data.size < num_bins:
        return np.zeros(num_bins, dtype=np.float32)

    # Apply a Hanning window to reduce spectral leakage at the edges of the sample
    windowed_data = iq_data * np.hanning(len(iq_data))

    # Perform the Fast Fourier Transform
    fft_result = np.fft.fft(windowed_data)
    fft_shifted = np.fft.fftshift(fft_result) # Shift zero-frequency component to the center
    fft_magnitude = np.abs(fft_shifted)

    # Normalize the entire spectrum's magnitude to be between 0 and 1
    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude /= max_fft_mag

    # Calculate the indices for the central bins
    center_index = len(fft_magnitude) // 2
    start_index = center_index - (num_bins // 2)
    end_index = start_index + num_bins

    selected_magnitudes = fft_magnitude[start_index:end_index]

    return selected_magnitudes.astype(np.float32)


def load_and_prepare_data(data_directory, fixed_length, file_pattern='detected_transients_*.npz'):
    """
    Loads all .npz files, creates features from both magnitude and spectral
    data, and formats them for a CNN.

    Args:
        data_directory (str): The path to the folder containing the .npz files.
        fixed_length (int): The target length for the magnitude part of the sample.
        file_pattern (str): The glob pattern to find the data files.

    Returns:
        tuple: A tuple containing two NumPy arrays: (all_processed_samples, all_labels).
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching the pattern '{search_path}'")
        return None, None

    all_samples = []
    all_labels = []

    sdr_labels = {}
    current_label = 0

    print(f"Found {len(file_paths)} files to process...")

    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        try:
            sdr_id = [part for part in filename.split('_') if 'sdr' in part][0]
        except IndexError:
            print(f"Warning: Could not determine SDR ID from filename '{filename}'. Skipping.")
            continue

        if sdr_id not in sdr_labels:
            print(f"Found new source: '{sdr_id}'. Assigning label: {current_label}")
            sdr_labels[sdr_id] = current_label
            current_label += 1
        label = sdr_labels[sdr_id]

        with np.load(file_path) as data:
            for key in data.keys():
                iq_data = data[key]

                # --- Feature 1: Magnitude (Time Domain) ---
                magnitude_data = np.abs(iq_data)
                max_val = np.max(magnitude_data)
                if max_val > 0:
                    magnitude_data /= max_val

                current_length = len(magnitude_data)
                if current_length > fixed_length:
                    start = (current_length - fixed_length) // 2
                    processed_magnitude = magnitude_data[start: start + fixed_length]
                else:
                    pad_width = fixed_length - current_length
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    processed_magnitude = np.pad(magnitude_data, (pad_left, pad_right), 'constant')

                # --- Feature 2: Spectral Bins (Frequency Domain) ---
                spectral_bins = get_central_spectral_bins(iq_data, num_bins=7)

                # --- Combine Features ---
                combined_features = np.concatenate([processed_magnitude, spectral_bins])

                # --- Format for CNN ---
                cnn_formatted_sample = np.expand_dims(combined_features, axis=-1)
                all_samples.append(cnn_formatted_sample)
                all_labels.append(label)

    print(f"\nFinished processing. Found {len(all_samples)} total transients.")
    print("Label mapping:", sdr_labels)
    return np.array(all_samples, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def get_target_length(data_directory, percentile=50, file_pattern='detected_transients_*.npz'):
    """
    Calculates a target length for transients by analyzing the distribution of lengths.
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)
    if not file_paths:
        return None

    # --- FIXED: Reverted to a for-loop to correctly handle file loading ---
    lengths = []
    for file_path in file_paths:
        with np.load(file_path) as data:
            for key in data.keys():
                lengths.append(len(data[key]))

    if not lengths:
        return None
    target_len = int(np.percentile(lengths, percentile))
    print(
        f"Analyzed {len(lengths)} transients. Recommended fixed length ({percentile}th percentile): {target_len} samples.")
    return target_len


def plot_spectral_bins_from_array(iq_data, transient_name="Transient"):
    """
    Performs an FFT on a given IQ data array and plots the 7 central
    frequency bins. Provided for visualization/utility.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return
    try:
        windowed_data = iq_data * np.hanning(len(iq_data))
        fft_result = np.fft.fft(windowed_data)
        fft_shifted = np.fft.fftshift(fft_result)
        fft_magnitude = np.abs(fft_shifted)
        center_index = len(fft_magnitude) // 2
        if len(fft_magnitude) < 7:
            print(f"Error: FFT size ({len(fft_magnitude)}) is too small to plot 7 bins.")
            return
        start_index = center_index - 3
        end_index = center_index + 4
        selected_magnitudes = fft_magnitude[start_index:end_index]
        bin_labels = ['-3', '-2', '-1', '0', '1', '2', '3']
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(bin_labels, selected_magnitudes)
        ax.set_title(f'Central Frequency Bins for {transient_name}', fontsize=16)
        ax.set_xlabel('Frequency Bin (Relative to Center)', fontsize=12)
        ax.set_ylabel('Spectral Magnitude', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        textstr = f'FFT Size: {len(iq_data)} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(f'{transient_name}_spectral_bins.png')
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == '__main__':
    DATA_SOURCE_DIR = '.'
    OUTPUT_FILE = 'cnn_ready_data.npz'

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
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_central_spectral_bins(iq_data, num_bins=7):
    """
    Performs an FFT on IQ data and returns the normalized central frequency bins.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        num_bins (int): The number of central bins to extract.

    Returns:
        np.ndarray: A 1D NumPy array of the normalized magnitudes of the central bins.
    """
    # Ensure there's enough data for an FFT and to extract the bins
    if iq_data.size < num_bins:
        return np.zeros(num_bins, dtype=np.float32)

    # Apply a Hanning window to reduce spectral leakage at the edges of the sample
    windowed_data = iq_data * np.hanning(len(iq_data))

    # Perform the Fast Fourier Transform
    fft_result = np.fft.fft(windowed_data)
    fft_shifted = np.fft.fftshift(fft_result) # Shift zero-frequency component to the center
    fft_magnitude = np.abs(fft_shifted)

    # Normalize the entire spectrum's magnitude to be between 0 and 1
    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude /= max_fft_mag

    # Calculate the indices for the central bins
    center_index = len(fft_magnitude) // 2
    start_index = center_index - (num_bins // 2)
    end_index = start_index + num_bins

    selected_magnitudes = fft_magnitude[start_index:end_index]

    return selected_magnitudes.astype(np.float32)


def load_and_prepare_data(data_directory, fixed_length, file_pattern='detected_transients_*.npz'):
    """
    Loads all .npz files, creates features from both magnitude and spectral
    data, and formats them for a CNN.

    Args:
        data_directory (str): The path to the folder containing the .npz files.
        fixed_length (int): The target length for the magnitude part of the sample.
        file_pattern (str): The glob pattern to find the data files.

    Returns:
        tuple: A tuple containing two NumPy arrays: (all_processed_samples, all_labels).
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching the pattern '{search_path}'")
        return None, None

    all_samples = []
    all_labels = []

    sdr_labels = {}
    current_label = 0

    print(f"Found {len(file_paths)} files to process...")

    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        try:
            sdr_id = [part for part in filename.split('_') if 'sdr' in part][0]
        except IndexError:
            print(f"Warning: Could not determine SDR ID from filename '{filename}'. Skipping.")
            continue

        if sdr_id not in sdr_labels:
            print(f"Found new source: '{sdr_id}'. Assigning label: {current_label}")
            sdr_labels[sdr_id] = current_label
            current_label += 1
        label = sdr_labels[sdr_id]

        with np.load(file_path) as data:
            for key in data.keys():
                iq_data = data[key]

                # --- Feature 1: Magnitude (Time Domain) ---
                magnitude_data = np.abs(iq_data)
                max_val = np.max(magnitude_data)
                if max_val > 0:
                    magnitude_data /= max_val

                current_length = len(magnitude_data)
                if current_length > fixed_length:
                    start = (current_length - fixed_length) // 2
                    processed_magnitude = magnitude_data[start: start + fixed_length]
                else:
                    pad_width = fixed_length - current_length
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    processed_magnitude = np.pad(magnitude_data, (pad_left, pad_right), 'constant')

                # --- Feature 2: Spectral Bins (Frequency Domain) ---
                spectral_bins = get_central_spectral_bins(iq_data, num_bins=7)

                # --- Combine Features ---
                combined_features = np.concatenate([processed_magnitude, spectral_bins])

                # --- Format for CNN ---
                cnn_formatted_sample = np.expand_dims(combined_features, axis=-1)
                all_samples.append(cnn_formatted_sample)
                all_labels.append(label)

    print(f"\nFinished processing. Found {len(all_samples)} total transients.")
    print("Label mapping:", sdr_labels)
    return np.array(all_samples, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def get_target_length(data_directory, percentile=50, file_pattern='detected_transients_*.npz'):
    """
    Calculates a target length for transients by analyzing the distribution of lengths.
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)
    if not file_paths:
        return None

    # --- FIXED: Reverted to a for-loop to correctly handle file loading ---
    lengths = []
    for file_path in file_paths:
        with np.load(file_path) as data:
            for key in data.keys():
                lengths.append(len(data[key]))

    if not lengths:
        return None
    target_len = int(np.percentile(lengths, percentile))
    print(
        f"Analyzed {len(lengths)} transients. Recommended fixed length ({percentile}th percentile): {target_len} samples.")
    return target_len


def plot_spectral_bins_from_array(iq_data, transient_name="Transient"):
    """
    Performs an FFT on a given IQ data array and plots the 7 central
    frequency bins. Provided for visualization/utility.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return
    try:
        windowed_data = iq_data * np.hanning(len(iq_data))
        fft_result = np.fft.fft(windowed_data)
        fft_shifted = np.fft.fftshift(fft_result)
        fft_magnitude = np.abs(fft_shifted)
        center_index = len(fft_magnitude) // 2
        if len(fft_magnitude) < 7:
            print(f"Error: FFT size ({len(fft_magnitude)}) is too small to plot 7 bins.")
            return
        start_index = center_index - 3
        end_index = center_index + 4
        selected_magnitudes = fft_magnitude[start_index:end_index]
        bin_labels = ['-3', '-2', '-1', '0', '1', '2', '3']
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(bin_labels, selected_magnitudes)
        ax.set_title(f'Central Frequency Bins for {transient_name}', fontsize=16)
        ax.set_xlabel('Frequency Bin (Relative to Center)', fontsize=12)
        ax.set_ylabel('Spectral Magnitude', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        textstr = f'FFT Size: {len(iq_data)} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(f'{transient_name}_spectral_bins.png')
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == '__main__':
    DATA_SOURCE_DIR = '.'
    OUTPUT_FILE = 'cnn_ready_data.npz'

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