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