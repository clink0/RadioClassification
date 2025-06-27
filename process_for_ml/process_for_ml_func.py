import numpy as np
import os
import glob
from scipy.stats import skew, kurtosis


def get_envelope_shape_features(magnitude_data):
    """
    Calculates statistical moments of the signal's magnitude envelope.
    """
    if magnitude_data.size < 4 or np.all(magnitude_data == magnitude_data[0]):
        return np.zeros(3, dtype=np.float32)

    std_dev = np.std(magnitude_data)
    skewness = skew(magnitude_data)
    kurt = kurtosis(magnitude_data)

    features = np.array([std_dev, skewness, kurt], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def get_central_spectral_bins(iq_data, num_bins=7):
    """
    Performs an FFT on IQ data and returns the normalized central frequency bins.
    """
    if iq_data.size < num_bins:
        return np.zeros(num_bins, dtype=np.float32)

    windowed_data = iq_data * np.hanning(len(iq_data))
    fft_result = np.fft.fft(windowed_data)
    fft_shifted = np.fft.fftshift(fft_result)
    fft_magnitude = np.abs(fft_shifted)

    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude /= max_fft_mag

    center_index = len(fft_magnitude) // 2
    start_index = center_index - (num_bins // 2)
    end_index = start_index + num_bins

    return fft_magnitude[start_index:end_index].astype(np.float32)


def load_and_prepare_data_lightweight(data_directory, file_pattern='detected_transients_*.npz'):
    """
    Loads all .npz files and creates a lightweight 2D feature matrix containing
    only the most important scalar features.

    Features created (11 total):
    1. Signal Energy (1 feature)
    2. Envelope Shape (std, skew, kurtosis) (3 features)
    3. Central Spectral Bins (7 features)
    """
    search_path = os.path.join(data_directory, file_pattern)
    file_paths = glob.glob(search_path)

    if not file_paths:
        print(f"Error: No files found matching the pattern '{search_path}'")
        return None, None

    all_features = []
    all_labels = []
    sdr_labels = {}
    current_label = 0

    print(f"Found {len(file_paths)} files to process...")

    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        try:
            sdr_id = [part for part in filename.split('_') if 'sdr' in part][0]
        except IndexError:
            continue

        if sdr_id not in sdr_labels:
            sdr_labels[sdr_id] = current_label
            current_label += 1
        label = sdr_labels[sdr_id]

        with np.load(file_path) as data:
            for key in data.keys():
                iq_data = data[key]
                if len(iq_data) < 4: continue

                magnitude_data = np.abs(iq_data)

                # --- Extract only the most important features ---
                signal_energy = np.array([np.sum(magnitude_data ** 2)])
                envelope_features = get_envelope_shape_features(magnitude_data)
                spectral_bins = get_central_spectral_bins(iq_data, num_bins=7)

                # --- Combine into a single, small feature vector ---
                combined_features = np.concatenate([
                    signal_energy,
                    #envelope_features,
                    spectral_bins
                ]).astype(np.float32)

                all_features.append(combined_features)
                all_labels.append(label)

    print(f"\nFinished processing. Found {len(all_features)} total transients.")
    print("Label mapping:", sdr_labels)
    # The final feature matrix will have shape (n_samples, 11)
    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)
