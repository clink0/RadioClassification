# new_analysis_script.py

import numpy as np
import matplotlib.pyplot as plt


# --- Paste the refactored functions above here ---
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_spectral_bins_from_array(iq_data, transient_name="Transient"):
    """
    Performs an FFT on a given IQ data array and plots the 7 central
    frequency bins.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return

    try:
        # The data is already complex, so we can proceed directly
        windowed_data = iq_data * np.hanning(len(iq_data))
        fft_result = np.fft.fft(windowed_data)
        fft_shifted = np.fft.fftshift(fft_result)
        fft_magnitude = np.abs(fft_shifted)

        center_index = len(fft_magnitude) // 2
        # Ensure there are enough bins for the plot
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


def plot_amplitude_vs_time_from_array(iq_data, sample_rate, transient_name="Transient"):
    """
    Plots the signal amplitude vs. time for a given IQ data array.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        sample_rate (int or float): The sample rate of the data.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return

    try:
        amplitude = np.abs(iq_data)
        # The time vector is now relative to the start of the transient (starts at 0)
        time_vector = np.arange(len(amplitude)) / sample_rate

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(time_vector, amplitude, color='mediumvioletred', linewidth=1)
        ax.set_title(f'Amplitude vs. Time for {transient_name}', fontsize=16)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        duration_ms = (len(iq_data) / sample_rate) * 1000
        textstr = (f'Sample Rate: {sample_rate / 1e6:.2f} MS/s\n'
                   f'Duration: {duration_ms:.4f} ms')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(f'{transient_name}_amplitude_vs_time.png')
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


def plot_iq_components_from_array(iq_data, transient_name="Transient"):
    """
    Plots the I and Q components from a given IQ data array.

    Args:
        iq_data (np.ndarray): A NumPy array of complex IQ samples.
        transient_name (str): An identifier for the plot title.
    """
    if iq_data.size == 0:
        print("Error: Input IQ data is empty.")
        return

    try:
        i_samples = iq_data.real
        q_samples = iq_data.imag
        # The x-axis is now just the sample index within the transient
        sample_indices = np.arange(len(iq_data))

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(sample_indices, i_samples, label='I Component', color='dodgerblue', linewidth=1)
        ax.plot(sample_indices, q_samples, label='Q Component', color='orangered', linewidth=1, alpha=0.9)
        ax.set_title(f'I/Q Data for {transient_name}', fontsize=16)
        ax.set_xlabel('Sample Number (within transient)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True)
        textstr = f'Displaying {len(iq_data):,} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(f'{transient_name}_iq_data.png')
        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")

# --- Also include the loader function from before ---
def load_transients_from_npz(file_path):
    """
    Loads transient data from a .npz file into a list of NumPy arrays.
    """
    try:
        with np.load(file_path) as loader:
            keys = sorted(loader.files, key=lambda k: int(k.split('_')[1]))
            transients_list = [loader[key] for key in keys]
        print(f"Successfully loaded {len(transients_list)} transients from '{file_path}'")
        return transients_list
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return []


# --- Main Analysis Workflow ---

# 1. Configuration
data_file = 'detected_transients_6feet.npz'
SAMPLE_RATE = 2e6  # The sample rate used when capturing the original data

# 2. Load the pre-processed data
my_transients = load_transients_from_npz(data_file)

if my_transients:
    # 3. Choose which transient to analyze
    # Let's pick the 5th transient (index 4), for example.
    transient_index = 4

    if transient_index < len(my_transients):
        transient_to_analyze = my_transients[transient_index]
        plot_name = f"Transient_{transient_index}"

        print(f"\n--- Analyzing {plot_name} ---")

        # 4. Use the new functions to plot the data from the selected array

        # Plot I/Q components
        plot_iq_components_from_array(transient_to_analyze, transient_name=plot_name)

        # Plot Amplitude vs. Time
        plot_amplitude_vs_time_from_array(transient_to_analyze, SAMPLE_RATE, transient_name=plot_name)

        # Plot the central FFT bins
        plot_spectral_bins_from_array(transient_to_analyze, transient_name=plot_name)

    else:
        print(f"Error: Transient index {transient_index} is out of range. "
              f"There are only {len(my_transients)} transients.")