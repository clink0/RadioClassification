import numpy as np
import matplotlib.pyplot as plt
import os
import math

# --- Core Detection Function ---
def find_transient_ranges(file_path, chunk_size, window_size, start_slope_threshold, plateau_slope_threshold,
                          off_amplitude_threshold):
    """
    Scans a .cf32 file to find the start and end of all transient ramps
    using a 3-state machine.

    Args:
        file_path (str): Path to the .cf32 file.
        chunk_size (int): The number of samples to process at a time to manage memory.
        window_size (int): The size of the sliding window for slope calculation.
        start_slope_threshold (float): The positive slope value that triggers a "start" detection.
        plateau_slope_threshold (float): The slope value that indicates the ramp has ended.
        off_amplitude_threshold (float): The amplitude below which the signal is considered "off".

    Returns:
        list of tuples: A list where each tuple is (start_sample, end_sample).
    """
    transient_ranges = []
    total_samples_processed = 0
    overlap_size = window_size * 3
    state = 'searching_for_start'
    temp_start_index = 0

    try:
        with open(file_path, 'rb') as f:
            print(f"--- Starting detailed scan with state: {state} ---")
            while True:
                # This logic reads the file chunk by chunk
                seek_pos_bytes = max(0, total_samples_processed - overlap_size) * 8
                f.seek(seek_pos_bytes)
                count_to_read = (chunk_size + overlap_size) * 2
                raw_data = np.fromfile(f, dtype=np.float32, count=count_to_read)
                if raw_data.size < window_size * 2:
                    print("\nEnd of file reached.")
                    break

                complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                amplitude_chunk = np.abs(complex_data)
                x = np.arange(window_size)
                i = 0
                while i < len(amplitude_chunk) - window_size:
                    chunk_start_sample = max(0, total_samples_processed - overlap_size)
                    absolute_index = chunk_start_sample + i

                    if state == 'searching_for_start':
                        window_data = amplitude_chunk[i: i + window_size]
                        try:
                            slope, _ = np.polyfit(x, window_data, 1)
                            if slope > start_slope_threshold:
                                print(f"\nTransient start detected at sample ~{absolute_index:,}")
                                temp_start_index = absolute_index
                                state = 'searching_for_plateau'
                                i += window_size
                                continue
                        except np.linalg.LinAlgError:
                            pass
                    elif state == 'searching_for_plateau':
                        window_data = amplitude_chunk[i: i + window_size]
                        try:
                            slope, _ = np.polyfit(x, window_data, 1)
                            if slope < plateau_slope_threshold:
                                transient_ranges.append((temp_start_index, absolute_index))
                                print(f"Transient end (plateau) detected at sample ~{absolute_index:,}")
                                state = 'waiting_for_signal_off'
                                i += window_size
                                continue
                        except np.linalg.LinAlgError:
                            pass
                    elif state == 'waiting_for_signal_off':
                        if amplitude_chunk[i] < off_amplitude_threshold:
                            state = 'searching_for_start'
                        i += 50
                        continue
                    i += 1

                effective_samples_in_chunk = len(amplitude_chunk) - overlap_size
                if effective_samples_in_chunk <= 0:
                    break
                total_samples_processed += chunk_size
                print(f"\rDetailed scan in progress... Scanned up to sample {total_samples_processed:,}...", end="")

        print("\nFinished detailed scan.")
        return transient_ranges

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


# --- Data Loading and Saving Functions ---
def load_transient_samples(file_path, ranges):
    """
    Loads the raw complex IQ data for each detected transient range.
    """
    transient_data = []
    try:
        with open(file_path, 'rb') as f:
            for i, (start, end) in enumerate(ranges):
                seek_pos_bytes = start * 8
                num_samples_to_read = end - start
                f.seek(seek_pos_bytes)
                raw_data = np.fromfile(f, dtype=np.float32, count=num_samples_to_read * 2)
                complex_samples = raw_data[0::2] + 1j * raw_data[1::2]
                transient_data.append(complex_samples)
                print(f"\rLoading sample data... {i + 1}/{len(ranges)}", end="")
        print("\nFinished loading sample data.")
        return transient_data
    except Exception as e:
        print(f"\nAn unexpected error occurred during sample loading: {e}")
        return []


def save_transients_to_npz(file_path, transients_list):
    """
    Saves a list of NumPy arrays to a single compressed .npz file.
    """
    transient_dict = {f'transient_{i}': arr for i, arr in enumerate(transients_list)}
    try:
        np.savez_compressed(file_path, **transient_dict)
        print(f"\nSuccessfully saved {len(transients_list)} transients to '{file_path}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    pass


# --- NEW Plotting Function ---
def plot_first_transients(file_path, ranges, num_to_plot=10):
    """
    Reads the data for the first few detected transients and plots their amplitude.

    Args:
        file_path (str): The path to the .cf32 file.
        ranges (list of tuples): The list of detected (start, end) sample ranges.
        num_to_plot (int): The maximum number of transients to plot.
    """
    print("\n--- Generating verification plots ---")
    num_to_plot = min(len(ranges), num_to_plot)
    if num_to_plot == 0:
        print("No ranges to plot.")
        return

    # Determine grid size for subplots
    ncols = 2
    nrows = math.ceil(num_to_plot / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    with open(file_path, 'rb') as f:
        for i in range(num_to_plot):
            ax = axes[i]
            start_sample, end_sample = ranges[i]

            # Define a plotting window with padding around the detected range
            plot_padding = (end_sample - start_sample) * 2
            plot_start = max(0, start_sample - plot_padding)
            plot_end = end_sample + plot_padding
            samples_to_read = plot_end - plot_start

            f.seek(plot_start * 8)
            raw_data = np.fromfile(f, dtype=np.float32, count=samples_to_read * 2)

            if raw_data.size > 0:
                complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                amplitude = np.abs(complex_data)
                time_axis = np.arange(len(amplitude)) + plot_start

                ax.plot(time_axis, amplitude, label='Signal Amplitude', color='dodgerblue')
                ax.axvline(x=start_sample, color='g', linestyle='--', label=f'Start ({start_sample})')
                ax.axvline(x=end_sample, color='r', linestyle='--', label=f'End ({end_sample})')
                ax.set_title(f'Transient #{i + 1}')
                ax.set_xlabel('Sample Number')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True)

    # Hide any unused subplots
    for j in range(num_to_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Verification of First Detected Transients', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("transient_verification_plots.png")
    print("Verification plots saved to 'transient_verification_plots.png'")
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    file_to_read = '/home/luke/SDR/Luke/GnuRadio Companion Flowcharts/SimpleCosine/SimpleCosine.cf32'
    output_npz_file = 'detected_transients_sdr2_4.npz'

    # --- MANUAL Detector Parameters (Tune these for your data) ---
    CHUNK_SIZE = 500000
    WINDOW_SIZE = 40
    START_SLOPE_THRESHOLD = 0.0033
    PLATEAU_SLOPE_THRESHOLD = 0.00001
    AMPLITUDE_OFF_THRESHOLD = 0.2

    # --- Run the Simplified Workflow ---

    # Step 1: Run the main, detailed scan from the beginning of the file.
    detected_ranges = find_transient_ranges(
        file_path=file_to_read,
        chunk_size=CHUNK_SIZE,
        window_size=WINDOW_SIZE,
        start_slope_threshold=START_SLOPE_THRESHOLD,
        plateau_slope_threshold=PLATEAU_SLOPE_THRESHOLD,
        off_amplitude_threshold=AMPLITUDE_OFF_THRESHOLD
    )

    if detected_ranges:
        print(f"\nFound {len(detected_ranges)} transients.")

        # Step 2: Plot the first 10 transients for visual verification
        plot_first_transients(file_to_read, detected_ranges, num_to_plot=10)

        # Step 3: Load the sample data for each detected transient.
        print(f"\n--- Loading sample data ---")
        all_transient_samples = load_transient_samples(file_to_read, detected_ranges)

        if all_transient_samples:
            # Step 4: Export the loaded samples to a file.
            print("\n--- Exporting loaded samples ---")
            save_transients_to_npz(output_npz_file, all_transient_samples)

    else:
        print("\nNo transients were detected with the current parameters.")
