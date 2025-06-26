import numpy as np
import matplotlib.pyplot as plt
import os


def plot_file_quarters(file_path):
    """
    Divides a .cf32 file into four equal quarters and generates a separate
    amplitude plot for each quarter.

    Args:
        file_path (str): The full path to the .cf32 file.
    """
    try:
        # --- Step 1: Determine the file size and quarter points ---
        print(f"Analyzing file: {file_path}")
        file_size_bytes = os.path.getsize(file_path)
        # Each IQ sample is two 4-byte floats (8 bytes total)
        total_samples = file_size_bytes // 8

        if total_samples == 0:
            print("Error: File is empty or not a valid .cf32 file.")
            return

        quarter_size_samples = total_samples // 4
        print(f"Total samples in file: {total_samples:,}")
        print(f"Samples per quarter:   {quarter_size_samples:,}")

        # Define the start and number of samples for each quarter
        quarters = [
            {"quarter_num": 1, "start_sample": 0, "num_samples": quarter_size_samples},
            {"quarter_num": 2, "start_sample": quarter_size_samples, "num_samples": quarter_size_samples},
            {"quarter_num": 3, "start_sample": 2 * quarter_size_samples, "num_samples": quarter_size_samples},
            # The last quarter takes the rest of the samples to handle rounding
            {"quarter_num": 4, "start_sample": 3 * quarter_size_samples,
             "num_samples": total_samples - (3 * quarter_size_samples)}
        ]

    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'")
        return
    except Exception as e:
        print(f"An error occurred calculating file size: {e}")
        return

    # --- Step 2: Loop through each quarter, read data, and plot ---
    with open(file_path, 'rb') as f:
        for q_info in quarters:
            quarter_num = q_info["quarter_num"]
            start_sample = q_info["start_sample"]
            num_samples = q_info["num_samples"]

            print(f"\nProcessing Quarter {quarter_num} (starting at sample {start_sample:,})...")

            try:
                # Seek to the start of the quarter
                # Each sample is 8 bytes
                f.seek(start_sample * 8)

                # Read the data for the entire quarter
                # Note: This can use a lot of memory for large files!
                raw_data = np.fromfile(f, dtype=np.float32, count=num_samples * 2)

                if raw_data.size == 0:
                    print(f"Warning: No data read for Quarter {quarter_num}.")
                    continue

                # Convert to amplitude
                complex_data = raw_data[0::2] + 1j * raw_data[1::2]
                amplitude = np.abs(complex_data)

                # Create the x-axis (sample indices)
                sample_indices = np.arange(start_sample, start_sample + len(amplitude))

                # --- Plotting ---
                plt.figure(figsize=(15, 7))
                plt.plot(sample_indices, amplitude, color='darkcyan', linewidth=0.5)

                plt.title(f'Amplitude Plot - Quarter {quarter_num} of 4', fontsize=16)
                plt.xlabel('Absolute Sample Number', fontsize=12)
                plt.ylabel('Amplitude', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                # Use scientific notation for large sample numbers
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

                # Save the plot to a file
                output_filename = f'file_quarter_{quarter_num}.png'
                plt.savefig(output_filename)
                print(f"Plot saved to '{output_filename}'")

                # Close the plot to free up memory before processing the next quarter
                plt.close()

            except Exception as e:
                print(f"An error occurred while processing Quarter {quarter_num}: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Update this path to your large .cf32 file
    file_to_analyze = '/home/luke/SDR/Luke/GnuRadio Companion Flowcharts/SimpleCosine/SimpleCosine.cf32'

    # --- Run the analysis ---
    plot_file_quarters(file_to_analyze)

    print("\nAnalysis complete.")
