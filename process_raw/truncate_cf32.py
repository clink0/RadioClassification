import os
import sys


def truncate_cf32_file(file_path, samples_to_remove):
    """
    Removes a specified number of samples from the beginning of a .cf32 file
    by reading the remaining data and overwriting the original file.

    *** WARNING: THIS FUNCTION PERMANENTLY MODIFIES THE FILE. ***

    Args:
        file_path (str): The full path to the .cf32 file to be modified.
        samples_to_remove (int): The number of IQ samples to remove from the start.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return False

    # Each IQ sample is two 4-byte floats (float32), so 8 bytes total.
    bytes_per_sample = 8
    bytes_to_skip = samples_to_remove * bytes_per_sample

    try:
        # --- Safety Check and Information Display ---
        file_size_bytes = os.path.getsize(file_path)

        # --- FIX: Define total_samples right after getting the file size ---
        total_samples = file_size_bytes // bytes_per_sample

        if bytes_to_skip >= file_size_bytes:
            print(f"Error: Cannot remove {samples_to_remove:,} samples ({bytes_to_skip:,} bytes) "
                  f"as it is more than or equal to the total file size of {file_size_bytes:,} bytes.")
            return False

        print(f"File: '{os.path.basename(file_path)}'")
        print(f"Current size: {file_size_bytes / 1e6:.2f} MB ({total_samples:,} samples)")
        print(f"Samples to remove: {samples_to_remove:,} ({bytes_to_skip / 1e6:.2f} MB)")
        new_total_samples = (file_size_bytes - bytes_to_skip) // bytes_per_sample
        print(f"New estimated size: {(file_size_bytes - bytes_to_skip) / 1e6:.2f} MB ({new_total_samples:,} samples)")

        # --- Read the data that will be kept ---
        print("\nReading data to keep...")
        with open(file_path, 'rb') as f:
            # Move the file pointer past the part we want to remove
            f.seek(bytes_to_skip)
            # Read the rest of the file into memory
            remaining_data = f.read()

        print("Data read successfully.")

        # --- Overwrite the original file ---
        print("Overwriting original file...")
        with open(file_path, 'wb') as f:
            # Writing in 'wb' mode automatically truncates the file before writing
            f.write(remaining_data)

        print("File successfully truncated.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Set this to the path of the file you want to modify.
    file_to_modify = '/home/luke/SDR/Luke/GnuRadio Companion Flowcharts/SimpleCosine/SimpleCosine.cf32'

    # Set the number of IQ samples you want to remove from the beginning.
    # For example, to remove the first 10 million samples:
    SAMPLES_TO_REMOVE = 13000000

    # --- Run the Truncation ---
    if SAMPLES_TO_REMOVE > 0:
        truncate_cf32_file(file_to_modify, SAMPLES_TO_REMOVE)
    else:
        print("SAMPLES_TO_REMOVE is set to 0. No changes will be made.")
