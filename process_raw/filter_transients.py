from filter_transients_func import *

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
