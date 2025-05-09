#!/usr/bin/env python3

import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath('..'))

try:
    from sign_language_translation.asl_functions import load_frame_data, find_interpolation_frames, find_file_with_partial_name, has_numbers_on_both_sides, modify_npz_file, interpolate_undetected_hand_landmarks_new, interpolate_undetected_hand_landmarks_new_wrapper, process_landmarks_dataframe_new, find_latest_checkpoint, find_landmark_directories
except ImportError as e:
    sys.exit(1)


def main():
    try:
        data = find_landmark_directories("./Open_asl_all_data/clips")
        print("dataframe made")
        #last_row = find_latest_checkpoint("./report")[0]
        df_after_interpolation = process_landmarks_dataframe_new(dataframe=data, interpolate_func=interpolate_undetected_hand_landmarks_new_wrapper, batch_size=4, report_dir='./report', start_from_row=0)
        df_after_interpolation.to_csv('df_after_interpolation.csv', index=False)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()