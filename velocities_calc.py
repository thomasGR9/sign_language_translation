#!/usr/bin/env python3

import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath('..'))

try:
    from sign_language_translation.asl_functions import is_data_frame_good, is_valid_detection, has_value, cartesian_to_spherical, cartesian_to_polar_features, sorted_npz_files_checked, compute_landmark_velocities, compute_landmark_velocities_wrapper, process_landmark_velocities_parallel, find_landmark_directories, find_latest_velocity_checkpoint
except ImportError as e:
    sys.exit(1)


def main():
    try:
        data = find_landmark_directories("./Open_asl_all_data/clips")
        print("Dataframe done")
        #last_row = find_latest_velocity_checkpoint("./report")[0]
        df_after_speed = process_landmark_velocities_parallel(dataframe=data, velocity_func=compute_landmark_velocities_wrapper, batch_size=4, report_dir="./report", start_from_row=0)
        df_after_speed.to_csv('df_after_speed.csv', index=False)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()