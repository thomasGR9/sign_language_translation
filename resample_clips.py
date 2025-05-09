#!/usr/bin/env python3

import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath('..'))

try:
    from sign_language_translation.asl_functions import resample_single_video, resample_videos_in_dataframe, make_videos_df
except ImportError as e:
    sys.exit(1)


def main():
    try:
        data = pd.read_csv("./videos_df.csv")
        print("Loaded dataframe")
        if sum(data['Frame Count']>0) != data.shape[0]:
            print("Some videos don't contain frames, exiting ...")
            sys.exit(1)
        df_after_resampling = resample_videos_in_dataframe(df=data, desired_fps=15, batch_size=4, save_checkpoint=False, checkpoint_file='resampling_progress.csv', file_path_col='file_path', delete_originals=True)
        df_after_resampling.to_csv('df_after_resampling.csv', index=False)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()