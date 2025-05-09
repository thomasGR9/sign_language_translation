#!/usr/bin/env python3

import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath('..'))

try:
    from sign_language_translation.asl_functions import batch_process_videos, process_video_new_2, batch_call_process_video_validated
except ImportError as e:
    sys.exit(1)

def main():
    try:
        # Load the dataframe
        df_path = "final_df.csv"
        if not os.path.exists(df_path):
            sys.exit(1)
            
        resampled_renamed_df = pd.read_csv(df_path)
        
        # Check if dataframe is loaded correctly
        if resampled_renamed_df.empty:
            sys.exit(1)
            
        row_count = len(resampled_renamed_df)
        
        # Load current processing report
        report_path = "./report/video_processing_report_current.json"
        if not os.path.exists(report_path):
            print("Couldn't find video_processing_report_current.json file")
            sys.exit(1)
            
        with open(report_path, "r") as f:
            current_report = json.load(f)
            last_row = current_report["processing_info"]["last_processed_row"]
            next_row = last_row + 1
        
        # Check if next_row is valid
        if next_row >= row_count:
            sys.exit(0)
            
            

        
        # Call the batch processing function
        batch_process_videos(
            video_df=resampled_renamed_df, 
            process_video_func=batch_call_process_video_validated, 
            detection_threshold_dom=70, 
            detection_threshold_non_dom=50, 
            detection_threshold_dom_small_length=30, 
            detection_threshold_non_dom_small_length=0, 
            report_dir="./report", 
            start_time_seconds=0, 
            end_time_seconds=None, 
            delete_videos=True, 
            start_from_row=next_row
        )
        

        
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()
