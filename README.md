## Project Files

- **asl_citizen_csv.ipynb** - Notebook processing the ASL Citizen dataset
- **asl_functions.py** - All the functions used to preprocess the data in the format that the model will see
- **distilled_gpt.ipynb** - Notebook testing out Distilled-GPT-2 embedding space and tokenizer. Here the embedding-informed label smoothing labels are created
- **interpolate_open.py** - Script that interpolates all the missing frames in given videos
- **process_videos.py** - Script that deletes videos based on hand detection thresholds
- **resample_clips.py** - Script that lowers the FPS of given videos (via sampling from frames)
- **resume.py** - Script to resume (or start new) training runs for the model
- **test.ipynb** - Notebook preprocessing Open ASL dataset
- **translation_model_functions.py** - Contains all the needed functions to build, load, train, and validate the model
- **velocities_calc.py** - Script that calculates the velocity features from the videos
