asl_citizen_csv.ipynb: Notebook processing the asl_citizen dataset
asl_functions.py: All the used function that preprocess the data in the format that the model will see
distilled_gpt.ipynb: Notebook testing out Distilled-GPT-2 embedding space and tokenizer. Here the Embedding informed label smoothing labels would be created.
interpolate_open.py: Script that interpolates all the missing frames in given videos
process_videos.py: Script that deletes videos based on hand's detection thresholds
resample_clips.py: Script that lowers the fps of given videos (via sampling from frames)
resume.py: Script to resume (or start new) training run for the model
test.ipynb: Notebook preprocessing Open ASL dataset
translation_model_functions.py: Contains all the needed functions to build, load, train, validate the model
velocities_calc.py: Script that calculated the velocity features from the videos
