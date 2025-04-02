hand_model_path = "hand_landmarker.task"
face_model_path = "face_landmarker.task"

import os
import logging


import os
import sys
import contextlib
import tempfile

class SuppressOutput:
    """
    A context manager that suppresses stdout and stderr from C/C++ libraries.
    This is a more aggressive approach than just Python's logging or environment variables.
    """
    def __init__(self):
        # Create a temporary file to redirect output
        self.null_fds = [tempfile.TemporaryFile(mode='w+b') for _ in range(2)]
        # Save the original file descriptors to restore later
        self.save_fds = [os.dup(1), os.dup(2)]
        
    def __enter__(self):
        # Redirect stdout and stderr to the null files
        os.dup2(self.null_fds[0].fileno(), 1)
        os.dup2(self.null_fds[1].fileno(), 2)
        # Also redirect Python-level stdout/stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, *args):
        # Restore normal stdout and stderr
        for fd in self.null_fds:
            fd.close()
        for fd in range(2):
            os.dup2(self.save_fds[fd], fd + 1)
        for fd in self.save_fds:
            os.close(fd)
        # Restore Python-level stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

try:
    import ctypes
    libc = ctypes.CDLL(None)
    # Attempt to get C's stdout/stderr file descriptor
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), c_stdout.value)
    os.dup2(devnull.fileno(), c_stderr.value)
except:
    # If this approach fails, continue with other methods
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['GLOG_minloglevel'] = '3'      # Suppress Google logging (used by MediaPipe)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Optional: Disable GPU logging messages

os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# ABSL specific flags
os.environ['ABSL_LOGGING_LEVEL'] = '50'  # Higher than any level that should be output
os.environ['PYTHONWARNINGS'] = 'ignore'

logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)\

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time
from pathlib import Path
import tempfile
import json
from datetime import datetime
import pandas as pd
import shutil
import glob
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 




def detect(image_path, hand_model_path, face_model_path, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, num_hands=2, dominand_hand='Right', visualize=False, output_face_blendshapes=True, adaptive_threshold=True, max_attempts=3, threshold_reduction_factor=0.7, min_threshold=0.2):
    """
    Detects hands and face in an image, extracts hand landmark coordinates and face blendshapes.
    
    Args:
        image_path (str): Path to the image file
        min_hand_detection_confidence (float): Confidence threshold for hand detection (0.0-1.0)
        min_hand_presence_confidence (float): Confidence threshold for hand presence (0.0-1.0)
        num_hands (int): Maximum number of hands to detect
        dominand_hand (str): Dominant hand preference ('Left' or 'Right')
        visualize (bool): Whether to visualize the results
        output_face_blendshapes (bool): Whether to detect and extract face blendshapes
        
    Returns:
        tuple: (dom_landmarks, non_dom_landmarks, wrists, confidence_scores, detection_status, 
                blendshape_scores, face_landmark_5, face_detected)
               - dom_landmarks: NumPy array of shape [20, 3] with coordinates of dominant hand landmarks
               - non_dom_landmarks: NumPy array of shape [20, 3] with coordinates of non-dominant hand landmarks
               - wrists: NumPy array of shape [2, 2] with coordinates of both wrists [x, y]
               - confidence_scores: NumPy array of shape [2] with confidence scores [dominant_hand, non_dominant_hand]
               - detection_status: NumPy array of shape [2] with binary detection status [dominant_hand, non_dominant_hand]
               - blendshape_scores: NumPy array of shape [26] with selected face blendshape scores
               - face_landmark_5: NumPy array of shape [2] with coordinates of the 5th face landmark [x, y]
               - face_detected: Binary value (1 if face detected, 0 if not)
    """
    # Initialize output arrays for face detection
    blendshape_scores = np.zeros(52)
    nose_landmark = np.zeros(2)
    left_eye_landmark = np.zeros(2)
    right_eye_landmark = np.zeros(2)
    face_detected = 0
    
    # PART 1: HAND LANDMARK DETECTION
    # 1.1: Configure the hand landmarker
    hand_base_options = python.BaseOptions(
        model_asset_path=hand_model_path
    )

    VisionRunningMode = mp.tasks.vision.RunningMode
    # Configure detection options
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=num_hands,                             
        min_hand_detection_confidence=min_hand_detection_confidence,       
        min_hand_presence_confidence=min_hand_presence_confidence,        
        min_tracking_confidence=0.5,             
        running_mode=VisionRunningMode.IMAGE
    )

    # Create the hand detector
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # 1.2: Load the input image
    image = mp.Image.create_from_file(image_path)

    # 1.3: Detect hand landmarks
    hand_detection_result = hand_detector.detect(image)
    
    # Initialize hand output arrays with zeros
    dom_landmarks = np.zeros((20, 3))       # 20 landmarks (excluding wrist), [x,y,z]
    non_dom_landmarks = np.zeros((20, 3))   # 20 landmarks (excluding wrist), [x,y,z]
    wrists = np.zeros((2, 2))               # 2 wrists, [x,y]
    confidence_scores = np.zeros(2)         # Confidence scores for [dominant, non-dominant]
    interpolation_scores = np.zeros(2) #Interpolation scores for [dominant, non-dominant]. Used later.
    detection_status = np.zeros(2, dtype=np.int32)  # Binary detection status [dominant, non-dominant]
    nose_to_wrist_dist = np.zeros((2, 2))
    
    # 1.4: Process hand landmarks if hands are detected
    if hand_detection_result.hand_landmarks and hand_detection_result.handedness:
        dom_hand_found = False
        non_dom_hand_found = False
        
        # First, find the dominant and non-dominant hands in detection results
        for idx, handedness in enumerate(hand_detection_result.handedness):
            hand_type = handedness[0].category_name  # 'Left' or 'Right'
            hand_score = handedness[0].score  # Confidence score for the handedness classification
            
            if hand_type == dominand_hand:
                # This is the dominant hand
                dom_hand_found = True
                detection_status[0] = 1  # Set detection status to 1 (detected)
                confidence_scores[0] = hand_score  # Store confidence score
                interpolation_scores[0] = 1
                
                # Store dominant hand wrist coordinates [x,y]
                dom_hand_landmarks = hand_detection_result.hand_landmarks[idx]
                wrists[0, 0] = dom_hand_landmarks[0].x
                wrists[0, 1] = dom_hand_landmarks[0].y
                
                # Store all other dominant hand landmarks (excluding wrist)
                for i in range(1, 21):  # Landmarks 1-20 (skipping wrist which is index 0)
                    dom_landmarks[i-1, 0] = dom_hand_landmarks[i].x
                    dom_landmarks[i-1, 1] = dom_hand_landmarks[i].y
                    dom_landmarks[i-1, 2] = dom_hand_landmarks[i].z
                    
            elif hand_type != dominand_hand:
                # This is the non-dominant hand
                non_dom_hand_found = True
                detection_status[1] = 1  # Set detection status to 1 (detected)
                confidence_scores[1] = hand_score  # Store confidence score
                interpolation_scores[1] = 1
                
                # Store non-dominant hand wrist coordinates [x,y]
                non_dom_hand_landmarks = hand_detection_result.hand_landmarks[idx]
                wrists[1, 0] = non_dom_hand_landmarks[0].x
                wrists[1, 1] = non_dom_hand_landmarks[0].y
                
                # Store all other non-dominant hand landmarks (excluding wrist)
                for i in range(1, 21):  # Landmarks 1-20 (skipping wrist)
                    non_dom_landmarks[i-1, 0] = non_dom_hand_landmarks[i].x
                    non_dom_landmarks[i-1, 1] = non_dom_hand_landmarks[i].y
                    non_dom_landmarks[i-1, 2] = non_dom_hand_landmarks[i].z
                    
        # Log information about which hands were found
        print(f"Dominant hand ({dominand_hand}) detected: {dom_hand_found}")
        print(f"Non-dominant hand detected: {non_dom_hand_found}")
    

   # PART 2: FACE LANDMARK DETECTION (If requested)
    if output_face_blendshapes:
        try:
            # 2.1: Configure the face landmarker
            face_base_options = python.BaseOptions(
                model_asset_path=face_model_path
            )
            
            # Configure face detection options
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                min_face_detection_confidence=min_face_detection_confidence,
                min_face_presence_confidence=min_face_presence_confidence,
                output_face_blendshapes=True,
                num_faces=1,
                running_mode=VisionRunningMode.IMAGE
            )
            
            # Create the face detector
            face_detector = vision.FaceLandmarker.create_from_options(face_options)
            
            # 2.2: Detect face landmarks (reuse the same image)
            face_detection_result = face_detector.detect(image)
            
            # 2.3: Process face blendshapes if face is detected
            if (face_detection_result.face_blendshapes and len(face_detection_result.face_blendshapes) > 0 and
                face_detection_result.face_landmarks and len(face_detection_result.face_landmarks) > 0):
                
                # Set face detected flag to 1
                face_detected = 1
                
                # Get all blendshapes from the first face
                all_blendshapes = face_detection_result.face_blendshapes[0]
                
                # Initialize blendshape_scores with the correct size to hold all blendshapes
                # Assuming MediaPipe returns all 52 blendshapes
                blendshape_scores = np.zeros(len(all_blendshapes))
                
                # Fill the blendshape_scores array with ALL scores
                for i in range(len(all_blendshapes)):
                    blendshape_scores[i] = all_blendshapes[i].score
                
                # Get nose coordinates
                nose = face_detection_result.face_landmarks[0][4]
                nose_landmark[0] = nose.x
                nose_landmark[1] = nose.y
    
                # Get eye coordinates
                left_eye = face_detection_result.face_landmarks[0][473]
                left_eye_landmark[0] = left_eye.x
                left_eye_landmark[1] = left_eye.y
    
                right_eye = face_detection_result.face_landmarks[0][468]
                right_eye_landmark[0] = right_eye.x
                right_eye_landmark[1] = right_eye.y
            
        except Exception as e:
            print(f"Error during face detection: {e}")
            # Keep default zero values for face outputs if detection fails
    
    
    
    # PART 3: VISUALIZATION
    if visualize:
        # Load the image with OpenCV for visualization
        img_cv = cv2.imread(image_path)
        img_height, img_width, _ = img_cv.shape

        # 3.1: Draw hand landmarks if hands are detected
        if hand_detection_result.hand_landmarks:
            print(f"Visualizing {len(hand_detection_result.hand_landmarks)} hands")
            
            # Define connections between landmarks for hand skeleton
            connections = [
                # Thumb connections
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger connections
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger connections
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger connections
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky finger connections
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm connections
                (0, 5), (5, 9), (9, 13), (13, 17)
            ]
            
            for idx, hand_landmarks in enumerate(hand_detection_result.hand_landmarks):
                # Determine if this is the dominant hand
                is_dominant = False
                if hand_detection_result.handedness:
                    hand_type = hand_detection_result.handedness[idx][0].category_name
                    is_dominant = (hand_type == dominand_hand)
                
                # Use different colors for dominant vs non-dominant hand
                hand_color = (0, 0, 255) if is_dominant else (255, 0, 0)  # Blue for dominant, Red for non-dominant
                
                # Draw all landmark points
                for landmark in hand_landmarks:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * img_width)
                    y = int(landmark.y * img_height)
                    
                    # Draw the landmark point
                    cv2.circle(img_cv, (x, y), 5, hand_color, -1)
                
                # Draw connections between landmarks (hand skeleton)
                for connection in connections:
                    start_idx, end_idx = connection
                    
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_point = hand_landmarks[start_idx]
                        end_point = hand_landmarks[end_idx]
                        
                        # Convert normalized coordinates to pixel coordinates
                        start_x = int(start_point.x * img_width)
                        start_y = int(start_point.y * img_height)
                        end_x = int(end_point.x * img_width)
                        end_y = int(end_point.y * img_height)
                        
                        # Draw the connection line
                        cv2.line(img_cv, (start_x, start_y), (end_x, end_y), hand_color, 2)
                
                # Add hand type label (Left/Right, Dominant/Non-dominant)
                if hand_detection_result.handedness:
                    handedness = hand_detection_result.handedness[idx]
                    hand_type = handedness[0].category_name  # 'Left' or 'Right'
                    hand_score = handedness[0].score
                    dom_status = "Dominant" if hand_type == dominand_hand else "Non-dominant"
                    cv2.putText(img_cv, f"{hand_type} Hand - {dom_status} ({hand_score:.2f})", 
                            (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, hand_color, 2)
                    
                    # Calculate and draw a bounding box
                    x_coords = [landmark.x for landmark in hand_landmarks]
                    y_coords = [landmark.y for landmark in hand_landmarks]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # Convert to pixel coordinates
                    min_x, max_x = int(min_x * img_width), int(max_x * img_width)
                    min_y, max_y = int(min_y * img_height), int(max_y * img_height)
                    
                    # Draw bounding box
                    cv2.rectangle(img_cv, (min_x, min_y), (max_x, max_y), hand_color, 2)

        # 3.2: Draw Nose if face was detected
        if face_detected == 1:
            # Convert normalized coordinates to pixel coordinates
            face_x = int(nose_landmark[0] * img_width)
            face_y = int(nose_landmark[1] * img_height)
            
            # Draw the Nose with a distinctive color and size
            cv2.circle(img_cv, (face_x, face_y), 8, (0, 255, 255), -1)  # Yellow circle
            cv2.putText(img_cv, "Nose", (face_x + 10, face_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw eyes
            left_eye_x = int(left_eye_landmark[0] * img_width)
            left_eye_y = int(left_eye_landmark[1] * img_height)
            right_eye_x = int(right_eye_landmark[0] * img_width)
            right_eye_y = int(right_eye_landmark[1] * img_height)
            
            cv2.circle(img_cv, (left_eye_x, left_eye_y), 6, (255, 255, 0), -1)  # Cyan circle
            cv2.circle(img_cv, (right_eye_x, right_eye_y), 6, (255, 255, 0), -1)  # Cyan circle
            cv2.line(img_cv, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (255, 255, 0), 2)
        # 3.3: Add detection status information to visualization
        y_pos = img_height - 80
        hand_status_text = f"Hand Detection: Dom={detection_status[0]}, Non-Dom={detection_status[1]}"
        hand_conf_text = f"Hand Confidence: Dom={confidence_scores[0]:.2f}, Non-Dom={confidence_scores[1]:.2f}"
        face_status_text = f"Face Detection: {face_detected}"
        
        cv2.putText(img_cv, hand_status_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_cv, hand_conf_text, (10, y_pos + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_cv, face_status_text, (10, y_pos + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3.4: Display the result
        cv2.imshow('Hand and Face Landmarks', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    if face_detected==1:
        #Calculate distance between the eyes
        eyes_diff = right_eye_landmark-left_eye_landmark
        eyes_distance = np.sqrt(eyes_diff.dot(eyes_diff))
        if detection_status[0]==1 and detection_status[1]==1:
            nose_to_wrist_dist = (wrists-nose_landmark) / eyes_distance
            #Make every hand's landmark potision relative to the wrist, and scaled by the eye's distance
            dom_landmarks[:, 0:2] = (dom_landmarks[:, 0:2] - wrists[0, :]) / eyes_distance
            non_dom_landmarks[:, 0:2] = (non_dom_landmarks[:, 0:2] - wrists[1, :]) / eyes_distance
        elif detection_status[0]==1 and detection_status[1]==0:
            nose_to_wrist_dist[0, :] = (wrists[0, :]-nose_landmark) / eyes_distance
            #Make every hand's landmark potision relative to the wrist, and scaled by the eye's distance
            dom_landmarks[:, 0:2] = (dom_landmarks[:, 0:2] - wrists[0, :]) / eyes_distance
        elif detection_status[0]==0 and detection_status[1]==1:
            nose_to_wrist_dist[1,:] = (wrists[1,:]-nose_landmark) / eyes_distance
            #Make every hand's landmark potision relative to the wrist, and scaled by the eye's distance
            non_dom_landmarks[:, 0:2] = (non_dom_landmarks[:, 0:2] - wrists[0, :]) / eyes_distance
        
    elif face_detected==0 and detection_status[0]==1:
        #Calculate palm width distance as fallback scaling factor
        palm_width_diff = dom_landmarks[5, :]- dom_landmarks[17, :]
        palm_width_dist = np.sqrt(palm_width_diff.dot(palm_width_diff))
        if detection_status[1]==1:
            nose_to_wrist_dist = (wrists-nose_landmark) / palm_width_dist
            #Make every hand's landmark potision relative to the wrist, and scaled by the palm width 
            dom_landmarks[:, 0:2] = (dom_landmarks[:, 0:2] - wrists[0, :]) / palm_width_dist
            non_dom_landmarks[:, 0:2] = (non_dom_landmarks[:, 0:2] - wrists[1, :]) / palm_width_dist
        elif detection_status[1]==0:
            nose_to_wrist_dist[0,:] = (wrists[0,:]-nose_landmark) / palm_width_dist
            #Make every hand's landmark potision relative to the wrist, and scaled by the palm width 
            dom_landmarks[:, 0:2] = (dom_landmarks[:, 0:2] - wrists[0, :]) / palm_width_dist
    elif face_detected==0 and detection_status[0]==0 and detection_status[1]==1:
        #Calculate palm width distance as fallback scaling factor
        palm_width_diff = non_dom_landmarks[5, :]- non_dom_landmarks[17, :]
        palm_width_dist = np.sqrt(palm_width_diff.dot(palm_width_diff))
        nose_to_wrist_dist[1,:] = (wrists[1,:]-nose_landmark) / palm_width_dist
        #Make every hand's landmark potision relative to the wrist, and scaled by the palm width 
        non_dom_landmarks[:, 0:2] = (non_dom_landmarks[:, 0:2] - wrists[1, :]) / palm_width_dist
    

    
    # Return all requested outputs
    return dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist



def adaptive_detect(image_path, hand_model_path, face_model_path, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, 
                   min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, 
                   num_hands=2, dominand_hand='Right', visualize=False, output_face_blendshapes=True,
                   max_attempts=3, threshold_reduction_factor=0.7, min_threshold=0.2):
    """
    Adaptively detects hands and face by progressively lowering detection thresholds
    for undetected body parts.
    
    Args:
        image_path (str): Path to the image file
        min_hand_detection_confidence (float): Initial confidence threshold for hand detection
        min_hand_presence_confidence (float): Initial confidence threshold for hand presence
        min_face_detection_confidence (float): Initial confidence threshold for face detection
        min_face_presence_confidence (float): Initial confidence threshold for face presence
        num_hands (int): Maximum number of hands to detect
        dominand_hand (str): Dominant hand preference ('Left' or 'Right')
        visualize (bool): Whether to visualize the final results
        output_face_blendshapes (bool): Whether to detect and extract face blendshapes
        max_attempts (int): Maximum number of detection attempts with lowered thresholds
        threshold_reduction_factor (float): Factor to multiply thresholds by on each attempt (0-1)
        min_threshold (float): Minimum threshold to prevent excessive lowering
        
    Returns:
        Same output as the detect() function
    """
    # Import the original detect function
    #from your_module import detect  # Replace with actual module name
    
    # Store original thresholds
    orig_hand_detection_conf = min_hand_detection_confidence
    orig_hand_presence_conf = min_hand_presence_confidence
    orig_face_detection_conf = min_face_detection_confidence
    orig_face_presence_conf = min_face_presence_confidence
    
    # Initialize best results and detection status
    best_results = None
    best_detection_status = [0, 0]  # [dom_hand, non_dom_hand]
    best_face_detected = 0
    

    
    # Try detection with progressively lower thresholds
    for attempt in range(max_attempts):
        print(f"\n--- Attempt {attempt+1}/{max_attempts} ---")
        
        # Calculate current thresholds
        if attempt > 0:
            # Only lower thresholds for undetected parts
            # For hands
            if best_detection_status[0] == 0:  # Dominant hand not detected
                hand_detection_conf_dom = max(orig_hand_detection_conf * (threshold_reduction_factor ** attempt), min_threshold)
                hand_presence_conf_dom = max(orig_hand_presence_conf * (threshold_reduction_factor ** attempt), min_threshold)
                print(f"Lowering dominant hand thresholds: {hand_detection_conf_dom:.3f}, {hand_presence_conf_dom:.3f}")
            else:
                hand_detection_conf_dom = orig_hand_detection_conf
                hand_presence_conf_dom = orig_hand_presence_conf
                
            if best_detection_status[1] == 0:  # Non-dominant hand not detected
                hand_detection_conf_non_dom = max(orig_hand_detection_conf * (threshold_reduction_factor ** attempt), min_threshold)
                hand_presence_conf_non_dom = max(orig_hand_presence_conf * (threshold_reduction_factor ** attempt), min_threshold)
                print(f"Lowering non-dominant hand thresholds: {hand_detection_conf_non_dom:.3f}, {hand_presence_conf_non_dom:.3f}")
            else:
                hand_detection_conf_non_dom = orig_hand_detection_conf
                hand_presence_conf_non_dom = orig_hand_presence_conf
            
            # Use the minimum of the two calculated thresholds (MediaPipe doesn't support per-hand thresholds)
            current_hand_detection_conf = min(hand_detection_conf_dom, hand_detection_conf_non_dom)
            current_hand_presence_conf = min(hand_presence_conf_dom, hand_presence_conf_non_dom)
            
            # For face
            if output_face_blendshapes and best_face_detected == 0:  # Face not detected
                current_face_detection_conf = max(orig_face_detection_conf * (threshold_reduction_factor ** attempt), min_threshold)
                current_face_presence_conf = max(orig_face_presence_conf * (threshold_reduction_factor ** attempt), min_threshold)
                print(f"Lowering face thresholds: {current_face_detection_conf:.3f}, {current_face_presence_conf:.3f}")
            else:
                current_face_detection_conf = orig_face_detection_conf
                current_face_presence_conf = orig_face_presence_conf
        else:
            # Use original thresholds for first attempt
            current_hand_detection_conf = orig_hand_detection_conf
            current_hand_presence_conf = orig_hand_presence_conf
            current_face_detection_conf = orig_face_detection_conf
            current_face_presence_conf = orig_face_presence_conf
            print(f"Using original thresholds: hands={current_hand_detection_conf}, face={current_face_detection_conf}")
        
        # Call detect with current thresholds (don't visualize intermediate attempts)
        results = detect(image_path,  hand_model_path=hand_model_path, face_model_path=face_model_path,
                        min_hand_detection_confidence=current_hand_detection_conf,
                        min_hand_presence_confidence=current_hand_presence_conf,
                        min_face_detection_confidence=current_face_detection_conf,
                        min_face_presence_confidence=current_face_presence_conf,
                        num_hands=num_hands,
                        dominand_hand=dominand_hand,
                        visualize=False,
                        output_face_blendshapes=output_face_blendshapes)
        
        # Unpack results
        dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist = results
        
        # Compare with best results so far
        current_detection_count = detection_status[0] + detection_status[1] + face_detected
        best_detection_count = best_detection_status[0] + best_detection_status[1] + best_face_detected
        
        if best_results is None or current_detection_count > best_detection_count:
            best_results = results
            best_detection_status = [detection_status[0], detection_status[1]]
            best_face_detected = face_detected
            
            print(f"New best detection: dominant hand={detection_status[0]}, "
                  f"non-dominant hand={detection_status[1]}, face={face_detected}")
            
            # If everything is detected, we can stop early
            if detection_status[0] == 1 and detection_status[1] == 1 and (face_detected == 1 or not output_face_blendshapes):
                print("All body parts detected. Stopping early.")
                break
        else:
            print("No improvement in detection. Continuing to next attempt.")
    
    # Run final detection with visualization if requested
    if visualize:
        print("\n--- Visualizing final results ---")
        # Call detect one more time with the parameters that gave best results, but with visualize=True
        # For simplicity, we'll just use the best thresholds we found
        # This is slightly inefficient (one extra detection) but keeps the code clean
        
        # Determine which thresholds gave the best results
        if best_detection_status[0] == 0:  # If dominant hand not detected in best result
            hand_detection_conf = min_threshold
            hand_presence_conf = min_threshold
        else:
            hand_detection_conf = orig_hand_detection_conf
            hand_presence_conf = orig_hand_presence_conf
            
        if output_face_blendshapes and best_face_detected == 0:  # If face not detected in best result
            face_detection_conf = min_threshold
            face_presence_conf = min_threshold
        else:
            face_detection_conf = orig_face_detection_conf
            face_presence_conf = orig_face_presence_conf
        
        # Run final detection with visualization
        final_results = detect(image_path, hand_model_path=hand_model_path, face_model_path=face_model_path,
                              min_hand_detection_confidence=hand_detection_conf,
                              min_hand_presence_confidence=hand_presence_conf, 
                              min_face_detection_confidence=face_detection_conf,
                              min_face_presence_confidence=face_presence_conf,
                              num_hands=num_hands,
                              dominand_hand=dominand_hand,
                              visualize=True,
                              output_face_blendshapes=output_face_blendshapes)
        
        # Use these results if they're better than our best so far
        dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist = final_results
        current_detection_count = detection_status[0] + detection_status[1] + face_detected
        best_detection_count = best_detection_status[0] + best_detection_status[1] + best_face_detected
        
        if current_detection_count > best_detection_count:
            best_results = final_results
    
    # Print final detection summary
    print("\n=== Detection Summary ===")
    dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist = best_results
    print(f"Dominant hand detected: {detection_status[0] == 1} (confidence: {confidence_scores[0]:.3f})")
    print(f"Non-dominant hand detected: {detection_status[1] == 1} (confidence: {confidence_scores[1]:.3f})")
    if output_face_blendshapes:
        print(f"Face detected: {face_detected == 1}")
    print(f"Total detection attempts: {attempt+1}")
    return best_results



def update_progress(frame_idx, total_frames, timestamp_formatted):
    # Get terminal width to clear the entire line
    terminal_width = shutil.get_terminal_size().columns
    
    # Create progress message
    progress = f"Processing frame {frame_idx}/{total_frames} (timestamp: {timestamp_formatted})"
    
    # Pad with spaces to ensure previous text is overwritten
    padded_progress = progress.ljust(terminal_width)
    
    # Print with carriage return and no newline
    print(f"\r{padded_progress}", end='', flush=True)
    
    # Print a newline when done (call this separately at the end)
    if frame_idx == total_frames:
        print()




def process_video(video_path, adaptive_detect_func=adaptive_detect, hand_model_path=hand_model_path, face_model_path=face_model_path,
                 min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                 min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                 num_hands=2, output_face_blendshapes=True,
                 max_attempts=3, threshold_reduction_factor=0.7, min_threshold=0.2, 
                 frame_step=1, start_time_seconds=0, end_time_seconds=None,
                 save_failure_screenshots=False):
    """
    Process a video frame-by-frame using the adaptive_detect function and save results.
    
    Args:
        video_path (str): Path to the video file
        adaptive_detect_func: The adaptive detection function to use
        min_hand_detection_confidence (float): Initial confidence threshold for hand detection
        min_hand_presence_confidence (float): Initial confidence threshold for hand presence
        min_face_detection_confidence (float): Initial confidence threshold for face detection
        min_face_presence_confidence (float): Initial confidence threshold for face presence
        num_hands (int): Maximum number of hands to detect
        dominand_hand (str): Dominant hand preference ('Left' or 'Right')
        output_face_blendshapes (bool): Whether to detect face blendshapes
        max_attempts (int): Maximum detection attempts for adaptive detection
        threshold_reduction_factor (float): Factor to reduce thresholds by
        min_threshold (float): Minimum threshold limit
        frame_step (int): Process every Nth frame (1 = all frames)
        start_time_seconds (float): Time in seconds to start processing from
        end_time_seconds (float): Time in seconds to end processing (None = process until end)
        save_failure_screenshots (bool): Save screenshots for all frames with any detection failures
        
    Returns:
        str: Path to the directory containing saved frame results
    """
    # Extract video name for directory creation
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_name = video_path.stem  # Get filename without extension
    
    # Extract dominant hand information from filename
    if video_name.endswith("_R"):
        extracted_dominant_hand = "Right"
    elif video_name.endswith("_L"):
        extracted_dominant_hand = "Left"
    else:
        # Default if not specified in filename
        extracted_dominant_hand = "Right"
        print(f"Warning: Could not determine dominant hand from filename, using default: {extracted_dominant_hand}")

    # Use the extracted dominant hand instead of the parameter
    dominand_hand = extracted_dominant_hand
    print(f"Detected dominant hand from filename: {dominand_hand}")

    # Create output directory
    output_dir = video_dir / f"{video_name}_landmarks"
    output_dir.mkdir(exist_ok=True)
    
    # Create screenshots directory if screenshot option is enabled
    screenshots_dir = None
    if save_failure_screenshots:
        screenshots_dir = output_dir / "failure_screenshots"
        screenshots_dir.mkdir(exist_ok=True)
    
    # Create a log file to track processing
    log_file = output_dir / "processing_log.txt"
    
    # Create a detailed statistics file
    stats_file = output_dir / "detection_statistics.json"

    # Initialize statistics tracking
    stats = {
        "video_info": {
            "name": video_name,
            "path": str(video_path),
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0,
            "duration_seconds": 0,
            "start_time": start_time_seconds,
            "end_time": end_time_seconds,
            "dominant_hand": dominand_hand,
            "processing_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_completed": None
        },
        "detection_rates": {
            "dominant_hand": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "non_dominant_hand": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "face": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "overall": {
                "all_detected": 0,
                "partial_detections": 0,
                "no_detections": 0,
                "success_rate": 0
            }
        },
        "failed_frames": {
            "dominant_hand_failures": [],
            "non_dominant_hand_failures": [],
            "face_failures": [],
            "all_failures": []
        },
        "processing_performance": {
            "average_processing_time_ms": 0,
            "total_processing_time_seconds": 0
        }
    }
    
    with open(log_file, "w") as log:
        log.write(f"Processing video: {video_path}\n")
        log.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Parameters:\n")
        log.write(f"  - frame_step: {frame_step}\n")
        log.write(f"  - start_time: {start_time_seconds} seconds\n")
        if end_time_seconds is not None:
            log.write(f"  - end_time: {end_time_seconds} seconds\n")
        log.write(f"  - dominand_hand: {dominand_hand}\n")
        log.write(f"  - num_hands: {num_hands}\n")
        log.write(f"  - detection confidence thresholds: {min_hand_detection_confidence}, {min_face_detection_confidence}\n")
        log.write("\n--- Frame processing log ---\n")
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    # Update stats with video info
    stats["video_info"]["total_frames"] = total_frames
    stats["video_info"]["fps"] = fps
    stats["video_info"]["duration_seconds"] = duration_seconds
    if end_time_seconds==None:
        stats["video_info"]["end_time"] = duration_seconds
    
    # Convert time to frame indices
    start_frame = int(max(0, start_time_seconds * fps))
    
    # Set end frame if specified
    if end_time_seconds is not None:
        end_frame = min(total_frames, int(end_time_seconds * fps))
    else:
        end_frame = total_frames
    
    print(f"Video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Processing frames {start_frame} to {end_frame} (time {start_time_seconds:.2f}s to {end_time_seconds if end_time_seconds is not None else duration_seconds:.2f}s)")
    print(f"Output directory: {output_dir}")
    
    # Process frames
    frame_idx = 0
    processed_count = 0
    total_processing_time = 0
    
    # Skip to start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
    
    with tempfile.TemporaryDirectory() as temp_dir:
        while frame_idx < end_frame:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Only process every frame_step frames
            if (frame_idx - start_frame) % frame_step != 0:
                frame_idx += 1
                continue
                
            # Get timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)
            timestamp_formatted = f"{timestamp_ms//60000:02d}m{(timestamp_ms//1000)%60:02d}s{timestamp_ms%1000:03d}ms"
            
            # Temporary frame path
            temp_frame_path = Path(temp_dir) / f"temp_frame_{frame_idx}.jpg"
            
            # Save the current frame as an image
            cv2.imwrite(str(temp_frame_path), frame)
            
            # Process the frame with adaptive_detect
            update_progress(frame_idx, total_frames, timestamp_formatted)
            
            start_time = time.time()
            try:
                with SuppressOutput():
                # Use adaptive_detect on the frame
                    results = adaptive_detect_func(
                        str(temp_frame_path), hand_model_path, face_model_path,
                        min_hand_detection_confidence=min_hand_detection_confidence,
                        min_hand_presence_confidence=min_hand_presence_confidence,
                        min_face_detection_confidence=min_face_detection_confidence,
                        min_face_presence_confidence=min_face_presence_confidence,
                        num_hands=num_hands,
                        dominand_hand=dominand_hand,
                        visualize=False,
                        output_face_blendshapes=output_face_blendshapes,
                        max_attempts=max_attempts,
                        threshold_reduction_factor=threshold_reduction_factor,
                        min_threshold=min_threshold
                    )
                
                # Calculate processing time
                proc_time = time.time() - start_time
                total_processing_time += proc_time
                
                # Unpack results
                dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist = results
                
                # Update detection statistics
                dom_hand_detected = detection_status[0] == 1
                non_dom_hand_detected = detection_status[1] == 1
                face_was_detected = face_detected == 1
                
                if dom_hand_detected:
                    stats["detection_rates"]["dominant_hand"]["detected"] += 1
                else:
                    stats["detection_rates"]["dominant_hand"]["failed"] += 1
                    stats["failed_frames"]["dominant_hand_failures"].append({
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "file": f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
                    })
                
                if non_dom_hand_detected:
                    stats["detection_rates"]["non_dominant_hand"]["detected"] += 1
                else:
                    stats["detection_rates"]["non_dominant_hand"]["failed"] += 1
                    stats["failed_frames"]["non_dominant_hand_failures"].append({
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "file": f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
                    })
                
                if face_was_detected:
                    stats["detection_rates"]["face"]["detected"] += 1
                else:
                    stats["detection_rates"]["face"]["failed"] += 1
                    stats["failed_frames"]["face_failures"].append({
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "file": f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
                    })
                
                # Track combined detection status
                detection_count = dom_hand_detected + non_dom_hand_detected + face_was_detected
                
                if detection_count == 3:
                    stats["detection_rates"]["overall"]["all_detected"] += 1
                elif detection_count == 0:
                    stats["detection_rates"]["overall"]["no_detections"] += 1
                    stats["failed_frames"]["all_failures"].append({
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "file": f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
                    })
                else:
                    stats["detection_rates"]["overall"]["partial_detections"] += 1
                
                # Save screenshot if any detection failed and screenshots are enabled
                if save_failure_screenshots and (not dom_hand_detected or not non_dom_hand_detected or not face_was_detected):
                    # Create a detailed failure type description for the filename
                    failure_type = []
                    if not dom_hand_detected:
                        failure_type.append("DomHand")
                    if not non_dom_hand_detected:
                        failure_type.append("NonDomHand")
                    if not face_was_detected:
                        failure_type.append("Face")
                    
                    failure_str = "_".join(failure_type)
                    screenshot_filename = f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}_missing_{failure_str}.jpg"
                    screenshot_path = screenshots_dir / screenshot_filename
                    
                    # Copy the frame to the screenshots directory
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"Saved failure screenshot: {screenshot_filename}")
                
                # Create output filename with frame info
                output_filename = f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
                output_path = output_dir / output_filename
                
                # Save all results in a single .npz file
                np.savez(
                    output_path,
                    dom_landmarks=dom_landmarks,
                    non_dom_landmarks=non_dom_landmarks,
                    confidence_scores=confidence_scores,
                    interpolation_scores=interpolation_scores,
                    detection_status=detection_status,
                    blendshape_scores=blendshape_scores,
                    face_detected=face_detected,
                    nose_to_wrist_dist=nose_to_wrist_dist,
                    frame_idx=np.array([frame_idx]),
                    timestamp_ms=np.array([timestamp_ms])
                )
                
                # Update processing log
                detection_summary = f"Dom: {detection_status[0]}, Non-dom: {detection_status[1]}, Face: {face_detected}"
                log_entry = f"Frame {frame_idx}: {detection_summary} (proc time: {proc_time:.2f}s)\n"
                
                with open(log_file, "a") as log:
                    log.write(log_entry)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                with open(log_file, "a") as log:
                    log.write(f"Error on frame {frame_idx}: {str(e)}\n")
            
            # Clean up temporary frame file
            if temp_frame_path.exists():
                temp_frame_path.unlink()
                
            frame_idx += 1
    
    # Close the video file
    cap.release()
    
    # Update final statistics
    stats["video_info"]["processed_frames"] = processed_count
    stats["video_info"]["processing_completed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate detection rates
    if processed_count > 0:
        stats["detection_rates"]["dominant_hand"]["detection_rate"] = (
            stats["detection_rates"]["dominant_hand"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["non_dominant_hand"]["detection_rate"] = (
            stats["detection_rates"]["non_dominant_hand"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["face"]["detection_rate"] = (
            stats["detection_rates"]["face"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["overall"]["success_rate"] = (
            stats["detection_rates"]["overall"]["all_detected"] / processed_count * 100
        )
    
    # Calculate processing performance
    if processed_count > 0:
        stats["processing_performance"]["average_processing_time_ms"] = (
            total_processing_time / processed_count * 1000
        )
    stats["processing_performance"]["total_processing_time_seconds"] = total_processing_time
    
    # Save statistics to JSON file
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Add summary statistics to log file
    with open(log_file, "a") as log:
        log.write(f"\n\n===== PROCESSING SUMMARY =====\n")
        log.write(f"Completed at: {stats['video_info']['processing_completed']}\n")
        log.write(f"Frames processed: {processed_count} from {start_frame} to {min(end_frame, frame_idx-1)}\n\n")
        
        log.write("DETECTION RATES:\n")
        log.write(f"  Dominant hand ({dominand_hand}): {stats['detection_rates']['dominant_hand']['detection_rate']:.1f}%\n")
        log.write(f"  Non-dominant hand: {stats['detection_rates']['non_dominant_hand']['detection_rate']:.1f}%\n")
        log.write(f"  Face: {stats['detection_rates']['face']['detection_rate']:.1f}%\n")
        log.write(f"  All parts detected: {stats['detection_rates']['overall']['success_rate']:.1f}%\n\n")
        
        log.write("DETECTION FAILURES:\n")
        log.write(f"  Frames with dominant hand failures: {len(stats['failed_frames']['dominant_hand_failures'])}\n")
        log.write(f"  Frames with non-dominant hand failures: {len(stats['failed_frames']['non_dominant_hand_failures'])}\n")
        log.write(f"  Frames with face failures: {len(stats['failed_frames']['face_failures'])}\n")
        log.write(f"  Frames with all parts missing: {len(stats['failed_frames']['all_failures'])}\n\n")
        
        log.write("PERFORMANCE:\n")
        log.write(f"  Average processing time per frame: {stats['processing_performance']['average_processing_time_ms']:.2f} ms\n")
        log.write(f"  Total processing time: {stats['processing_performance']['total_processing_time_seconds']:.2f} seconds\n")
    
    print(f"\n===== PROCESSING SUMMARY =====")
    print(f"Processed {processed_count} frames")
    print(f"Detection rates: Dom hand: {stats['detection_rates']['dominant_hand']['detection_rate']:.1f}%, " +
          f"Non-dom hand: {stats['detection_rates']['non_dominant_hand']['detection_rate']:.1f}%, " +
          f"Face: {stats['detection_rates']['face']['detection_rate']:.1f}%")
    print(f"All parts detected in {stats['detection_rates']['overall']['success_rate']:.1f}% of frames")
    print(f"Full statistics saved to: {stats_file}")
    print(f"Results saved to: {output_dir}")
    
    return str(output_dir)



def load_frame_data(npz_path):
    """
    Load saved frame data from an NPZ file.
    
    Args:
        npz_path (str): Path to the saved .npz file
        
    Returns:
        tuple: All the detection results for the frame
    """
    data = np.load(npz_path)
    
    # Extract all arrays from the npz file
    dom_landmarks = data['dom_landmarks']
    non_dom_landmarks = data['non_dom_landmarks']
    confidence_scores = data['confidence_scores']
    interpolation_scores = data['interpolation_scores']
    detection_status = data['detection_status']
    blendshape_scores = data['blendshape_scores']
    face_detected = data['face_detected'].item()  # Convert 0-d array to scalar
    nose_to_wrist_dist = data['nose_to_wrist_dist']
    frame_idx = data['frame_idx'].item()
    timestamp_ms = data['timestamp_ms'].item()
    
    return (dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores,
            detection_status, blendshape_scores, face_detected, 
            nose_to_wrist_dist, frame_idx, timestamp_ms)



def find_interpolation_frames(x, nums_list):
    """
    Returns integers in the range [x-5, x+5] that are not equal to x
    and are not in nums_list.
    
    Args:
        x (int): The reference integer
        nums_list (list): A list of integers
        
    Returns:
        list: Integers in [x-5, x+5] excluding x and elements in nums_list
    """
    # Create the set of all integers in the range [x-5, x+5]
    all_range = set(range(x-5, x+6))  # +6 because range is exclusive at upper bound
    
    # Remove x itself
    all_range.discard(x)
    
    # Remove numbers that are in the input list
    result = all_range - set(nums_list)
    
    # Convert back to a list and return
    return sorted(list(result))





def find_file_with_partial_name(partial_name, search_dir='.', recursive=False):
    """
    Find files that start with the given partial name.
    
    Args:
        partial_name (str): Partial file name to match
        search_dir (str): Directory to search in (default: current directory)
        recursive (bool): Whether to search in subdirectories
        
    Returns:
        list: Complete paths of all matching files
    """
    # Create a search pattern for files starting with the partial name
    search_pattern = os.path.join(search_dir, f"{partial_name}*")
    
    # Use recursive glob if requested
    if recursive:
        matches = []
        for root, _, _ in os.walk(search_dir):
            matches.extend(glob.glob(os.path.join(root, f"{os.path.basename(partial_name)}*")))
        return matches
    else:
        return glob.glob(search_pattern)
    


def has_numbers_on_both_sides(x, missing_numbers):
    """
    Checks if the list of missing numbers has at least one number smaller than x
    AND at least one number larger than x.
    
    Args:
        x (int): The reference integer
        missing_numbers (list): Output from find_missing_numbers(x, nums_list)
        
    Returns:
        bool: False if all numbers are either all smaller or all larger than x.
              True if there's at least one smaller and one larger number.
    """
    has_smaller = False
    has_larger = False
    
    for num in missing_numbers:
        if num < x:
            has_smaller = True
        elif num > x:
            has_larger = True
            
        # Early exit if we found both smaller and larger numbers
        if has_smaller and has_larger:
            return True
    
    # If we get here, we didn't find both smaller and larger numbers
    return False


def modify_npz_file(file_path, modifications):
    """
    Load a .npz file, modify existing arrays and add new ones, then save it back.
    
    Args:
        file_path (str): Path to the .npz file
        modifications (dict): Dictionary with keys as array names and values as new arrays
                             or functions that take the original array and return a modified version
    """
    # Load the npz file
    with np.load(file_path) as data:
        # Create a copy of all arrays
        arrays = {name: data[name] for name in data.files}
    
    # Apply modifications and add new arrays
    for name, modification in modifications.items():
        if name in arrays:
            if callable(modification):
                # If the modification is a function, apply it to the original array
                arrays[name] = modification(arrays[name])
            else:
                # Otherwise, replace the array
                arrays[name] = modification
        else:
            # Add new array
            arrays[name] = modification
            print(f"Adding new array '{name}' to the file")
    
    # Save back to the file with same format
    np.savez(file_path, **arrays)
    
    print(f"Successfully modified/added {len(modifications)} arrays in {file_path}")



def interpolate_undetected_hand_landmarks(directory_path):  
    """
    Interpolate landmarks for frames where hand detection failed.
    """
    print(f"Starting interpolation for directory: {directory_path}")
    
    # Load detection statistics JSON
    with open(os.path.join(directory_path, 'detection_statistics.json')) as f:
        data = json.load(f)
    
    first_frame_number = round(data['video_info']['fps'] * data['video_info']['start_time'])
    final_frame_number = round(data['video_info']['fps'] * data['video_info']['end_time'])
    
    print(f"Processing frames range: {first_frame_number} to {final_frame_number}")
    
    # Maximum possible sum of weights for normalization (when all 10 frames are available)
    MAX_WEIGHT_SUM = 2.92722222
    
    # Process non-dominant hand failures
    print("Processing non-dominant hand failures...")
    missing_non_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['non_dominant_hand_failures']]
    
    non_dom_interpolated_count = 0
    
    for missing_frame in data['failed_frames']['non_dominant_hand_failures']:
        frame_number = missing_frame['frame']
        filepath = missing_frame['file']
        
        # Only interpolate frames not at the edges of the video
        if (frame_number - 5) <= first_frame_number or (frame_number + 5) >= final_frame_number:
            print(f"Skipping frame {frame_number} - too close to video boundary")
            continue
        
        # Find frames with valid detections for interpolation
        interpolation_frames = find_interpolation_frames(frame_number, missing_non_dominant_frame_list)
        
        if not interpolation_frames:
            print(f"No valid frames found for interpolating frame {frame_number}")
            continue
        
        # Calculate interpolated landmarks
        interpolation_weights_sum = 0
        interpolated_coordinates = np.zeros(shape=(20, 3))
        interpolated_wrist_to_nose = np.zeros(2)
        
        for interp_frame in interpolation_frames:
            weight = 1 / ((frame_number - interp_frame) ** 2)
            interpolation_weights_sum += weight
            
            # Find and load the reference frame
            interp_partial_filename = data['video_info']['name'] + f"_frame{interp_frame:06d}"
            try:
                interp_files = find_file_with_partial_name(
                    interp_partial_filename, 
                    search_dir=directory_path, 
                    recursive=False
                )
                
                if not interp_files:
                    print(f"Warning: Could not find file for frame {interp_frame}")
                    continue
                    
                interp_filepath = interp_files[0]
                
                # Load the frame data - index 1 for non-dominant hand landmarks
                frame_data = load_frame_data(interp_filepath)
                non_dom_landmarks = frame_data[1]  # Correct index for non-dominant hand
                nose_to_wrist_non_dom = frame_data[7][1, :]
                
                
                # Add weighted contribution
                interpolated_coordinates += weight * non_dom_landmarks
                interpolated_wrist_to_nose += weight * nose_to_wrist_non_dom
                
            except Exception as e:
                print(f"Error processing frame {interp_frame}: {e}")
                continue
        
        # Normalize by sum of weights (crucial step!)
        if interpolation_weights_sum > 0:
            interpolated_coordinates /= interpolation_weights_sum
            interpolated_wrist_to_nose /= interpolation_weights_sum
            
            # Calculate confidence based on weights and frame distribution
            has_frames_on_both_sides = has_numbers_on_both_sides(frame_number, interpolation_frames)
            
            if has_frames_on_both_sides:
                interpolation_confidence = interpolation_weights_sum / MAX_WEIGHT_SUM
            else:
                interpolation_confidence = (interpolation_weights_sum / MAX_WEIGHT_SUM) * 0.8
                
            print(f"Frame {frame_number}: Interpolated with confidence {interpolation_confidence:.2f}")
            
            # Update the file with interpolated data
            def update_interp_scores(arr):
                new_arr = arr.copy()
                new_arr[1] = interpolation_confidence  # Index 1 for non-dominant hand
                return new_arr
            
            def update_nose_to_wrist_scores(matrix):
                new_matrix = matrix.copy()
                new_matrix[1, :] = interpolated_wrist_to_nose
                return new_matrix
                
            modifications = {
                'non_dom_landmarks': interpolated_coordinates,
                'interpolation_scores': update_interp_scores,
                'nose_to_wrist_dist': update_nose_to_wrist_scores
            }
            
            modify_npz_file(
                file_path=os.path.join(directory_path, filepath),
                modifications=modifications
            )
            
            non_dom_interpolated_count += 1
    
    # Process dominant hand failures
    print(f"Interpolated {non_dom_interpolated_count} non-dominant hand frames")
    print("Processing dominant hand failures...")
    
    missing_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['dominant_hand_failures']]
    
    dom_interpolated_count = 0
    
    for missing_frame in data['failed_frames']['dominant_hand_failures']:
        frame_number = missing_frame['frame']
        filepath = missing_frame['file']
        
        # Only interpolate frames not at the edges of the video
        if (frame_number - 5) <= first_frame_number or (frame_number + 5) >= final_frame_number:
            continue
        
        # Find frames with valid detections for interpolation
        interpolation_frames = find_interpolation_frames(frame_number, missing_dominant_frame_list)
        
        if not interpolation_frames:
            continue
        
        # Calculate interpolated landmarks
        interpolation_weights_sum = 0
        interpolated_coordinates = np.zeros(shape=(20, 3))
        interpolated_wrist_to_nose = np.zeros(2)
        
        for interp_frame in interpolation_frames:
            weight = 1 / ((frame_number - interp_frame) ** 2)
            interpolation_weights_sum += weight
            
            # Find and load the reference frame
            interp_partial_filename = data['video_info']['name'] + f"_frame{interp_frame:06d}"
            try:
                interp_files = find_file_with_partial_name(
                    interp_partial_filename, 
                    search_dir=directory_path, 
                    recursive=False
                )
                
                if not interp_files:
                    continue
                    
                interp_filepath = interp_files[0]
                
                # Load the frame data - index 0 for dominant hand landmarks
                frame_data = load_frame_data(interp_filepath)
                dom_landmarks = frame_data[0]  # Correct index for dominant hand
                nose_to_wrist_dom = frame_data[7][0, :]
                
                # Add weighted contribution
                interpolated_coordinates += weight * dom_landmarks
                interpolated_wrist_to_nose += weight * nose_to_wrist_non_dom
                
            except Exception as e:
                print(f"Error processing frame {interp_frame}: {e}")
                continue
        
        # Normalize by sum of weights
        if interpolation_weights_sum > 0:
            interpolated_coordinates /= interpolation_weights_sum
            interpolated_wrist_to_nose /= interpolation_weights_sum
            # Calculate confidence based on weights and frame distribution
            has_frames_on_both_sides = has_numbers_on_both_sides(frame_number, interpolation_frames)
            
            if has_frames_on_both_sides:
                interpolation_confidence = interpolation_weights_sum / MAX_WEIGHT_SUM
            else:
                interpolation_confidence = (interpolation_weights_sum / MAX_WEIGHT_SUM) * 0.8
            
            # Update the file with interpolated data
            def update_interp_scores(arr):
                new_arr = arr.copy()
                new_arr[0] = interpolation_confidence  # Index 0 for dominant hand
                return new_arr
            
            def update_nose_to_wrist_scores(matrix):
                new_matrix = matrix.copy()
                new_matrix[0, :] = interpolated_wrist_to_nose
                return new_matrix
                
            modifications = {
                'dom_landmarks': interpolated_coordinates,
                'interpolation_scores': update_interp_scores,
                'nose_to_wrist_dist': update_nose_to_wrist_scores
            }
            

            modify_npz_file(
                file_path=os.path.join(directory_path, filepath),
                modifications=modifications
            )
            
            dom_interpolated_count += 1
    
    print(f"Interpolated {dom_interpolated_count} dominant hand frames")
    print(f"Total interpolated: {non_dom_interpolated_count + dom_interpolated_count} frames")
    
    return non_dom_interpolated_count + dom_interpolated_count





def is_valid_detection(frame_data, is_dominant_hand):
    """
    Check if the frame has valid detection (not interpolated) for a specific hand.
    
    Args:
        frame_data: The loaded frame data
        is_dominant_hand: If True, check dominant hand; if False, check non-dominant hand;
                         if None, check if either hand is detected
    
    Returns:
        bool: Whether the specified hand(s) is/are detected
    """
    detection_status = frame_data[4]
    
    if is_dominant_hand:
        # Check specifically for dominant hand
        return detection_status[0] == 1
    else:
        # Check specifically for non-dominant hand
        return detection_status[1] == 1
    


def has_value(frame_data, is_dominant_hand):
    """Check if the frame exists and has any value (detection or interpolation)"""
    detection_status = frame_data[4]
    interpolation_scores = frame_data[3]
    if is_dominant_hand:
        return (detection_status[0]==1) or (interpolation_scores[0]>0)
    else:
        return (detection_status[1]==1) or (interpolation_scores[1]>0)
    
        

def cartesian_to_spherical(velocities):
    """
    Convert Cartesian velocities (ux, uy, uz) to spherical coordinate features.
    
    Args:
        velocities: NumPy array of shape (20, 3) with Cartesian velocities
        
    Returns:
        NumPy array of shape (20, 5) with spherical features:
            [vmagnitude, ϕsin, ϕcos, θsin, θcos]
    """
    num_landmarks = velocities.shape[0]
    spherical_features = np.zeros((num_landmarks, 5))
    
    for i in range(num_landmarks):
        ux, uy, uz = velocities[i]
        
        # Calculate velocity magnitude
        vmagnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        spherical_features[i, 0] = vmagnitude
        
        # Handle edge cases to avoid division by zero
        if vmagnitude == 0:
            # If velocity is zero, set all angles to zero
            spherical_features[i, 1:] = 0
            continue
        
        # Calculate azimuth angle (ϕ)
        phi = np.arctan2(uy, ux)
        spherical_features[i, 1] = np.sin(phi)  # ϕsin
        spherical_features[i, 2] = np.cos(phi)  # ϕcos
        
        # Calculate elevation angle (θ)
        # Clamp uz/vmagnitude to range [-1, 1] to avoid numerical errors
        cos_theta = np.clip(uz / vmagnitude, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        spherical_features[i, 3] = np.sin(theta)  # θsin
        spherical_features[i, 4] = cos_theta      # θcos (already calculated)
    
    return spherical_features



def cartesian_to_polar_features(velocities):
    """
    Convert Cartesian velocity coordinates to polar features.
    
    Parameters:
    -----------
    velocities : numpy.ndarray
        Array of shape (2, 2) where each row represents an object's [Ux, Uy]

        
    Returns:
    --------
    numpy.ndarray
        Array of shape (2, 3) with columns [magnitude, sin(direction), cos(direction)]
    """
    # Calculate magnitude
    magnitude = np.sqrt(np.sum(velocities**2, axis=1))
    
    # Initialize result array
    result = np.zeros((velocities.shape[0], 3))
    result[:, 0] = magnitude  # Set first column to magnitude
    
    # Create a mask for non-zero magnitudes
    non_zero = magnitude > 0
    
    # For non-zero magnitudes, calculate direction components
    if np.any(non_zero):
        # Get direction for non-zero magnitudes
        direction = np.arctan2(velocities[non_zero, 1], velocities[non_zero, 0])
        
        # Calculate sin and cos
        result[non_zero, 1] = np.sin(direction)  # sin(direction)
        result[non_zero, 2] = np.cos(direction)  # cos(direction)
    
    # Handle zero magnitudes 
    zero_indices = ~non_zero
    if np.any(zero_indices):
        result[zero_indices, 1] = 0.0
        result[zero_indices, 2] = 0.0
    
    return result

def compute_landmark_velocities(directory_path):
    """
    Compute velocity features for hand landmarks using central differencing with two window sizes,
    and convert to spherical coordinates.
    
    Args:
        directory_path (str): Path to the directory containing frame NPZ files
    
    Returns:
        int: Number of frames processed
    """
    # List all NPZ files in the directory
    npz_files = sorted(glob.glob(os.path.join(directory_path, "*.npz")))
    
    # Skip if no files found
    if not npz_files:
        print(f"No NPZ files found in {directory_path}")
        return 0
    
    print(f"Computing velocities for {len(npz_files)} files...")
    
    # Create a mapping of frame indices to file paths
    frame_to_file = {}
    for file_path in npz_files:
        frame_data = load_frame_data(file_path)
        frame_idx = frame_data[8]  # Index for frame_idx
        frame_to_file[frame_idx] = file_path
    
    frame_indices = sorted(frame_to_file.keys())
    processed_count = 0
    

    min_frame = min(frame_indices)
    max_frame = max(frame_indices)
    safe_margin = 5  # Skip processing frames within 5 frames of the edge
    
    # Process each frame
    for i, curr_idx in enumerate(frame_indices):
        if curr_idx < min_frame + safe_margin or curr_idx > max_frame - safe_margin:
            dom_velocity_small = np.zeros((20, 5))
            dom_velocity_large = np.zeros((20, 5))
            non_dom_velocity_small = np.zeros((20, 5))
            non_dom_velocity_large = np.zeros((20, 5))
            
            wrist_velocity_small = np.zeros((2, 3))  
            wrist_velocity_large = np.zeros((2, 3))
            
            # Create zero confidence arrays
            velocity_confidence = np.zeros(2)
            velocity_calculation_confidence = np.zeros(2)
            
            # Save these zero arrays
            modifications = {
                'dom_velocity_small': dom_velocity_small,
                'dom_velocity_large': dom_velocity_large,
                'non_dom_velocity_small': non_dom_velocity_small,
                'non_dom_velocity_large': non_dom_velocity_large,
                'velocity_confidence': velocity_confidence,
                'velocity_calculation_confidence': velocity_calculation_confidence,
                'wrist_velocity_small': wrist_velocity_small,
                'wrist_velocity_large': wrist_velocity_large,
            }
            
            # Get the file path for this frame
            current_file_path = frame_to_file[curr_idx]
            modify_npz_file(current_file_path, modifications)
            processed_count += 1
            
            # Log that we're skipping calculation
            print(f"Frame {curr_idx} too close to video boundary - setting zero velocities")
            continue
        # Load current frame
        current_file_path = frame_to_file[curr_idx]
        curr_frame_data = load_frame_data(current_file_path)
        
        # Store needed frames in a dictionary for easy access
        frame_cache = {curr_idx: curr_frame_data}
        
        # Load all potentially needed frames in the -5 to +5 range
        for offset in range(-5, 6):
            if offset == 0:  # Skip current frame (already loaded)
                continue
            
            check_idx = curr_idx + offset
            if check_idx in frame_to_file:
                frame_cache[check_idx] = load_frame_data(frame_to_file[check_idx])
            else:
                frame_cache[check_idx] = None  # Mark as not available
        
        # Extract dominant and non-dominant hand landmarks from current frame
        dom_landmarks = curr_frame_data[0]
        non_dom_landmarks = curr_frame_data[1]
        
        # Initialize velocity arrays in Cartesian coordinates
        dom_velocity_small_cart = np.zeros_like(dom_landmarks)
        dom_velocity_large_cart = np.zeros_like(dom_landmarks)
        non_dom_velocity_small_cart = np.zeros_like(non_dom_landmarks)
        non_dom_velocity_large_cart = np.zeros_like(non_dom_landmarks)
        
        wrist_velocity_small = np.zeros((2, 2))  # 2 hands × [x, y] coordinates
        wrist_velocity_large = np.zeros((2, 2))  # 2 hands × [x, y] coordinates
        
        # Initialize confidence and method weight tracking
        dom_small_conf = 0.0
        dom_large_conf = 0.0
        non_dom_small_conf = 0.0
        non_dom_large_conf = 0.0
        
        dom_small_method_weight = 0.0
        dom_large_method_weight = 0.0
        non_dom_small_method_weight = 0.0
        non_dom_large_method_weight = 0.0
        
        dom_small_source_quality = 0.0
        dom_large_source_quality = 0.0
        non_dom_small_source_quality = 0.0
        non_dom_large_source_quality = 0.0
        
        # ===== DOMINANT HAND VELOCITY CALCULATION =====
        
        # Small window [-1, +1] velocity with fallbacks
        if (curr_idx + 1 in frame_cache and frame_cache[curr_idx + 1] is not None and 
            is_valid_detection(frame_cache[curr_idx + 1], True) and 
            curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
            has_value(frame_cache[curr_idx - 1], True)):
            # Ideal case: (t+1, t-1)
            dom_velocity_small_cart = (frame_cache[curr_idx + 1][0] - frame_cache[curr_idx - 1][0]) / 2.0
            wrist_velocity_small[0, :] = (frame_cache[curr_idx + 1][7][0, :] - frame_cache[curr_idx - 1][7][0, :]) / 2.0
            dom_small_conf = min(frame_cache[curr_idx + 1][2][0], frame_cache[curr_idx - 1][2][0])  # Detection confidence of t+1 frame
            dom_small_method_weight = 1.0  # Ideal frames
            # Calculate source quality factor (average of interpolation confidences)
            t_plus_1_interp = frame_cache[curr_idx + 1][3][0]
            t_minus_1_interp = frame_cache[curr_idx - 1][3][0]
            dom_small_source_quality = (t_plus_1_interp + t_minus_1_interp) / 2.0
            
        elif (curr_idx + 2 in frame_cache and frame_cache[curr_idx + 2] is not None and 
              is_valid_detection(frame_cache[curr_idx + 2], True) and 
              curr_idx - 2 in frame_cache and frame_cache[curr_idx - 2] is not None and 
              has_value(frame_cache[curr_idx - 2], True)):
            # Fallback 1: (t+2, t-2)
            dom_velocity_small_cart = (frame_cache[curr_idx + 2][0] - frame_cache[curr_idx - 2][0]) / 4.0
            wrist_velocity_small[0, :] = (frame_cache[curr_idx + 2][7][0, :] - frame_cache[curr_idx - 2][7][0, :]) / 4.0
            dom_small_conf = min(frame_cache[curr_idx + 2][2][0], frame_cache[curr_idx - 2][2][0])  # Detection confidence of t+2 frame
            dom_small_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_2_interp = frame_cache[curr_idx + 2][3][0]
            t_minus_2_interp = frame_cache[curr_idx - 2][3][0]
            dom_small_source_quality = (t_plus_2_interp + t_minus_2_interp) / 2.0
            
        elif (curr_idx + 2 in frame_cache and frame_cache[curr_idx + 2] is not None and 
              is_valid_detection(frame_cache[curr_idx + 2], True)):
            if (curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
                has_value(frame_cache[curr_idx - 1], True)):
                # Fallback 2: (t+2, t-1)
                dom_velocity_small_cart = (frame_cache[curr_idx + 2][0] - frame_cache[curr_idx - 1][0]) / 3.0
                wrist_velocity_small[0, :] = (frame_cache[curr_idx + 2][7][0, :] - frame_cache[curr_idx - 1][7][0, :]) / 3.0
                dom_small_conf = min(frame_cache[curr_idx + 2][2][0], frame_cache[curr_idx - 1][2][0])  # Detection confidence of t+2 frame
                dom_small_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_2_interp = frame_cache[curr_idx + 2][3][0]
                t_minus_1_interp = frame_cache[curr_idx - 1][3][0]
                dom_small_source_quality = (t_plus_2_interp + t_minus_1_interp) / 2.0
                
            elif is_valid_detection(curr_frame_data, True):
                # Fallback 3: (t+2, t)
                dom_velocity_small_cart = (frame_cache[curr_idx + 2][0] - curr_frame_data[0]) / 2.0
                wrist_velocity_small[0, :] = (frame_cache[curr_idx + 2][7][0, :] - curr_frame_data[7][0, :]) / 2.0
                dom_small_conf = min(frame_cache[curr_idx + 2][2][0], curr_frame_data[2][0])
                dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_plus_2_interp = frame_cache[curr_idx + 2][3][0]
                t_interp = curr_frame_data[3][0]
                dom_small_source_quality = (t_plus_2_interp + t_interp) / 2.0
                
        elif is_valid_detection(curr_frame_data, True):
            if (curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
                has_value(frame_cache[curr_idx - 1], True)):
                # Fallback 4: (t, t-1)
                dom_velocity_small_cart = (curr_frame_data[0] - frame_cache[curr_idx - 1][0])
                wrist_velocity_small[0, :] = (curr_frame_data[7][0, :] - frame_cache[curr_idx - 1][7][0, :]) 
                dom_small_conf = min(curr_frame_data[2][0], frame_cache[curr_idx - 1][2][0])  # Detection confidence of current frame
                dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_interp = curr_frame_data[3][0]
                t_minus_1_interp = frame_cache[curr_idx - 1][3][0]
                dom_small_source_quality = (t_interp + t_minus_1_interp) / 2.0
                
            elif (curr_idx - 2 in frame_cache and frame_cache[curr_idx - 2] is not None and 
                  has_value(frame_cache[curr_idx - 2], True)):
                # Fallback 5: (t, t-2)
                dom_velocity_small_cart = (curr_frame_data[0] - frame_cache[curr_idx - 2][0]) / 2.0
                wrist_velocity_small[0, :] = (curr_frame_data[7][0, :] - frame_cache[curr_idx - 2][7][0, :]) / 2.0
                dom_small_conf = min(curr_frame_data[2][0], frame_cache[curr_idx - 2][2][0])  # Detection confidence of current frame
                dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_interp = curr_frame_data[3][0]
                t_minus_2_interp = frame_cache[curr_idx - 2][3][0]
                dom_small_source_quality = (t_interp + t_minus_2_interp) / 2.0
        
        # Large window [-5, +5] velocity with fallbacks

        if (curr_idx + 5 in frame_cache and frame_cache[curr_idx + 5] is not None and 
            is_valid_detection(frame_cache[curr_idx + 5], True) and 
            curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
            has_value(frame_cache[curr_idx - 5], True)):
            # Ideal case: (t+5, t-5)
            dom_velocity_large_cart = (frame_cache[curr_idx + 5][0] - frame_cache[curr_idx - 5][0]) / 10.0
            wrist_velocity_large[0, :] = (frame_cache[curr_idx + 5][7][0, :] - frame_cache[curr_idx - 5][7][0, :]) / 10.0
            dom_large_conf = min(frame_cache[curr_idx + 5][2][0], frame_cache[curr_idx - 5][2][0])  # Detection confidence of t+5 frame
            dom_large_method_weight = 1.0  # Ideal frames
            # Calculate source quality factor
            t_plus_5_interp = frame_cache[curr_idx + 5][3][0]
            t_minus_5_interp = frame_cache[curr_idx - 5][3][0]
            dom_large_source_quality = (t_plus_5_interp + t_minus_5_interp) / 2.0
            
        elif (curr_idx + 4 in frame_cache and frame_cache[curr_idx + 4] is not None and 
        is_valid_detection(frame_cache[curr_idx + 4], True) and 
        curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
        has_value(frame_cache[curr_idx - 4], True)):
        # Fallback 1: (t+4, t-4)
            dom_velocity_large_cart = (frame_cache[curr_idx + 4][0] - frame_cache[curr_idx - 4][0]) / 8.0
            wrist_velocity_large[0, :] = (frame_cache[curr_idx + 4][7][0, :] - frame_cache[curr_idx - 4][7][0, :]) / 8.0
            dom_large_conf = min(frame_cache[curr_idx + 4][2][0], frame_cache[curr_idx - 4][2][0]) # Detection confidence of t+4 frame
            dom_large_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_4_interp = frame_cache[curr_idx + 4][3][0]
            t_minus_4_interp = frame_cache[curr_idx - 4][3][0]
            dom_large_source_quality = (t_plus_4_interp + t_minus_4_interp) / 2.0
    
        elif (curr_idx + 3 in frame_cache and frame_cache[curr_idx + 3] is not None and 
            is_valid_detection(frame_cache[curr_idx + 3], True) and 
            curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
            has_value(frame_cache[curr_idx - 3], True)):
            # Fallback 2: (t+3, t-3)
            dom_velocity_large_cart = (frame_cache[curr_idx + 3][0] - frame_cache[curr_idx - 3][0]) / 6.0
            wrist_velocity_large[0, :] = (frame_cache[curr_idx + 3][7][0, :] - frame_cache[curr_idx - 3][7][0, :]) / 6.0
            dom_large_conf = min(frame_cache[curr_idx + 3][2][0], frame_cache[curr_idx - 3][2][0])  # Detection confidence of t+3 frame
            dom_large_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_3_interp = frame_cache[curr_idx + 3][3][0]
            t_minus_3_interp = frame_cache[curr_idx - 3][3][0]
            dom_large_source_quality = (t_plus_3_interp + t_minus_3_interp) / 2.0
            
        # Asymmetric fallbacks for large window
        elif (curr_idx + 5 in frame_cache and frame_cache[curr_idx + 5] is not None and 
            is_valid_detection(frame_cache[curr_idx + 5], True)):
            if (curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
                has_value(frame_cache[curr_idx - 4], True)):
                # Fallback 3: (t+5, t-4)
                dom_velocity_large_cart = (frame_cache[curr_idx + 5][0] - frame_cache[curr_idx - 4][0]) / 9.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 5][7][0, :] - frame_cache[curr_idx - 4][7][0, :]) / 9.0
                dom_large_conf = min(frame_cache[curr_idx + 5][2][0], frame_cache[curr_idx - 4][2][0])  # Detection confidence of t+5 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_5_interp = frame_cache[curr_idx + 5][3][0]
                t_minus_4_interp = frame_cache[curr_idx - 4][3][0]
                dom_large_source_quality = (t_plus_5_interp + t_minus_4_interp) / 2.0
                
            elif (curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
                has_value(frame_cache[curr_idx - 3], True)):
                # Fallback 4: (t+5, t-3)
                dom_velocity_large_cart = (frame_cache[curr_idx + 5][0] - frame_cache[curr_idx - 3][0]) / 8.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 5][7][0, :] - frame_cache[curr_idx - 3][7][0, :]) / 8.0
                dom_large_conf = min(frame_cache[curr_idx + 5][2][0], frame_cache[curr_idx - 3][2][0])  # Detection confidence of t+5 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_5_interp = frame_cache[curr_idx + 5][3][0]
                t_minus_3_interp = frame_cache[curr_idx - 3][3][0]
                dom_large_source_quality = (t_plus_5_interp + t_minus_3_interp) / 2.0
        
        elif (curr_idx + 4 in frame_cache and frame_cache[curr_idx + 4] is not None and 
            is_valid_detection(frame_cache[curr_idx + 4], True)):
            if (curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
                has_value(frame_cache[curr_idx - 5], True)):
                # Fallback 5: (t+4, t-5)
                dom_velocity_large_cart = (frame_cache[curr_idx + 4][0] - frame_cache[curr_idx - 5][0]) / 9.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 4][7][0, :] - frame_cache[curr_idx - 5][7][0, :]) / 9.0
                dom_large_conf = min(frame_cache[curr_idx + 4][2][0], frame_cache[curr_idx - 5][2][0])  # Detection confidence of t+4 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_4_interp = frame_cache[curr_idx + 4][3][0]
                t_minus_5_interp = frame_cache[curr_idx - 5][3][0]
                dom_large_source_quality = (t_plus_4_interp + t_minus_5_interp) / 2.0
                
            elif (curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
                has_value(frame_cache[curr_idx - 3], True)):
                # Fallback 6: (t+4, t-3)
                dom_velocity_large_cart = (frame_cache[curr_idx + 4][0] - frame_cache[curr_idx - 3][0]) / 7.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 4][7][0, :] - frame_cache[curr_idx - 3][7][0, :]) / 7.0
                dom_large_conf = min(frame_cache[curr_idx + 4][2][0], frame_cache[curr_idx - 3][2][0])  # Detection confidence of t+4 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_4_interp = frame_cache[curr_idx + 4][3][0]
                t_minus_3_interp = frame_cache[curr_idx - 3][3][0]
                dom_large_source_quality = (t_plus_4_interp + t_minus_3_interp) / 2.0
                
        elif (curr_idx + 3 in frame_cache and frame_cache[curr_idx + 3] is not None and 
            is_valid_detection(frame_cache[curr_idx + 3], True)):
            if (curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
                has_value(frame_cache[curr_idx - 5], True)):
                # Fallback 7: (t+3, t-5)
                dom_velocity_large_cart = (frame_cache[curr_idx + 3][0] - frame_cache[curr_idx - 5][0]) / 8.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 3][7][0, :] - frame_cache[curr_idx - 5][7][0, :]) / 8.0
                dom_large_conf = min(frame_cache[curr_idx + 3][2][0], frame_cache[curr_idx - 5][2][0])  # Detection confidence of t+3 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_3_interp = frame_cache[curr_idx + 3][3][0]
                t_minus_5_interp = frame_cache[curr_idx - 5][3][0]
                dom_large_source_quality = (t_plus_3_interp + t_minus_5_interp) / 2.0
                
            elif (curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
                has_value(frame_cache[curr_idx - 4], True)):
                # Fallback 8: (t+3, t-4)
                dom_velocity_large_cart = (frame_cache[curr_idx + 3][0] - frame_cache[curr_idx - 4][0]) / 7.0
                wrist_velocity_large[0, :] = (frame_cache[curr_idx + 3][7][0, :] - frame_cache[curr_idx - 4][7][0, :]) / 7.0
                dom_large_conf = min(frame_cache[curr_idx + 3][2][0], frame_cache[curr_idx - 4][2][0])  # Detection confidence of t+3 frame
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_3_interp = frame_cache[curr_idx + 3][3][0]
                t_minus_4_interp = frame_cache[curr_idx - 4][3][0]
                dom_large_source_quality = (t_plus_3_interp + t_minus_4_interp) / 2.0

        # ===== NON-DOMINANT HAND VELOCITY CALCULATION =====

    # Small window [-1, +1] velocity with fallbacks for non-dominant hand
        if (curr_idx + 1 in frame_cache and frame_cache[curr_idx + 1] is not None and 
            is_valid_detection(frame_cache[curr_idx + 1], False) and 
            curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
            has_value(frame_cache[curr_idx - 1], False)):
            # Ideal case: (t+1, t-1)
            non_dom_velocity_small_cart = (frame_cache[curr_idx + 1][1] - frame_cache[curr_idx - 1][1]) / 2.0
            wrist_velocity_small[1, :] = (frame_cache[curr_idx + 1][7][1, :] - frame_cache[curr_idx - 1][7][1, :]) / 2.0
            non_dom_small_conf = min(frame_cache[curr_idx + 1][2][1], frame_cache[curr_idx - 1][2][1])  # Detection confidence of t+1 frame
            non_dom_small_method_weight = 1.0  # Ideal frames
            # Calculate source quality factor
            t_plus_1_interp = frame_cache[curr_idx + 1][3][1]
            t_minus_1_interp = frame_cache[curr_idx - 1][3][1]
            non_dom_small_source_quality = (t_plus_1_interp + t_minus_1_interp) / 2.0
            
        elif (curr_idx + 2 in frame_cache and frame_cache[curr_idx + 2] is not None and 
            is_valid_detection(frame_cache[curr_idx + 2], False) and 
            curr_idx - 2 in frame_cache and frame_cache[curr_idx - 2] is not None and 
            has_value(frame_cache[curr_idx - 2], False)):
            # Fallback 1: (t+2, t-2)
            non_dom_velocity_small_cart = (frame_cache[curr_idx + 2][1] - frame_cache[curr_idx - 2][1]) / 4.0
            wrist_velocity_small[1, :] = (frame_cache[curr_idx + 2][7][1, :] - frame_cache[curr_idx - 2][7][1, :]) / 4.0
            non_dom_small_conf = min(frame_cache[curr_idx + 2][2][1], frame_cache[curr_idx - 2][2][1]) 
            non_dom_small_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_2_interp = frame_cache[curr_idx + 2][3][1]
            t_minus_2_interp = frame_cache[curr_idx - 2][3][1]
            non_dom_small_source_quality = (t_plus_2_interp + t_minus_2_interp) / 2.0
            
        elif (curr_idx + 2 in frame_cache and frame_cache[curr_idx + 2] is not None and 
            is_valid_detection(frame_cache[curr_idx + 2], False)):
            if (curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
                has_value(frame_cache[curr_idx - 1], False)):
                # Fallback 2: (t+2, t-1)
                non_dom_velocity_small_cart = (frame_cache[curr_idx + 2][1] - frame_cache[curr_idx - 1][1]) / 3.0
                wrist_velocity_small[1, :] = (frame_cache[curr_idx + 2][7][1, :] - frame_cache[curr_idx - 2][7][1, :]) / 3.0
                non_dom_small_conf = min(frame_cache[curr_idx + 2][2][1], frame_cache[curr_idx - 1][2][1])  
                non_dom_small_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_2_interp = frame_cache[curr_idx + 2][3][1]
                t_minus_1_interp = frame_cache[curr_idx - 1][3][1]
                non_dom_small_source_quality = (t_plus_2_interp + t_minus_1_interp) / 2.0
                
            elif is_valid_detection(curr_frame_data, False):
                # Fallback 3: (t+2, t)
                non_dom_velocity_small_cart = (frame_cache[curr_idx + 2][1] - curr_frame_data[1]) / 2.0
                wrist_velocity_small[1, :] = (frame_cache[curr_idx + 2][7][1, :] - curr_frame_data[7][1, :]) / 2.0
                non_dom_small_conf = min(frame_cache[curr_idx + 2][2][1], curr_frame_data[2][1])
                non_dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_plus_2_interp = frame_cache[curr_idx + 2][3][1]
                t_interp = curr_frame_data[3][1]
                non_dom_small_source_quality = (t_plus_2_interp + t_interp) / 2.0
                
        elif is_valid_detection(curr_frame_data, False):
            if (curr_idx - 1 in frame_cache and frame_cache[curr_idx - 1] is not None and 
                has_value(frame_cache[curr_idx - 1], False)):
                # Fallback 4: (t, t-1)
                non_dom_velocity_small_cart = (curr_frame_data[1] - frame_cache[curr_idx - 1][1])
                wrist_velocity_small[1, :] = (curr_frame_data[7][1, :] - frame_cache[curr_idx - 1][7][1, :]) 
                non_dom_small_conf = min(curr_frame_data[2][1], frame_cache[curr_idx - 1][2][1])  # Detection confidence of current frame
                non_dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_interp = curr_frame_data[3][1]
                t_minus_1_interp = frame_cache[curr_idx - 1][3][1]
                non_dom_small_source_quality = (t_interp + t_minus_1_interp) / 2.0
                
            elif (curr_idx - 2 in frame_cache and frame_cache[curr_idx - 2] is not None and 
                has_value(frame_cache[curr_idx - 2], False)):
                # Fallback 5: (t, t-2)
                non_dom_velocity_small_cart = (curr_frame_data[1] - frame_cache[curr_idx - 2][1]) / 2.0
                wrist_velocity_small[1, :] = (curr_frame_data[7][1, :] - frame_cache[curr_idx -21][7][1, :]) / 2.0
                non_dom_small_conf = min(curr_frame_data[2][1], frame_cache[curr_idx -2][2][1])  # Detection confidence of current frame
                non_dom_small_method_weight = 0.4  # One-sided derivative
                # Calculate source quality factor
                t_interp = curr_frame_data[3][1]
                t_minus_2_interp = frame_cache[curr_idx - 2][3][1]
                non_dom_small_source_quality = (t_interp + t_minus_2_interp) / 2.0
    
        # Large window [-5, +5] velocity with fallbacks for non-dominant hand
        if (curr_idx + 5 in frame_cache and frame_cache[curr_idx + 5] is not None and 
            is_valid_detection(frame_cache[curr_idx + 5], False) and 
            curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
            has_value(frame_cache[curr_idx - 5], False)):
            # Ideal case: (t+5, t-5)
            non_dom_velocity_large_cart = (frame_cache[curr_idx + 5][1] - frame_cache[curr_idx - 5][1]) / 10.0
            wrist_velocity_large[1, :] = (frame_cache[curr_idx + 5][7][1, :] - frame_cache[curr_idx - 5][7][1, :]) / 10.0
            non_dom_large_conf = min(frame_cache[curr_idx + 5][2][1], frame_cache[curr_idx - 5][2][1])  
            non_dom_large_method_weight = 1.0  # Ideal frames
            # Calculate source quality factor
            t_plus_5_interp = frame_cache[curr_idx + 5][3][1]
            t_minus_5_interp = frame_cache[curr_idx - 5][3][1]
            non_dom_large_source_quality = (t_plus_5_interp + t_minus_5_interp) / 2.0
            
        elif (curr_idx + 4 in frame_cache and frame_cache[curr_idx + 4] is not None and 
            is_valid_detection(frame_cache[curr_idx + 4], False) and 
            curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
            has_value(frame_cache[curr_idx - 4], False)):
            # Fallback 1: (t+4, t-4)
            non_dom_velocity_large_cart = (frame_cache[curr_idx + 4][1] - frame_cache[curr_idx - 4][1]) / 8.0
            wrist_velocity_large[1, :] = (frame_cache[curr_idx + 4][7][1, :] - frame_cache[curr_idx - 4][7][1, :]) / 8.0
            non_dom_large_conf = min(frame_cache[curr_idx + 4][2][1], frame_cache[curr_idx - 4][2][1])  
            non_dom_large_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_4_interp = frame_cache[curr_idx + 4][3][1]
            t_minus_4_interp = frame_cache[curr_idx - 4][3][1]
            non_dom_large_source_quality = (t_plus_4_interp + t_minus_4_interp) / 2.0
            
        elif (curr_idx + 3 in frame_cache and frame_cache[curr_idx + 3] is not None and 
            is_valid_detection(frame_cache[curr_idx + 3], False) and 
            curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
            has_value(frame_cache[curr_idx - 3], False)):
            # Fallback 2: (t+3, t-3)
            non_dom_velocity_large_cart = (frame_cache[curr_idx + 3][1] - frame_cache[curr_idx - 3][1]) / 6.0
            wrist_velocity_large[1, :] = (frame_cache[curr_idx + 3][7][1, :] - frame_cache[curr_idx - 3][7][1, :]) / 6.0
            non_dom_large_conf = min(frame_cache[curr_idx + 3][2][1], frame_cache[curr_idx - 3][2][1])  # Detection confidence of t+3 frame
            non_dom_large_method_weight = 0.8  # Wider symmetric window
            # Calculate source quality factor
            t_plus_3_interp = frame_cache[curr_idx + 3][3][1]
            t_minus_3_interp = frame_cache[curr_idx - 3][3][1]
            non_dom_large_source_quality = (t_plus_3_interp + t_minus_3_interp) / 2.0
            
        # Asymmetric fallbacks for large window
        elif (curr_idx + 5 in frame_cache and frame_cache[curr_idx + 5] is not None and 
            is_valid_detection(frame_cache[curr_idx + 5], False)):
            if (curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
                has_value(frame_cache[curr_idx - 4], False)):
                # Fallback 3: (t+5, t-4)
                non_dom_velocity_large_cart = (frame_cache[curr_idx + 5][1] - frame_cache[curr_idx - 4][1]) / 9.0
                wrist_velocity_large[1, :] = (frame_cache[curr_idx + 5][7][1, :] - frame_cache[curr_idx - 4][7][1, :]) / 9.0
                non_dom_large_conf = min(frame_cache[curr_idx + 5][2][1], frame_cache[curr_idx - 4][2][1])  # Detection confidence of t+5 frame
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_5_interp = frame_cache[curr_idx + 5][3][1]
                t_minus_4_interp = frame_cache[curr_idx - 4][3][1]
                non_dom_large_source_quality = (t_plus_5_interp + t_minus_4_interp) / 2.0
                
            elif (curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
                has_value(frame_cache[curr_idx - 3], False)):
                # Fallback 4: (t+5, t-3)
                non_dom_velocity_large_cart = (frame_cache[curr_idx + 5][1] - frame_cache[curr_idx - 3][1]) / 8.0
                wrist_velocity_large[1, :] = (frame_cache[curr_idx + 5][7][1, :] - frame_cache[curr_idx - 3][7][1, :]) / 8.0
                non_dom_large_conf = min(frame_cache[curr_idx + 5][2][1], frame_cache[curr_idx - 3][2][1])  # Detection confidence of t+5 frame
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_5_interp = frame_cache[curr_idx + 5][3][1]
                t_minus_3_interp = frame_cache[curr_idx - 3][3][1]
                non_dom_large_source_quality = (t_plus_5_interp + t_minus_3_interp) / 2.0
                
        elif (curr_idx + 4 in frame_cache and frame_cache[curr_idx + 4] is not None and 
            is_valid_detection(frame_cache[curr_idx + 4], False)):
            if (curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
                has_value(frame_cache[curr_idx - 5], False)):
                # Fallback 5: (t+4, t-5)
                non_dom_velocity_large_cart = (frame_cache[curr_idx + 4][1] - frame_cache[curr_idx - 5][1]) / 9.0
                wrist_velocity_large[1, :] = (frame_cache[curr_idx + 4][7][1, :] - frame_cache[curr_idx - 5][7][1, :]) / 9.0
                non_dom_large_conf = min(frame_cache[curr_idx + 4][2][1], frame_cache[curr_idx - 5][2][1])  # Detection confidence of t+4 frame
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_4_interp = frame_cache[curr_idx + 4][3][1]
                t_minus_5_interp = frame_cache[curr_idx - 5][3][1]
                non_dom_large_source_quality = (t_plus_4_interp + t_minus_5_interp) / 2.0
                
            elif (curr_idx - 3 in frame_cache and frame_cache[curr_idx - 3] is not None and 
                has_value(frame_cache[curr_idx - 3], False)):
                # Fallback 6: (t+4, t-3)
                non_dom_velocity_large_cart = (frame_cache[curr_idx + 4][1] - frame_cache[curr_idx - 3][1]) / 7.0
                wrist_velocity_large[1, :] = (frame_cache[curr_idx + 4][7][1, :] - frame_cache[curr_idx - 3][7][1, :]) / 7.0
                non_dom_large_conf = min(frame_cache[curr_idx + 4][2][1], frame_cache[curr_idx - 3][2][1])  # Detection confidence of t+4 frame
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_4_interp = frame_cache[curr_idx + 4][3][1]
                t_minus_3_interp = frame_cache[curr_idx - 3][3][1]
                non_dom_large_source_quality = (t_plus_4_interp + t_minus_3_interp) / 2.0
                
        elif (curr_idx + 3 in frame_cache and frame_cache[curr_idx + 3] is not None and 
            is_valid_detection(frame_cache[curr_idx + 3], False)):
            if (curr_idx - 5 in frame_cache and frame_cache[curr_idx - 5] is not None and 
                has_value(frame_cache[curr_idx - 5], False)):
                # Fallback 7: (t+3, t-5)
                non_dom_velocity_large_cart = (frame_cache[curr_idx + 3][1] - frame_cache[curr_idx - 5][1]) / 8.0
                wrist_velocity_large[1, :] = (frame_cache[curr_idx + 3][7][1, :] - frame_cache[curr_idx - 5][7][1, :]) / 8.0
                non_dom_large_conf = min(frame_cache[curr_idx + 3][2][1], frame_cache[curr_idx - 5][2][1])  # Detection confidence of t+3 frame
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                # Calculate source quality factor
                t_plus_3_interp = frame_cache[curr_idx + 3][3][1]
                t_minus_5_interp = frame_cache[curr_idx - 5][3][1]
                non_dom_large_source_quality = (t_plus_3_interp + t_minus_5_interp) / 2.0
            
        elif (curr_idx - 4 in frame_cache and frame_cache[curr_idx - 4] is not None and 
            has_value(frame_cache[curr_idx - 4], False)):
            # Fallback 8: (t+3, t-4)
            non_dom_velocity_large_cart = (frame_cache[curr_idx + 3][1] - frame_cache[curr_idx - 4][1]) / 7.0
            wrist_velocity_large[1, :] = (frame_cache[curr_idx + 3][7][1, :] - frame_cache[curr_idx - 4][7][1, :]) / 7.0
            non_dom_large_conf = min(frame_cache[curr_idx + 3][2][1], frame_cache[curr_idx - 4][2][1])  # Detection confidence of t+3 frame
            non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
            # Calculate source quality factor
            t_plus_3_interp = frame_cache[curr_idx + 3][3][1]
            t_minus_4_interp = frame_cache[curr_idx - 4][3][1]
            non_dom_large_source_quality = (t_plus_3_interp + t_minus_4_interp) / 2.0
            

        
        # Convert Cartesian velocities to spherical/polar coordinates
        dom_velocity_small = cartesian_to_spherical(dom_velocity_small_cart)
        dom_velocity_large = cartesian_to_spherical(dom_velocity_large_cart)
        non_dom_velocity_small = cartesian_to_spherical(non_dom_velocity_small_cart)
        non_dom_velocity_large = cartesian_to_spherical(non_dom_velocity_large_cart)
        
        wrist_velocity_small_polar = cartesian_to_polar_features(wrist_velocity_small)
        wrist_velocity_large_polar = cartesian_to_polar_features(wrist_velocity_large)
        
        # Calculate average confidence for each hand across both windows
        dom_avg_conf = (dom_small_conf + dom_large_conf) / 2.0
        non_dom_avg_conf = (non_dom_small_conf + non_dom_large_conf) / 2.0
        
        # Calculate velocityCalculationConfidence using method weight and source quality
        dom_small_vel_calc_conf = dom_small_method_weight * dom_small_source_quality
        dom_large_vel_calc_conf = dom_large_method_weight * dom_large_source_quality
        non_dom_small_vel_calc_conf = non_dom_small_method_weight * non_dom_small_source_quality
        non_dom_large_vel_calc_conf = non_dom_large_method_weight * non_dom_large_source_quality
        
        # Average across windows for each hand
        dom_vel_calc_conf = (dom_small_vel_calc_conf + dom_large_vel_calc_conf) / 2.0
        non_dom_vel_calc_conf = (non_dom_small_vel_calc_conf + non_dom_large_vel_calc_conf) / 2.0
        
        # Prepare arrays
        velocity_confidence = np.array([dom_avg_conf, non_dom_avg_conf])
        velocity_calculation_confidence = np.array([dom_vel_calc_conf, non_dom_vel_calc_conf])
        
        # Save back to the NPZ file
        modifications = {
            'dom_velocity_small': dom_velocity_small,
            'dom_velocity_large': dom_velocity_large,
            'non_dom_velocity_small': non_dom_velocity_small,
            'non_dom_velocity_large': non_dom_velocity_large,
            'velocity_confidence': velocity_confidence,
            'velocity_calculation_confidence': velocity_calculation_confidence,
            'wrist_velocity_small': wrist_velocity_small_polar,
            'wrist_velocity_large': wrist_velocity_large_polar,
            
        }
        
        modify_npz_file(current_file_path, modifications)
        processed_count += 1
        
        # Progress update
        if (i + 1) % 100 == 0 or i == len(frame_indices) - 1:
            print(f"Processed {i+1}/{len(frame_indices)} frames")
    
    print(f"Velocity computation complete. Processed {processed_count} frames.")
    return processed_count


def load_frame_data_with_velocities(npz_path):
    """
    Load saved frame data from an NPZ file.
    
    Args:
        npz_path (str): Path to the saved .npz file
        
    Returns:
        tuple: All the detection results for the frame
    """
    data = np.load(npz_path)
    
    # Extract all arrays from the npz file
    dom_landmarks = data['dom_landmarks']
    non_dom_landmarks = data['non_dom_landmarks']
    confidence_scores = data['confidence_scores']
    interpolation_scores = data['interpolation_scores']
    detection_status = data['detection_status']
    blendshape_scores = data['blendshape_scores']
    face_detected = data['face_detected'].item()  # Convert 0-d array to scalar
    nose_to_wrist_dist = data['nose_to_wrist_dist']
    frame_idx = data['frame_idx'].item()
    timestamp_ms = data['timestamp_ms'].item()
    dom_velocity_small = data['dom_velocity_small']
    dom_velocity_large = data['dom_velocity_large']
    non_dom_velocity_small = data['non_dom_velocity_small']
    non_dom_velocity_large = data['non_dom_velocity_large']
    velocity_confidence = data['velocity_confidence']
    velocity_calculation_confidence = data['velocity_calculation_confidence']
    nose_to_wrist_velocity_small = data['wrist_velocity_small']
    nose_to_wrist_velocity_large = data['wrist_velocity_large']
    
    return (dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores,
            detection_status, blendshape_scores, face_detected, 
            nose_to_wrist_dist, frame_idx, timestamp_ms, dom_velocity_small, dom_velocity_large, non_dom_velocity_small, non_dom_velocity_large, velocity_confidence, velocity_calculation_confidence, nose_to_wrist_velocity_small, nose_to_wrist_velocity_large)



def make_videos_df(directory_path):
    """
    Process video files in a directory and return information in a pandas DataFrame.
    
    Parameters:
    directory_path (str): Path to the directory containing video files
    
    Returns:
    pandas.DataFrame: DataFrame with video name, frame count, fps, Right/Left designation, and file path
    """
    # Lists to store video information
    data = []
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    # Walk through the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a video
            _, ext = os.path.splitext(file)
            if ext.lower() in video_extensions:
                file_path = os.path.join(root, file)
                
                # Extract video name (without extension)
                video_name = os.path.splitext(file)[0]
                
                # Determine if it's Right or Left
                if video_name.endswith("_R"):
                    dom_hand = "Right"
                elif video_name.endswith("_L"):
                    dom_hand = "Left"
                else:
                    dom_hand = None
                
                # Open the video file
                try:
                    cap = cv2.VideoCapture(file_path)
                    
                    # Get frames per second
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Get total number of frames
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Release the video capture object
                    cap.release()
                    
                    # Add row to data
                    data.append({
                        'Video Name': video_name,
                        'Frame Count': frame_count,
                        'FPS': fps,
                        'dom_hand': dom_hand,
                        'file_path': file_path  # Added file_path to the DataFrame
                    })
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    return df



def process_video_new_2(video_path, adaptive_detect_func=adaptive_detect, hand_model_path=hand_model_path, face_model_path=face_model_path,
                 min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                 min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                 num_hands=2, output_face_blendshapes=True,
                 max_attempts=3, threshold_reduction_factor=0.7, min_threshold=0.2, 
                 frame_step=1, start_time_seconds=0, end_time_seconds=None,
                 save_failure_screenshots=False,
                 num_workers=None,  # Added parameter for parallel processing
                 batch_mode=False):  # Added parameter to indicate batch processing
    """
    Process a video frame-by-frame using the adaptive_detect function and save results.
    
    Args:
        video_path (str): Path to the video file
        adaptive_detect_func: The adaptive detection function to use
        min_hand_detection_confidence (float): Initial confidence threshold for hand detection
        min_hand_presence_confidence (float): Initial confidence threshold for hand presence
        min_face_detection_confidence (float): Initial confidence threshold for face detection
        min_face_presence_confidence (float): Initial confidence threshold for face presence
        num_hands (int): Maximum number of hands to detect
        dominand_hand (str): Dominant hand preference ('Left' or 'Right')
        output_face_blendshapes (bool): Whether to detect face blendshapes
        max_attempts (int): Maximum detection attempts for adaptive detection
        threshold_reduction_factor (float): Factor to reduce thresholds by
        min_threshold (float): Minimum threshold limit
        frame_step (int): Process every Nth frame (1 = all frames)
        start_time_seconds (float): Time in seconds to start processing from
        end_time_seconds (float): Time in seconds to end processing (None = process until end)
        save_failure_screenshots (bool): Save screenshots for all frames with any detection failures
        num_workers (int): Number of parallel workers to use (None = auto-detect based on CPU cores)
        batch_mode (bool): Whether this is being run as part of a batch process
        
    Returns:
        str: Path to the directory containing saved frame results
    """
    # Import additional libraries for parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    from queue import Queue
    import multiprocessing
    import gc  # For garbage collection

    # Extract video name for directory creation
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_name = video_path.stem  # Get filename without extension
    
    # Extract dominant hand information from filename
    if video_name.endswith("_R"):
        extracted_dominant_hand = "Right"
    elif video_name.endswith("_L"):
        extracted_dominant_hand = "Left"
    else:
        # Default if not specified in filename
        extracted_dominant_hand = "Right"
        print(f"Warning: Could not determine dominant hand from filename, using default: {extracted_dominant_hand}")

    # Use the extracted dominant hand instead of the parameter
    dominand_hand = extracted_dominant_hand
    if not batch_mode:
        print(f"Detected dominant hand from filename: {dominand_hand}")
    else:
        print(f"[{os.path.basename(str(video_path))}] Dominant hand: {dominand_hand}")

    # Create output directory
    output_dir = video_dir / f"{video_name}_landmarks"
    output_dir.mkdir(exist_ok=True)
    
    # Create screenshots directory if screenshot option is enabled
    screenshots_dir = None
    if save_failure_screenshots:
        screenshots_dir = output_dir / "failure_screenshots"
        screenshots_dir.mkdir(exist_ok=True)
    
    # Create a log file to track processing
    log_file = output_dir / "processing_log.txt"
    
    # Create a detailed statistics file
    stats_file = output_dir / "detection_statistics.json"

    # Initialize statistics tracking
    stats = {
        "video_info": {
            "name": video_name,
            "path": str(video_path),
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0,
            "duration_seconds": 0,
            "start_time": start_time_seconds,
            "end_time": end_time_seconds,
            "dominant_hand": dominand_hand,
            "processing_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_completed": None
        },
        "detection_rates": {
            "dominant_hand": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "non_dominant_hand": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "face": {
                "detected": 0,
                "failed": 0,
                "detection_rate": 0
            },
            "overall": {
                "all_detected": 0,
                "partial_detections": 0,
                "no_detections": 0,
                "success_rate": 0
            }
        },
        "failed_frames": {
            "dominant_hand_failures": [],
            "non_dominant_hand_failures": [],
            "face_failures": [],
            "all_failures": []
        },
        "processing_performance": {
            "average_processing_time_ms": 0,
            "total_processing_time_seconds": 0
        }
    }
    
    # Setup logging
    with open(log_file, "w") as log:
        log.write(f"Processing video: {video_path}\n")
        log.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Parameters:\n")
        log.write(f"  - frame_step: {frame_step}\n")
        log.write(f"  - start_time: {start_time_seconds} seconds\n")
        if end_time_seconds is not None:
            log.write(f"  - end_time: {end_time_seconds} seconds\n")
        log.write(f"  - dominand_hand: {dominand_hand}\n")
        log.write(f"  - num_hands: {num_hands}\n")
        log.write(f"  - detection confidence thresholds: {min_hand_detection_confidence}, {min_face_detection_confidence}\n")
        log.write(f"  - batch_mode: {batch_mode}\n")
        log.write("\n--- Frame processing log ---\n")
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    # Update stats with video info
    stats["video_info"]["total_frames"] = total_frames
    stats["video_info"]["fps"] = fps
    stats["video_info"]["duration_seconds"] = duration_seconds
    if end_time_seconds==None:
        stats["video_info"]["end_time"] = duration_seconds
    
    # Convert time to frame indices
    start_frame = int(max(0, start_time_seconds * fps))
    
    # Set end frame if specified
    if end_time_seconds is not None:
        end_frame = min(total_frames, int(end_time_seconds * fps))
    else:
        end_frame = total_frames
    
    # Set the number of worker threads if not specified
    if num_workers is None:
        # More conservative thread allocation to prevent system overload
        # Use 50% of available cores (minimum 2, maximum 6) to leave resources for other processes
        num_workers = max(2, min(multiprocessing.cpu_count() // 2, 6))
    
    # When in batch mode, conserve resources even more and limit output
    if batch_mode:
        # Further reduce threads in batch mode to ensure stability
        num_workers = 8
        print(f"[{os.path.basename(str(video_path))}] Using {num_workers} worker threads")
    else:
        print(f"Video: {video_name}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Duration: {duration_seconds:.2f} seconds")
        print(f"Processing frames {start_frame} to {end_frame} (time {start_time_seconds:.2f}s to {end_time_seconds if end_time_seconds is not None else duration_seconds:.2f}s)")
        print(f"Output directory: {output_dir}")
        print(f"Using {num_workers} worker threads for parallel processing")
    
    # Function to process a single frame
    def process_single_frame(args):
        frame, frame_idx, temp_dir = args
        
        # Get timestamp in milliseconds
        timestamp_ms = int(frame_idx * 1000 / fps)
        timestamp_formatted = f"{timestamp_ms//60000:02d}m{(timestamp_ms//1000)%60:02d}s{timestamp_ms%1000:03d}ms"
        
        # Temporary frame path
        temp_frame_path = Path(temp_dir) / f"temp_frame_{frame_idx}.jpg"
        
        # Save the current frame as an image
        cv2.imwrite(str(temp_frame_path), frame)
        
        start_time = time.time()
        try:
            with SuppressOutput():
                # Use adaptive_detect on the frame
                results = adaptive_detect_func(
                    str(temp_frame_path), hand_model_path, face_model_path,
                    min_hand_detection_confidence=min_hand_detection_confidence,
                    min_hand_presence_confidence=min_hand_presence_confidence,
                    min_face_detection_confidence=min_face_detection_confidence,
                    min_face_presence_confidence=min_face_presence_confidence,
                    num_hands=num_hands,
                    dominand_hand=dominand_hand,
                    visualize=False,
                    output_face_blendshapes=output_face_blendshapes,
                    max_attempts=max_attempts,
                    threshold_reduction_factor=threshold_reduction_factor,
                    min_threshold=min_threshold
                )
            
            # Calculate processing time
            proc_time = time.time() - start_time
            
            # Create output filename with frame info
            output_filename = f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}.npz"
            output_path = output_dir / output_filename
            
            # Unpack results
            dom_landmarks, non_dom_landmarks, confidence_scores, interpolation_scores, detection_status, blendshape_scores, face_detected, nose_to_wrist_dist = results
            
            # Save screenshot if any detection failed and screenshots are enabled
            if save_failure_screenshots and (detection_status[0] != 1 or detection_status[1] != 1 or face_detected != 1):
                # Create a detailed failure type description for the filename
                failure_type = []
                if detection_status[0] != 1:
                    failure_type.append("DomHand")
                if detection_status[1] != 1:
                    failure_type.append("NonDomHand")
                if face_detected != 1:
                    failure_type.append("Face")
                
                failure_str = "_".join(failure_type)
                screenshot_filename = f"{video_name}_frame{frame_idx:06d}_{timestamp_formatted}_missing_{failure_str}.jpg"
                screenshot_path = screenshots_dir / screenshot_filename
                
                # Copy the frame to the screenshots directory
                cv2.imwrite(str(screenshot_path), frame)
            
            # Save all results in a single .npz file
            np.savez(
                output_path,
                dom_landmarks=dom_landmarks,
                non_dom_landmarks=non_dom_landmarks,
                confidence_scores=confidence_scores,
                interpolation_scores=interpolation_scores,
                detection_status=detection_status,
                blendshape_scores=blendshape_scores,
                face_detected=face_detected,
                nose_to_wrist_dist=nose_to_wrist_dist,
                frame_idx=np.array([frame_idx]),
                timestamp_ms=np.array([timestamp_ms])
            )
            
            return {
                "success": True,
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "timestamp_formatted": timestamp_formatted,
                "detection_status": detection_status,
                "face_detected": face_detected,
                "proc_time": proc_time
            }
        
        except Exception as e:
            return {
                "success": False,
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "error": str(e)
            }
        finally:
            # Clean up temporary frame file
            if temp_frame_path.exists():
                try:
                    temp_frame_path.unlink()
                except:
                    pass  # Ignore errors during cleanup
    
    # Create a thread-safe lock for updating the progress bar to avoid output issues
    progress_lock = threading.Lock()
    
    # Collect frames to process
    frames_to_process = []
    frame_idx = start_frame
    
    # Skip to start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    processed_count = 0
    total_processing_time = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        if not batch_mode:
            print("Reading frames to process...")
        
        # First phase: Read all frames to process
        while frame_idx < end_frame:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Only process every frame_step frames
            if (frame_idx - start_frame) % frame_step == 0:
                frames_to_process.append((frame.copy(), frame_idx, temp_dir))
            
            frame_idx += 1
        
        # Release the video capture to free resources
        cap.release()
        
        if not batch_mode:
            print(f"Starting parallel processing of {len(frames_to_process)} frames with {num_workers} workers...")
        
        try:
            # Process frames in parallel using ThreadPoolExecutor
            # Use a context manager to ensure all resources are properly released
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all frame processing tasks
                futures = [executor.submit(process_single_frame, args) for args in frames_to_process]
                
                # Process results as they complete
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    
                    if result["success"]:
                        frame_idx = result["frame_idx"]
                        detection_status = result["detection_status"]
                        face_detected = result["face_detected"]
                        proc_time = result["proc_time"]
                        
                        # Update detection statistics
                        dom_hand_detected = detection_status[0] == 1
                        non_dom_hand_detected = detection_status[1] == 1
                        face_was_detected = face_detected == 1
                        
                        if dom_hand_detected:
                            stats["detection_rates"]["dominant_hand"]["detected"] += 1
                        else:
                            stats["detection_rates"]["dominant_hand"]["failed"] += 1
                            stats["failed_frames"]["dominant_hand_failures"].append({
                                "frame": frame_idx,
                                "timestamp_ms": result["timestamp_ms"],
                                "file": f"{video_name}_frame{frame_idx:06d}_{result['timestamp_formatted']}.npz"
                            })
                        
                        if non_dom_hand_detected:
                            stats["detection_rates"]["non_dominant_hand"]["detected"] += 1
                        else:
                            stats["detection_rates"]["non_dominant_hand"]["failed"] += 1
                            stats["failed_frames"]["non_dominant_hand_failures"].append({
                                "frame": frame_idx,
                                "timestamp_ms": result["timestamp_ms"],
                                "file": f"{video_name}_frame{frame_idx:06d}_{result['timestamp_formatted']}.npz"
                            })
                        
                        if face_was_detected:
                            stats["detection_rates"]["face"]["detected"] += 1
                        else:
                            stats["detection_rates"]["face"]["failed"] += 1
                            stats["failed_frames"]["face_failures"].append({
                                "frame": frame_idx,
                                "timestamp_ms": result["timestamp_ms"],
                                "file": f"{video_name}_frame{frame_idx:06d}_{result['timestamp_formatted']}.npz"
                            })
                        
                        # Track combined detection status
                        detection_count = dom_hand_detected + non_dom_hand_detected + face_was_detected
                        
                        if detection_count == 3:
                            stats["detection_rates"]["overall"]["all_detected"] += 1
                        elif detection_count == 0:
                            stats["detection_rates"]["overall"]["no_detections"] += 1
                            stats["failed_frames"]["all_failures"].append({
                                "frame": frame_idx,
                                "timestamp_ms": result["timestamp_ms"],
                                "file": f"{video_name}_frame{frame_idx:06d}_{result['timestamp_formatted']}.npz"
                            })
                        else:
                            stats["detection_rates"]["overall"]["partial_detections"] += 1
                        
                        # Update processing log
                        detection_summary = f"Dom: {detection_status[0]}, Non-dom: {detection_status[1]}, Face: {face_detected}"
                        log_entry = f"Frame {frame_idx}: {detection_summary} (proc time: {proc_time:.2f}s)\n"
                        
                        with open(log_file, "a") as log:
                            log.write(log_entry)
                        
                        total_processing_time += proc_time
                        processed_count += 1
                        
                        # Update progress with thread safety, but only if not in batch mode
                        # This prevents console output conflicts when running in batch
                        if not batch_mode:
                            with progress_lock:
                                update_progress(frame_idx, total_frames, result["timestamp_formatted"])
                                
                    else:
                        # Handle error
                        with open(log_file, "a") as log:
                            log.write(f"Error on frame {result['frame_idx']}: {result['error']}\n")
                        
                        # Update progress (even for errors), but only if not in batch mode
                        if not batch_mode:
                            with progress_lock:
                                update_progress(result['frame_idx'], total_frames, "ERROR")
        except Exception as e:
            print(f"Error during parallel processing: {str(e)}")
            # Continue with cleanup to ensure resources are released
        finally:
            # Clear the frames_to_process list to free memory
            frames_to_process.clear()
    
    # Update final statistics
    stats["video_info"]["processed_frames"] = processed_count
    stats["video_info"]["processing_completed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate detection rates
    if processed_count > 0:
        stats["detection_rates"]["dominant_hand"]["detection_rate"] = (
            stats["detection_rates"]["dominant_hand"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["non_dominant_hand"]["detection_rate"] = (
            stats["detection_rates"]["non_dominant_hand"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["face"]["detection_rate"] = (
            stats["detection_rates"]["face"]["detected"] / processed_count * 100
        )
        stats["detection_rates"]["overall"]["success_rate"] = (
            stats["detection_rates"]["overall"]["all_detected"] / processed_count * 100
        )
    
    # Calculate processing performance
    if processed_count > 0:
        stats["processing_performance"]["average_processing_time_ms"] = (
            total_processing_time / processed_count * 1000
        )
    stats["processing_performance"]["total_processing_time_seconds"] = total_processing_time
    
    # Save statistics to JSON file
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Add summary statistics to log file
    with open(log_file, "a") as log:
        log.write(f"\n\n===== PROCESSING SUMMARY =====\n")
        log.write(f"Completed at: {stats['video_info']['processing_completed']}\n")
        log.write(f"Frames processed: {processed_count} from {start_frame} to {min(end_frame, frame_idx-1)}\n\n")
        
        log.write("DETECTION RATES:\n")
        log.write(f"  Dominant hand ({dominand_hand}): {stats['detection_rates']['dominant_hand']['detection_rate']:.1f}%\n")
        log.write(f"  Non-dominant hand: {stats['detection_rates']['non_dominant_hand']['detection_rate']:.1f}%\n")
        log.write(f"  Face: {stats['detection_rates']['face']['detection_rate']:.1f}%\n")
        log.write(f"  All parts detected: {stats['detection_rates']['overall']['success_rate']:.1f}%\n\n")
        
        log.write("DETECTION FAILURES:\n")
        log.write(f"  Frames with dominant hand failures: {len(stats['failed_frames']['dominant_hand_failures'])}\n")
        log.write(f"  Frames with non-dominant hand failures: {len(stats['failed_frames']['non_dominant_hand_failures'])}\n")
        log.write(f"  Frames with face failures: {len(stats['failed_frames']['face_failures'])}\n")
        log.write(f"  Frames with all parts missing: {len(stats['failed_frames']['all_failures'])}\n\n")
        
        log.write("PERFORMANCE:\n")
        log.write(f"  Average processing time per frame: {stats['processing_performance']['average_processing_time_ms']:.2f} ms\n")
        log.write(f"  Total processing time: {stats['processing_performance']['total_processing_time_seconds']:.2f} seconds\n")
    
    # Adjust summary output based on batch mode
    if batch_mode:
        print(f"[{os.path.basename(str(video_path))}] Processed {processed_count} frames: " +
              f"Dom: {stats['detection_rates']['dominant_hand']['detection_rate']:.1f}%, " +
              f"Non-dom: {stats['detection_rates']['non_dominant_hand']['detection_rate']:.1f}%, " +
              f"Face: {stats['detection_rates']['face']['detection_rate']:.1f}%")
    else:
        print(f"\n===== PROCESSING SUMMARY =====")
        print(f"Processed {processed_count} frames")
        print(f"Detection rates: Dom hand: {stats['detection_rates']['dominant_hand']['detection_rate']:.1f}%, " +
              f"Non-dom hand: {stats['detection_rates']['non_dominant_hand']['detection_rate']:.1f}%, " +
              f"Face: {stats['detection_rates']['face']['detection_rate']:.1f}%")
        print(f"All parts detected in {stats['detection_rates']['overall']['success_rate']:.1f}% of frames")
        print(f"Full statistics saved to: {stats_file}")
        print(f"Results saved to: {output_dir}")
        
    # Force garbage collection to free memory
    # This is especially important in batch processing
    gc.collect()
    
    return str(output_dir)







def batch_call_process_video(video_path, start_time_seconds=0, end_time_seconds=None):
    """
    Wrapper function to call process_video with optimized settings for batch processing
    
    This wrapper ensures that process_video runs efficiently when processing
    multiple videos one after another.
    """
    # Set batch_mode to True and use conservative worker count
    return process_video_new_2(
        video_path=video_path,
        adaptive_detect_func=adaptive_detect,  # Use your actual adaptive_detect function
        hand_model_path=hand_model_path,  # Use your actual path
        face_model_path=face_model_path,  # Use your actual path
        start_time_seconds=start_time_seconds,
        end_time_seconds=end_time_seconds,
        # Enable batch mode for reduced console output and resource usage
        batch_mode=True,  
        
        num_workers=8
    )


def batch_call_process_video_with_output(video_path, start_time_seconds=0, end_time_seconds=None):
    """Wrapper with explicit console output"""
    import sys
    
    print(f"\nStarting processing of video: {os.path.basename(video_path)}", flush=True)
    
    # Call the original function
    result = batch_call_process_video(video_path, start_time_seconds, end_time_seconds)
    
    print(f"Completed processing of video: {os.path.basename(video_path)}", flush=True)
    sys.stdout.flush()
    
    return result
















def batch_process_videos(video_df, process_video_func, detection_threshold_dom, detection_threshold_non_dom, 
                         detection_threshold_dom_small_length, detection_threshold_non_dom_small_length, report_dir, 
                         start_time_seconds=0, end_time_seconds=None, delete_videos=False, 
                         start_from_row=0):
    """
    Process multiple videos from a pandas dataframe with memory-efficient operation.
    
    This function processes videos one by one, storing detailed results on disk rather than in memory.
    This approach prevents memory growth regardless of how many videos are processed, making it
    suitable for processing thousands of videos in a single run.
    

    
    Args:
        video_df (pandas.DataFrame): DataFrame with 'file_path', 'FPS', and 'Frame Count' columns
        process_video_func: The process_video function to use for processing each video
        detection_threshold_dom (float): Minimum detection rate (%) for dominant hand
        detection_threshold_non_dom (float): Minimum detection rate (%) for non-dominant hand
        detection_threshold_dom_small_length (float): Threshold for short videos (dominant hand)
        detection_threshold_non_dom_small_length (float): Threshold for short videos (non-dominant hand)
        start_time_seconds (float): Start time for video processing
        end_time_seconds (float): End time for video processing
        delete_videos (bool): Whether to delete original videos after processing
        start_from_row (int): Index to start processing from (for resuming previous runs)
        report_dir (str): Directory to store reports and detailed results
        
    Returns:
        str: Path to the generated report JSON file
    """
    # Create report directory structure
    os.makedirs(report_dir, exist_ok=True)
    details_dir = os.path.join(report_dir, "video_details")
    os.makedirs(details_dir, exist_ok=True)
    
    # Define report file paths
    current_report_path = os.path.join(report_dir, "video_processing_report_current.json")
    temp_report_path = os.path.join(report_dir, "video_processing_report_temp.json")
    
    # Initialize statistics for logging
    if start_from_row > 0 and os.path.exists(current_report_path):
        # Load previous summary statistics if resuming
        print(f"Loading previous report from {current_report_path}")
        try:
            with open(current_report_path, "r") as f:
                stats = json.load(f)
            
            # Update resume information
            stats["processing_info"]["resume_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats["processing_info"]["resumed_from_row"] = start_from_row
            
            print(f"Successfully loaded previous report summary")
        except Exception as e:
            print(f"Error loading previous report: {e}")
            return

    else:
        stats = None
        
    # Create new statistics if not resuming or if loading failed
    if stats is None:
        stats = {
            "processing_info": {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "total_videos": len(video_df),
                "videos_processed": 0,
                "directories_deleted": 0,
                "videos_deleted": 0,
                "detection_threshold_dom": detection_threshold_dom,
                "detection_threshold_non_dom": detection_threshold_non_dom,
                "detection_threshold_dom_small_length": detection_threshold_dom_small_length,
                "detection_threshold_non_dom_small_length": detection_threshold_non_dom_small_length,
                "last_processed_row": -1,
            },
            "deleted_directories_summary": {
                "count": 0
            }
        }

    if start_from_row > 0:
        last_processed_row = stats["processing_info"].get("last_processed_row", -1)
        if last_processed_row >= 0 and start_from_row <= last_processed_row:
            print(f"Resuming from row {start_from_row} (last processed row was {last_processed_row})")
        elif last_processed_row >= 0 and start_from_row > last_processed_row + 1:
            print(f"Warning: Skipping rows {last_processed_row+1} to {start_from_row-1}") 
    
    # Calculate the number of videos to process
    total_videos = len(video_df)
    remaining_videos = total_videos - start_from_row
    
    print(f"Starting batch processing from row {start_from_row} ({remaining_videos} videos remaining)")
    print(f"Detection thresholds (normal): Dom={detection_threshold_dom}%, Non-Dom={detection_threshold_non_dom}%")
    print(f"Detection thresholds (short videos): Dom={detection_threshold_dom_small_length}%, Non-Dom={detection_threshold_non_dom_small_length}%")
    print(f"Using memory-efficient processing - detailed results stored in: {details_dir}")
    
    # Helper function to save the current report using atomic operations
    def save_current_report():
        stats["processing_info"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to temporary file first to avoid corruption
        with open(temp_report_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Atomic rename to ensure file integrity
        if os.path.exists(temp_report_path):
            os.rename(temp_report_path, current_report_path)
    
    # Save initial report
    save_current_report()
    
    # Process each video starting from the specified row
    for idx in range(start_from_row, total_videos):
        row = video_df.iloc[idx]
        video_path = row['file_path']
        video_fps = row['FPS']
        video_framecount = row['Frame Count']
        video_length = video_framecount / video_fps
        

        # Skip if file doesn't exist
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            
            # Create video detail file for skipped video
            video_detail = {
                "video_path": video_path,
                "status": "skipped",
                "reason": "file_not_found",
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "row_index": idx
            }
            
            # Save detail to separate file
            path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
            detail_filename = f"video_detail_{idx}_{path_hash}.json"
            with open(os.path.join(details_dir, detail_filename), 'w') as f:
                json.dump(video_detail, f, indent=2)
                
            # Mark as processed
            stats["processing_info"]["last_processed_row"] = idx
            
            # Save summary report
            save_current_report()
            continue
        
        print(f"\nProcessing video {idx-start_from_row+1}/{remaining_videos} (overall: {idx+1}/{total_videos}): {video_path}")
        
        try:
            # Determine output directory path (before actually processing)
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(video_dir, f"{video_name}_landmarks")
            
            # Process the video
            process_start_time = datetime.now()
            process_video_func(video_path, start_time_seconds=start_time_seconds, end_time_seconds=end_time_seconds)
            process_end_time = datetime.now()
            process_duration = (process_end_time - process_start_time).total_seconds()
            
            stats["processing_info"]["videos_processed"] += 1
            
            # Check detection statistics
            stats_file = os.path.join(output_dir, "detection_statistics.json")
            
            if not os.path.exists(stats_file):
                print(f"Warning: Statistics file not found for {video_path}")
                
                # Create video detail file for error
                video_detail = {
                    "video_path": video_path,
                    "status": "error",
                    "reason": "statistics_file_not_found",
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time_seconds": process_duration,
                    "row_index": idx
                }
                
                # Save detail to separate file
                path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
                detail_filename = f"video_detail_{idx}_{path_hash}.json"
                with open(os.path.join(details_dir, detail_filename), 'w') as f:
                    json.dump(video_detail, f, indent=2)
                    
                
                stats["processing_info"]["last_processed_row"] = idx
                
                # Save summary report
                save_current_report()
                continue
                
            # Read detection statistics
            with open(stats_file, "r") as f:
                detection_stats = json.load(f)
            
            # Check detection rates
            dom_hand_rate = detection_stats["detection_rates"]["dominant_hand"]["detection_rate"]
            non_dom_hand_rate = detection_stats["detection_rates"]["non_dominant_hand"]["detection_rate"]
            face_rate = detection_stats["detection_rates"]["face"]["detection_rate"]
            
            # Create video details
            video_detail = {
                "video_path": video_path,
                "output_dir": output_dir,
                "dominant_hand_rate": dom_hand_rate,
                "non_dominant_hand_rate": non_dom_hand_rate,
                "face_rate": face_rate,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": process_duration,
                "row_index": idx,
                "video_length_seconds": video_length
            }
            
            # Apply appropriate thresholds based on video length
            current_detection_threshold_dom = detection_threshold_dom
            current_detection_threshold_non_dom = detection_threshold_non_dom
            
            if video_length < 10: 
                current_detection_threshold_dom = detection_threshold_dom_small_length
                current_detection_threshold_non_dom = detection_threshold_non_dom_small_length
                video_detail["thresholds_used"] = "short_video"
            else:
                video_detail["thresholds_used"] = "normal"
                
            # Check if detection rates are below threshold
            if dom_hand_rate < current_detection_threshold_dom and non_dom_hand_rate < current_detection_threshold_non_dom:
                print(f"Low detection rate for {video_name}: Dom={dom_hand_rate:.1f}%, Non-Dom={non_dom_hand_rate:.1f}%")
                print(f"Deleting directory: {output_dir}")
                
                # Delete the directory
                shutil.rmtree(output_dir)
                
                # Update statistics
                stats["processing_info"]["directories_deleted"] += 1
                stats["deleted_directories_summary"]["count"] += 1
                
                video_detail["status"] = "deleted"
                video_detail["reason"] = "low_detection_rate"
            else:
                print(f"Detection rates acceptable: Dom={dom_hand_rate:.1f}%, Non-Dom={non_dom_hand_rate:.1f}%")
                video_detail["status"] = "kept"
            
            # Save detail to separate file
            path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
            detail_filename = f"video_detail_{idx}_{path_hash}.json"
            with open(os.path.join(details_dir, detail_filename), 'w') as f:
                json.dump(video_detail, f, indent=2)
            
            stats["processing_info"]["last_processed_row"] = idx
            
            # Delete original video if requested
            if delete_videos:
                os.remove(video_path)
                stats["processing_info"]["videos_deleted"] += 1
                print(f"Deleted original video: {video_path}")
            
            # Save report after each video
            save_current_report()
        
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            
            # Create video detail file for error
            video_detail = {
                "video_path": video_path,
                "status": "error",
                "reason": str(e),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "row_index": idx
            }
            
            # Save detail to separate file
            path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
            detail_filename = f"video_detail_{idx}_{path_hash}.json"
            with open(os.path.join(details_dir, detail_filename), 'w') as f:
                json.dump(video_detail, f, indent=2)
                
            
            stats["processing_info"]["last_processed_row"] = idx
            
            # Save report even when errors occur
            save_current_report()
    
    # Complete statistics
    stats["processing_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    # Save final current report
    save_current_report()
    
    # Also save a timestamped archive copy
    archive_report_path = os.path.join(report_dir, f"video_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy(current_report_path, archive_report_path)
    
    # Print summary
    print("\n===== PROCESSING SUMMARY =====")
    print(f"Total videos: {stats['processing_info']['total_videos']}")
    print(f"Videos processed (this run + previous): {stats['processing_info']['videos_processed']}")
    print(f"Directories deleted (low detection rate): {stats['processing_info']['directories_deleted']}")
    if delete_videos:
        print(f"Original videos deleted: {stats['processing_info']['videos_deleted']}")
    print(f"Current report saved to: {current_report_path}")
    print(f"Archive report saved to: {archive_report_path}")
    print(f"Detailed results stored in: {details_dir}")
    
    return archive_report_path






def resample_single_video(video_info):
    """Helper function to resample a single video"""
    video_path, desired_fps = video_info
    try:
        # Get original video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"path": video_path, "status": "error", "message": "Could not open video"}
        
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / current_fps
        output_total_frames = int(duration * desired_fps)
        
        # Generate output path
        video_dir = os.path.dirname(video_path)
        video_filename = os.path.basename(video_path)
        base_name, ext = os.path.splitext(video_filename)
        output_path = os.path.join(video_dir, f"{base_name}_fps{int(desired_fps)}{ext}")
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, desired_fps, (width, height))
        
        # Process frames
        for i in range(output_total_frames):
            original_frame_idx = round(i * current_fps / desired_fps)
            if original_frame_idx >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Get new video metadata
        new_cap = cv2.VideoCapture(output_path)
        new_frame_count = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_fps = new_cap.get(cv2.CAP_PROP_FPS)
        new_cap.release()
        
        return {
            "path": video_path,
            "new_path": output_path,
            "status": "success",
            "frame_count": new_frame_count,
            "fps": new_fps
        }
        
    except Exception as e:
        return {"path": video_path, "status": "error", "message": str(e)}

def resample_videos_in_dataframe(df, desired_fps, save_checkpoint=False, checkpoint_file='resampling_progress.csv', file_path_col='file_path', 
                                delete_originals=False, inplace=False, 
                                max_workers=None, batch_size=20):
    """
    Efficiently resamples videos in a dataframe with parallel processing and batching.
    
    Parameters:
        df (pandas.DataFrame): Dataframe containing video paths
        desired_fps (float): The desired FPS for the output videos
        file_path_col (str): Name of the column containing file paths
        delete_originals (bool): Whether to delete original videos after resampling
        inplace (bool): Whether to modify the original dataframe or return a copy
        max_workers (int): Max number of parallel processes (None = auto-detect)
        batch_size (int): Size of batches for processing
        
    Returns:
        pandas.DataFrame: Updated dataframe with new file paths and metadata
    """
    # Use original dataframe or create a copy
    result_df = df if inplace else df.copy()
    
    # Check if metadata columns exist
    has_frame_count_col = 'Frame Count' in result_df.columns
    has_fps_col = 'FPS' in result_df.columns
    
    # Get all valid video paths
    video_paths = []
    for idx, path in enumerate(result_df[file_path_col]):
        if isinstance(path, str) and os.path.exists(path):
            video_paths.append((path, desired_fps))
    
    total_videos = len(video_paths)
    print(f"Found {total_videos} valid videos to process")
    
    # Process in batches to manage memory
    results = []
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_videos-1)//batch_size + 1} ({len(batch)} videos)")
        
        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(resample_single_video, video_info) for video_info in batch]
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(batch), desc="Resampling"):
                result = future.result()
                results.append(result)
                
                # Update dataframe as results come in
                if result["status"] == "success":
                    # Find the corresponding row
                    row_idx = result_df[result_df[file_path_col] == result["path"]].index
                    if len(row_idx) > 0:
                        idx = row_idx[0]
                        # Update file path
                        result_df.at[idx, file_path_col] = result["new_path"]
                        
                        # Update metadata if columns exist
                        if has_frame_count_col:
                            result_df.at[idx, 'Frame Count'] = result["frame_count"]
                        if has_fps_col:
                            result_df.at[idx, 'FPS'] = result["fps"]
                        
                        # Delete original if requested
                        if delete_originals:
                            try:
                                os.remove(result["path"])
                            except Exception as e:
                                print(f"Warning: Could not delete {result['path']}: {e}")
        if save_checkpoint:
            result_df.to_csv(checkpoint_file, index=False)
            print(f"Saved checkpoint to {checkpoint_file}")
    
    # Summarize results
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nResampling complete:")
    print(f"  - Successfully processed: {successful}/{total_videos} videos")
    print(f"  - Failed: {failed}/{total_videos} videos")
    
    if failed > 0:
        print("\nFailed videos:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {os.path.basename(r['path'])}: {r.get('message', 'Unknown error')}")
    
    return result_df