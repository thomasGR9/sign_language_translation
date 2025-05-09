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



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['GLOG_minloglevel'] = '3'      # Suppress Google logging (used by MediaPipe)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Optional: Disable GPU logging messages



logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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
import gc
import psutil




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



def adaptive_detect(image_path, hand_model_path, face_model_path, dominand_hand, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, 
                   min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, 
                   num_hands=2, visualize=False, output_face_blendshapes=True,
                   max_attempts=1, threshold_reduction_factor=0.7, min_threshold=0.2):
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

    
    # Save back to the file with same format
    np.savez(file_path, **arrays)
    




def interpolate_undetected_hand_landmarks_new(directory_path):  
    """
    Interpolate landmarks for frames where hand detection failed.
    Optimized version for faster CPU processing.
    """
    print(f"Starting interpolation for directory: {directory_path}")
    
    # Load detection statistics JSON
    with open(os.path.join(directory_path, 'detection_statistics.json')) as f:
        data = json.load(f)
    
    first_frame_number = round(data['video_info']['fps'] * data['video_info']['start_time'])
    final_frame_number = round(data['video_info']['fps'] * data['video_info']['end_time'])
    
    
    
    # Maximum possible sum of weights for normalization (when all 10 frames are available)
    MAX_WEIGHT_SUM = 2.92722222
    
    # Create a cache for loaded frame data to avoid reloading the same frames
    frame_data_cache = {}
    
    # Helper function to process either dominant or non-dominant hand
    def process_hand_frames(failures, missing_frames_list, is_dominant):
        hand_index = 0 if is_dominant else 1
        hand_type = "dominant" if is_dominant else "non-dominant"
        landmarks_key = 'dom_landmarks' if is_dominant else 'non_dom_landmarks'
        
        
        interpolated_count = 0
        
        for missing_frame in failures:
            frame_number = missing_frame['frame']
            filepath = missing_frame['file']
            
            # Only interpolate frames not at the edges of the video
            if ((frame_number - 5) <= first_frame_number or (frame_number + 5) >= final_frame_number):
                continue
            
            # Find frames with valid detections for interpolation
            interpolation_frames = find_interpolation_frames(frame_number, missing_frames_list)
            
            if not interpolation_frames:
                continue
            
            # Calculate interpolated landmarks
            interpolation_weights_sum = 0
            interpolated_coordinates = np.zeros(shape=(20, 3))
            interpolated_wrist_to_nose = np.zeros(2)
            
            for interp_frame in interpolation_frames:
                weight = 1 / ((frame_number - interp_frame) ** 2)
                interpolation_weights_sum += weight
                
                # Find and load the reference frame - use cached version if available
                cache_key = f"{interp_frame}"
                if cache_key in frame_data_cache:
                    frame_data = frame_data_cache[cache_key]
                else:
                    # Format the filename following the same pattern as the original code
                    interp_partial_filename = data['video_info']['name'] + f"_frame{interp_frame:06d}"
                    try:
                        interp_files = find_file_with_partial_name(
                            interp_partial_filename, 
                            search_dir=directory_path, 
                            recursive=False
                        )
                        
                        if not interp_files:
                            print(f"Warning: Could not find file for frame {interp_frame}")
                            sys.exit(1)
                            
                        interp_filepath = interp_files[0]
                        
                        # Load and cache the frame data
                        frame_data = load_frame_data(interp_filepath)
                        frame_data_cache[cache_key] = frame_data
                        
                    except Exception as e:
                        print(f"Error processing frame {interp_frame}: {e}")
                        sys.exit(1)
                
                # Get landmarks and nose-to-wrist distance for this hand type
                landmarks = frame_data[hand_index]
                nose_to_wrist = frame_data[7][hand_index, :]
                
                # Add weighted contribution
                interpolated_coordinates += weight * landmarks
                interpolated_wrist_to_nose += weight * nose_to_wrist
            
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
                    new_arr[hand_index] = interpolation_confidence
                    return new_arr
                
                def update_nose_to_wrist_scores(matrix):
                    new_matrix = matrix.copy()
                    new_matrix[hand_index, :] = interpolated_wrist_to_nose
                    return new_matrix
                    
                modifications = {
                    landmarks_key: interpolated_coordinates,
                    'interpolation_scores': update_interp_scores,
                    'nose_to_wrist_dist': update_nose_to_wrist_scores
                }
                
                modify_npz_file(
                    file_path=os.path.join(directory_path, filepath),
                    modifications=modifications
                )
                
                interpolated_count += 1
        return interpolated_count        

        
    
    # Process non-dominant hand failures
    missing_non_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['non_dominant_hand_failures']]
    non_dom_count = process_hand_frames(
        data['failed_frames']['non_dominant_hand_failures'],
        missing_non_dominant_frame_list,
        is_dominant=False
    )
    
    # Process dominant hand failures
    missing_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['dominant_hand_failures']]
    dom_count = process_hand_frames(
        data['failed_frames']['dominant_hand_failures'],
        missing_dominant_frame_list,
        is_dominant=True
    )
    
    total_interpolated = non_dom_count + dom_count
    print(f"Total interpolated: {total_interpolated} frames")
    
    return total_interpolated
    

def interpolate_undetected_hand_landmarks_new_wrapper(directory_path):
    gc.collect()
    statistics_file = os.path.join(directory_path, 'detection_statistics.json')
    if not os.path.exists(statistics_file):
        print(f"Warning: Missing detection_statistics.json in {directory_path}")
        return -1, directory_path  # Special return value indicating file not found
    
    try:
        with open(statistics_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {statistics_file}")
        return -1, directory_path
    except Exception as e:
        print(f"Error reading {statistics_file}: {str(e)}")
        return -1, directory_path
    
    first_frame_number = round(data['video_info']['fps'] * data['video_info']['start_time'])
    final_frame_number = round(data['video_info']['fps'] * data['video_info']['end_time'])

    missing_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['dominant_hand_failures']]
    missing_dominant_frame_list_with_interpolation_frames = [frame for frame in missing_dominant_frame_list if find_interpolation_frames(frame, missing_dominant_frame_list)]
    missing_dominant_frame_list_no_edges = [frame for frame in missing_dominant_frame_list_with_interpolation_frames if not ((frame - 5) <= first_frame_number or (frame + 5) >= final_frame_number)]
    number_of_frames_to_interpolate_dom = len(missing_dominant_frame_list_no_edges)


    missing_non_dominant_frame_list = [frame['frame'] for frame in data['failed_frames']['non_dominant_hand_failures']]
    missing_non_dominant_frame_list_with_interpolation_frames = [frame for frame in missing_non_dominant_frame_list if find_interpolation_frames(frame, missing_non_dominant_frame_list)]
    missing_non_dominant_frame_list_no_edges = [frame for frame in missing_non_dominant_frame_list_with_interpolation_frames if not ((frame - 5) <= first_frame_number or (frame + 5) >= final_frame_number)]
    number_of_frames_to_interpolate_non_dom = len(missing_non_dominant_frame_list_no_edges)

    total_number_of_frames_to_interpolate = number_of_frames_to_interpolate_dom + number_of_frames_to_interpolate_non_dom
    
    
    total_actually_interpolated = interpolate_undetected_hand_landmarks_new(directory_path)
    
    if total_number_of_frames_to_interpolate != total_actually_interpolated:
        print(f"The function missed some interpolations. Expected {total_number_of_frames_to_interpolate} but actually got {total_actually_interpolated} , exiting...")
        sys.exit(1)
    else:
        gc.collect()
        return total_actually_interpolated, None





def is_data_frame_good(frame_data, expected_shapes_list):
    if len(frame_data)!=10:
        print(f"Expected 10 outputs but got {len(frame_data)}")
        return False
    for i in range(8):
        if i==6:
            continue
        if frame_data[i].shape != expected_shapes_list[i]:
            print(f"Dimensions problem in loaded frame, in intex {i}")
            return False
    if (not isinstance(frame_data[8], int)):
        print(f"Frame idx is not an integer")
        return False
    return True

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
    Vectorized version: Convert Cartesian velocities to spherical coordinate features.
    
    Args:
        ties: NumPy array of shape (20, 3) with Cartesian velocities
        
    Returns:
        NumPy array of shape (20, 5) with spherical features:
            [vmagnitude, ϕsin, ϕcos, θsin, θcos]
    """
    # Extract Cartesian components
    ux = velocities[:, 0]
    uy = velocities[:, 1]
    uz = velocities[:, 2]
    
    # Calculate velocity magnitude
    vmagnitude = np.sqrt(ux**2 + uy**2 + uz**2)
    
    # Create result array with magnitude in first column
    spherical_features = np.zeros((velocities.shape[0], 5))
    spherical_features[:, 0] = vmagnitude
    
    # Create mask for non-zero velocities
    nonzero_mask = vmagnitude > 0
    
    if np.any(nonzero_mask):
        # Calculate azimuth angle (phi) for non-zero velocities
        phi = np.zeros_like(vmagnitude)
        phi[nonzero_mask] = np.arctan2(uy[nonzero_mask], ux[nonzero_mask])
        
        # Store sin and cos of phi
        spherical_features[:, 1] = np.sin(phi)
        spherical_features[:, 2] = np.cos(phi)
        
        # Calculate elevation angle (theta) for non-zero velocities
        cos_theta = np.zeros_like(vmagnitude)
        cos_theta[nonzero_mask] = np.clip(uz[nonzero_mask] / vmagnitude[nonzero_mask], -1.0, 1.0)
        
        theta = np.arccos(cos_theta)
        spherical_features[:, 3] = np.sin(theta)
        spherical_features[:, 4] = cos_theta
    
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


def sorted_npz_files_checked(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all NPZ files in the directory
        npz_files = sorted(glob.glob(os.path.join(directory_path, "*.npz")))
    else:
        print(f"Directory path {directory_path} doesn't exist or it isn't a directory")
        sys.exit(1)
        
    
    # Skip if no files found
    if not npz_files:
        print(f"No NPZ files found in {directory_path}")
        sys.exit(1)
    
    
    with open(os.path.join(directory_path, 'detection_statistics.json')) as f:
        statistics_file = json.load(f)
    
    if statistics_file['video_info']['total_frames'] != len(npz_files):
        print("npz filepath list contain different amount of items than total frames")
        sys.exit(1)

    

    expected_shapes_list = [(20, 3), (20, 3), (2,), (2,), (2,), (52,), 0, (2, 2)]
    # Create a mapping of frame indices to file paths and check if all files are good
    frame_to_file = {}
    for file_path in npz_files:
        try:
            frame_data = load_frame_data(file_path)
        except Exception as e:
            print(f"Error loading frame with path: {file_path}: {e}")
            sys.exit(1)
            
        if not is_data_frame_good(frame_data=frame_data, expected_shapes_list=expected_shapes_list):
            print(f"Wrong loaded frame data dimensions for frame with path {file_path}, exiting...")
            sys.exit(1)
        frame_idx = frame_data[8]  # Index for frame_idx
        frame_to_file[frame_idx] = file_path

    
    frame_indices = sorted(frame_to_file.keys())
    if not all(frame_indices[i+1] - frame_indices[i] == 1 for i in range(len(frame_indices) - 1)):
        print("Consecutive frames are not different by one frame")
        sys.exit(1)

    

    min_frame = min(frame_indices)
    max_frame = max(frame_indices)
    return frame_to_file, frame_indices, min_frame, max_frame

def compute_landmark_velocities(directory_path):
    """
    Compute velocity features for hand landmarks using central differencing with two window sizes,
    and convert to spherical coordinates.
    
    Args:
        directory_path (str): Path to the directory containing frame NPZ files
    
    Returns:
        int: Number of frames processed
    """
    frame_to_file, frame_indices, min_frame, max_frame = sorted_npz_files_checked(directory_path=directory_path)
    processed_count = 0
    safe_margin = 5  # Skip processing frames within 5 frames of the edge
    frame_cache = {}
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
            

            continue
        
        if curr_idx==safe_margin:
            # Load current frame
            current_file_path = frame_to_file[curr_idx]
            try:
                curr_frame_data = load_frame_data(current_file_path)
            except Exception as e:
                print(f"Error loading frame {curr_idx}: {e}")
                sys.exit(1)
            
        
        
            # Load all potentially needed frames in the -5 to +5 range
            for offset in range(-5, 6):
                check_idx = curr_idx + offset
                if check_idx in frame_to_file:
                    try:
                        frame_cache[check_idx] = load_frame_data(frame_to_file[check_idx])
                    except Exception as e:
                        print(f"Error loading frame {check_idx}: {e}")
                        sys.exit(1)
                else:
                    print(f"Frame: {check_idx} not available")
                    sys.exit(1)  
        else:
            try:
                frame_cache[curr_idx+5] = load_frame_data(frame_to_file[curr_idx+5])
                del frame_cache[curr_idx-6]
            except Exception as e:
                print(f"Error loading frame {curr_idx+5}: {e}")
                sys.exit(1)
            current_file_path = frame_to_file[curr_idx]
            curr_frame_data = frame_cache[curr_idx]
        

        
        # Initialize velocity arrays in Cartesian coordinates
        dom_velocity_small_cart = np.zeros((20, 3))
        dom_velocity_large_cart = np.zeros((20, 3))
        non_dom_velocity_small_cart = np.zeros((20, 3))
        non_dom_velocity_large_cart = np.zeros((20, 3))
        
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
        
        def satisfied_symmetric_windows_condition(positive_offset, negative_offset, is_dom_hand):
            return (curr_idx + positive_offset in frame_cache and frame_cache[curr_idx + positive_offset] is not None and 
            is_valid_detection(frame_cache[curr_idx + positive_offset], is_dom_hand) and 
            curr_idx - negative_offset in frame_cache and frame_cache[curr_idx - negative_offset] is not None and 
            has_value(frame_cache[curr_idx - negative_offset], is_dom_hand))
        
        def single_positive_check(positive_offset, is_dom_hand):
            return (curr_idx + positive_offset in frame_cache and frame_cache[curr_idx + positive_offset] is not None and 
            is_valid_detection(frame_cache[curr_idx + positive_offset], is_dom_hand))
        
        def negative_check(negative_offset, is_dom_hand):
            return (curr_idx - negative_offset in frame_cache and frame_cache[curr_idx - negative_offset] is not None and 
            has_value(frame_cache[curr_idx - negative_offset], is_dom_hand))
            
        def velocity_and_confidence_calculations(positive_offset, negative_offset, is_dom_hand):
            if is_dom_hand:
                index=0
            else:
                index=1
            distance_of_offsets = positive_offset + negative_offset
            velocity_cart = (frame_cache[curr_idx + positive_offset][index] - frame_cache[curr_idx - negative_offset][index]) / distance_of_offsets
            wrist_velocity = (frame_cache[curr_idx + positive_offset][7][index, :] - frame_cache[curr_idx - negative_offset][7][index, :]) / distance_of_offsets
            conf = min(frame_cache[curr_idx + positive_offset][2][index], frame_cache[curr_idx - negative_offset][2][index])  
            # Calculate source quality factor (average of interpolation confidences)
            t_plus_1_interp = frame_cache[curr_idx + positive_offset][3][index]
            t_minus_1_interp = frame_cache[curr_idx - negative_offset][3][index]
            source_quality = (t_plus_1_interp + t_minus_1_interp) / 2.0
            return velocity_cart, wrist_velocity, conf, source_quality
        
        
        
        # ===== DOMINANT HAND VELOCITY CALCULATION =====
        
        # Small window [-1, +1] velocity with fallbacks
        true_positive_offset_dom_small =0
        true_negative_offset_dom_small =0
        if satisfied_symmetric_windows_condition(positive_offset=1, negative_offset=1, is_dom_hand=True):
            # Ideal case: (t+1, t-1)
            true_positive_offset_dom_small=1
            true_negative_offset_dom_small=1
            dom_small_method_weight = 1.0  # Ideal frames
            
        elif satisfied_symmetric_windows_condition(positive_offset=2, negative_offset=2, is_dom_hand=True):
            # Fallback 1: (t+2, t-2)
            true_positive_offset_dom_small=2
            true_negative_offset_dom_small=2
            dom_small_method_weight = 0.8  # Wider symmetric window
        elif single_positive_check(positive_offset=2, is_dom_hand=True):
            if negative_check(negative_offset=1, is_dom_hand=True):
                # Fallback 2: (t+2, t-1)
                true_positive_offset_dom_small=2
                true_negative_offset_dom_small=1
                dom_small_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif is_valid_detection(curr_frame_data, True):
                # Fallback 3: (t+2, t)
                true_positive_offset_dom_small=2
                dom_small_method_weight = 0.4  # One-sided derivative
                
                
        elif is_valid_detection(curr_frame_data, True):
            if negative_check(negative_offset=1, is_dom_hand=True):
                # Fallback 4: (t, t-1)
                true_negative_offset_dom_small=1
                dom_small_method_weight = 0.4  # One-sided derivative
                
            elif negative_check(negative_offset=2, is_dom_hand=True):
                # Fallback 5: (t, t-2)
                true_negative_offset_dom_small=2
                dom_small_method_weight = 0.4  # One-sided derivative
        if (true_positive_offset_dom_small!=0 or true_negative_offset_dom_small!=0):
            dom_velocity_small_cart, wrist_velocity_small[0, :], dom_small_conf, dom_small_source_quality = velocity_and_confidence_calculations(true_positive_offset_dom_small, true_negative_offset_dom_small, is_dom_hand=True)
        
        # Large window [-5, +5] velocity with fallbacks            
        true_positive_offset_dom_large =0
        true_negative_offset_dom_large =0
        if satisfied_symmetric_windows_condition(positive_offset=5, negative_offset=5, is_dom_hand=True):
            # Ideal case: (t+5, t-5)
            true_positive_offset_dom_large =5
            true_negative_offset_dom_large =5
            dom_large_method_weight = 1.0  # Ideal frames
            
        elif satisfied_symmetric_windows_condition(positive_offset=4, negative_offset=4, is_dom_hand=True):
        # Fallback 1: (t+4, t-4)
            true_positive_offset_dom_large = 4
            true_negative_offset_dom_large = 4
            dom_large_method_weight = 0.8  # Wider symmetric window
            
        elif satisfied_symmetric_windows_condition(positive_offset=3, negative_offset=3, is_dom_hand=True):
            # Fallback 2: (t+3, t-3)
            true_positive_offset_dom_large = 3
            true_negative_offset_dom_large = 3
            dom_large_method_weight = 0.8  # Wider symmetric window
            
        # Asymmetric fallbacks for large window
        elif single_positive_check(positive_offset=5, is_dom_hand=True):
            if negative_check(negative_offset=4, is_dom_hand=True):
                # Fallback 3: (t+5, t-4)
                true_positive_offset_dom_large = 5
                true_negative_offset_dom_large = 4
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif negative_check(negative_offset=3, is_dom_hand=True):
                # Fallback 4: (t+5, t-3)
                true_positive_offset_dom_large = 5
                true_negative_offset_dom_large = 3
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
        
        elif single_positive_check(positive_offset=4, is_dom_hand=True):
            if negative_check(negative_offset=5, is_dom_hand=True):
                # Fallback 5: (t+4, t-5)
                true_positive_offset_dom_large = 4
                true_negative_offset_dom_large = 5
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif negative_check(negative_offset=3, is_dom_hand=True):
                # Fallback 6: (t+4, t-3)
                true_positive_offset_dom_large = 4
                true_negative_offset_dom_large = 3
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
        elif single_positive_check(positive_offset=3, is_dom_hand=True):
            if negative_check(negative_offset=5, is_dom_hand=True):
                # Fallback 7: (t+3, t-5)
                true_positive_offset_dom_large = 3
                true_negative_offset_dom_large = 5
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif negative_check(negative_offset=4, is_dom_hand=True):
                # Fallback 8: (t+3, t-4)
                true_positive_offset_dom_large = 3
                true_negative_offset_dom_large = 4
                dom_large_method_weight = 0.6  # Asymmetric window maintaining center point

        if (true_positive_offset_dom_large!=0 or true_negative_offset_dom_large!=0):
            dom_velocity_large_cart, wrist_velocity_large[0, :], dom_large_conf, dom_large_source_quality = velocity_and_confidence_calculations(true_positive_offset_dom_large, true_negative_offset_dom_large, is_dom_hand=True)
        
        # ===== NON-DOMINANT HAND VELOCITY CALCULATION =====

    # Small window [-1, +1] velocity with fallbacks for non-dominant hand
        true_positive_offset_non_dom_small =0
        true_negative_offset_non_dom_small =0
        if satisfied_symmetric_windows_condition(positive_offset=1, negative_offset=1, is_dom_hand=False):
            # Ideal case: (t+1, t-1)
            true_positive_offset_non_dom_small =1
            true_negative_offset_non_dom_small =1
            non_dom_small_method_weight = 1.0  # Ideal frames
            
        elif satisfied_symmetric_windows_condition(positive_offset=2, negative_offset=2, is_dom_hand=False):
            # Fallback 1: (t+2, t-2)
            true_positive_offset_non_dom_small =2
            true_negative_offset_non_dom_small =2
            non_dom_small_method_weight = 0.8  # Wider symmetric window
            
        elif single_positive_check(positive_offset=2, is_dom_hand=False):
            if negative_check(negative_offset=1, is_dom_hand=False):
                # Fallback 2: (t+2, t-1)
                true_positive_offset_non_dom_small =2
                true_negative_offset_non_dom_small =1
                non_dom_small_method_weight = 0.6  # Ideal frames
                
            elif is_valid_detection(curr_frame_data, False):
                # Fallback 3: (t+2, t)
                true_positive_offset_non_dom_small =2
                non_dom_small_method_weight = 0.4  # One-sided derivative
                
        elif is_valid_detection(curr_frame_data, False):
            if negative_check(negative_offset=1, is_dom_hand=False):
                # Fallback 4: (t, t-1)
                true_negative_offset_non_dom_small =1
                non_dom_small_method_weight = 0.4  # One-sided derivative
                
            elif negative_check(negative_offset=2, is_dom_hand=False):
                # Fallback 5: (t, t-2)
                true_negative_offset_non_dom_small =2
                non_dom_small_method_weight = 0.4  # One-sided derivative
        
        if (true_positive_offset_non_dom_small!=0 or true_negative_offset_non_dom_small!=0):
            non_dom_velocity_small_cart, wrist_velocity_small[1, :], non_dom_small_conf, non_dom_small_source_quality = velocity_and_confidence_calculations(true_positive_offset_non_dom_small, true_negative_offset_non_dom_small, is_dom_hand=False)

        true_positive_offset_non_dom_large =0
        true_negative_offset_non_dom_large =0
        # Large window [-5, +5] velocity with fallbacks for non-dominant hand
        if satisfied_symmetric_windows_condition(positive_offset=5, negative_offset=5, is_dom_hand=False):
            # Ideal case: (t+5, t-5)
            true_positive_offset_non_dom_large =5
            true_negative_offset_non_dom_large =5
            non_dom_large_method_weight = 1.0  # Ideal frames
            
        elif satisfied_symmetric_windows_condition(positive_offset=4, negative_offset=4, is_dom_hand=False):
            # Fallback 1: (t+4, t-4)
            true_positive_offset_non_dom_large =4
            true_negative_offset_non_dom_large =4
            non_dom_large_method_weight = 0.8  # Wider symmetric window
            
        elif satisfied_symmetric_windows_condition(positive_offset=3, negative_offset=3, is_dom_hand=False):
            # Fallback 2: (t+3, t-3)
            true_positive_offset_non_dom_large =3
            true_negative_offset_non_dom_large =3
            non_dom_large_method_weight = 0.8  # Wider symmetric window
            
        # Asymmetric fallbacks for large window
        elif single_positive_check(positive_offset=5, is_dom_hand=False):
            if negative_check(negative_offset=4, is_dom_hand=False):
                # Fallback 3: (t+5, t-4)
                true_positive_offset_non_dom_large =5
                true_negative_offset_non_dom_large =4
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif negative_check(negative_offset=3, is_dom_hand=False):
                # Fallback 4: (t+5, t-3)
                true_positive_offset_non_dom_large =5
                true_negative_offset_non_dom_large =3
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
        elif single_positive_check(positive_offset=4, is_dom_hand=False):
            if negative_check(negative_offset=5, is_dom_hand=False):
                # Fallback 5: (t+4, t-5)
                true_positive_offset_non_dom_large =4
                true_negative_offset_non_dom_large =5
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
            elif negative_check(negative_offset=3, is_dom_hand=False):
                # Fallback 6: (t+4, t-3)
                true_positive_offset_non_dom_large =4
                true_negative_offset_non_dom_large =3
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
                
        elif single_positive_check(positive_offset=3, is_dom_hand=False):
            if negative_check(negative_offset=5, is_dom_hand=False):
                # Fallback 7: (t+3, t-5)
                true_positive_offset_non_dom_large =3
                true_negative_offset_non_dom_large =5
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point
            
            elif negative_check(negative_offset=4, is_dom_hand=False):
                # Fallback 8: (t+3, t-4)
                true_positive_offset_non_dom_large =3
                true_negative_offset_non_dom_large =4
                non_dom_large_method_weight = 0.6  # Asymmetric window maintaining center point

        if (true_positive_offset_non_dom_large!=0 or true_negative_offset_non_dom_large!=0):
            non_dom_velocity_large_cart, wrist_velocity_large[1, :], non_dom_large_conf, non_dom_large_source_quality = velocity_and_confidence_calculations(true_positive_offset_non_dom_large, true_negative_offset_non_dom_large, is_dom_hand=False)
            

        
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
        

    
    print(f"Velocity computation complete. Processed {processed_count} frames.")
    return processed_count


def compute_landmark_velocities_wrapper(directory_path):
    print(f"Computing velocities for directory path: {directory_path}")
    gc.collect()
    with open(os.path.join(directory_path, 'detection_statistics.json')) as f:
        data = json.load(f)
    expected_number_of_frames_to_compute = data['video_info']['processed_frames']
    number_of_frames_actually_computed = compute_landmark_velocities(directory_path=directory_path)
    if expected_number_of_frames_to_compute != number_of_frames_actually_computed:
        print(f"Expected number of frames to compute: {expected_number_of_frames_to_compute} is not equal to the actually computed {number_of_frames_actually_computed} for directory_path: {directory_path}")
        sys.exit(1)
    gc.collect()
    return number_of_frames_actually_computed



def process_landmark_velocities_parallel(dataframe, velocity_func, batch_size=4, 
                               report_dir=None, start_from_row=0):
    """
    Process multiple directories for velocity calculation from a dataframe with checkpointing and logging.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with a 'landmarks_file_path' column
        velocity_func (function): Function that calculates velocities for a directory and returns processed frames count
        batch_size (int): Size of directory batches to process
        report_dir (str): Directory to save checkpoint and report files
                         If None, no checkpointing or reporting is done
        start_from_row (int): Row index to start processing from (for resuming)
        
    Returns:
        pd.DataFrame: The input dataframe with an additional 'processed_frames' column
    """
    # Setup logging and checkpointing
    checkpoint_file = None
    report_data = {
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "total_directories": 0,
        "processed_directories": 0,
        "successful_processing": 0,
        "failed_processing": 0,
        "last_processed_row": start_from_row - 1,
        "status": "in_progress"
    }
    
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(report_dir, f"velocity_processing_{timestamp}.json")
        
        # Check if we're resuming from a crash
        if start_from_row > 0:
            print(f"Resuming from row {start_from_row}")
            report_data["notes"] = f"Resumed from row {start_from_row}"
        
        # Initial write
        with open(checkpoint_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    # Add directory column and reset index to ensure row tracking works correctly
    dataframe = dataframe.reset_index(drop=True)

    
    # If resuming, apply a filter based on start_from_row
    if start_from_row > 0:
        working_df = dataframe.iloc[start_from_row:].copy()
    else:
        working_df = dataframe.copy()
    
    # More efficient way to get unique directories
    unique_directories = working_df['landmarks_file_path'].tolist()
    total_dirs = len(unique_directories)
    
    report_data["total_directories"] = total_dirs
    print(f"Found {total_dirs} unique directories to process")
    
    # Limit concurrent directories based on system resources
    max_concurrent_dirs = batch_size
    print(f"Using {max_concurrent_dirs} concurrent processes")
    
    # Store results and track progress
    overall_start_time = time.time()
    processed_dirs = 0
    successful_processing = 0
    failed_processing = 0
    last_processed_row = start_from_row - 1
    
    # Update checkpoint function
    def update_checkpoint():
        if checkpoint_file:
            current_time = datetime.now()
            elapsed = time.time() - overall_start_time
            
            report_data["processed_directories"] = processed_dirs
            report_data["successful_processing"] = successful_processing
            report_data["failed_processing"] = failed_processing
            report_data["last_processed_row"] = last_processed_row
            report_data["duration_seconds"] = elapsed
            report_data["last_updated"] = current_time.isoformat()
            
            # Write to a temp file first then rename to avoid corruption if crashed during write
            temp_file = checkpoint_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Atomic rename operation
            if os.path.exists(temp_file):
                os.replace(temp_file, checkpoint_file)
    
    total_processed_frames = 0
    # Process directories in batches
    try:
        for batch_idx in range(0, len(unique_directories), batch_size):
            batch_directories = unique_directories[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(unique_directories) + batch_size - 1)//batch_size} "
                  f"({len(batch_directories)} directories)")
            
            mem_available = psutil.virtual_memory().available / (1024**3)  # GB
            if mem_available < 2:  # Less than 2GB available
                print(f"Warning: Only {mem_available:.1f}GB memory available. Waiting 1 second...")
                time.sleep(1)  # Wait for other processes to finish
                gc.collect()
            
            # Process this batch of directories in parallel
            with ProcessPoolExecutor(max_workers=max_concurrent_dirs) as executor:
                # Submit all directories in this batch
                future_to_dir = {
                    executor.submit(velocity_func, dir_path): dir_path 
                    for dir_path in batch_directories
                }
                
                # Process results as they complete
                for future in as_completed(future_to_dir):
                    dir_path = future_to_dir[future]
                    try:
                        # Get velocity calculation results (frames processed count)
                        processed_frames = future.result()
                        dir_indices = dataframe[dataframe['landmarks_file_path'] == dir_path].index
                        dataframe.loc[dir_indices, 'processed_frames'] = processed_frames
                        processed_dirs += 1
                        total_processed_frames += processed_frames
                        
                        # Track successful processing
                        if processed_frames > 0:
                            successful_processing += 1
                        
                        # Find the maximum row index for this landmarks_file_path
                        dir_rows = working_df[working_df['landmarks_file_path'] == dir_path].index
                        if len(dir_rows) > 0:
                            last_processed_row = max(last_processed_row, max(dir_rows) + start_from_row)
                    
                    except Exception as e:
                        print(f"Error processing directory {dir_path}: {str(e)}")
                        sys.exit(1)
            
            # Update checkpoint after each batch
            update_checkpoint()
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if checkpoint_file:
            report_data["status"] = "crashed"
            report_data["error"] = str(e)
            update_checkpoint()
        sys.exit(1)
    
    # Calculate final statistics
    total_elapsed = time.time() - overall_start_time
    
    print(f"\nProcessing completed in {total_elapsed/60:.1f} minutes")
    print(f"Total directories processed: {processed_dirs}/{total_dirs}")
    print(f"Total frames processed: {total_processed_frames}")
    print(f"Successful directories: {successful_processing}")
    print(f"Failed directories: {failed_processing}")
    
    # Final checkpoint update
    if checkpoint_file:
        report_data["status"] = "completed"
        report_data["end_time"] = datetime.now().isoformat()
        report_data["duration_seconds"] = total_elapsed
        report_data["processed_directories"] = processed_dirs
        report_data["successful_processing"] = successful_processing
        report_data["failed_processing"] = failed_processing
        report_data["total_processed_frames"] = total_processed_frames
        update_checkpoint()
    
    return dataframe


def find_latest_velocity_checkpoint(report_dir):
    """
    Find the latest velocity checkpoint file in the report directory.
    
    Args:
        report_dir (str): Directory containing checkpoint files
        
    Returns:
        tuple: (last_processed_row, checkpoint_data) or (None, None) if not found
    """
    if not os.path.exists(report_dir):
        return None, None
    
    checkpoint_files = [f for f in os.listdir(report_dir) 
                        if f.startswith("velocity_processing_") and f.endswith(".json")]
    
    if not checkpoint_files:
        return None, None
    
    # Sort by modification time (most recent first)
    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(report_dir, f)))
    latest_path = os.path.join(report_dir, latest_file)
    
    try:
        with open(latest_path, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data["last_processed_row"], checkpoint_data
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

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



def process_video_new_2(video_path, num_workers, adaptive_detect_func=adaptive_detect, hand_model_path=hand_model_path, face_model_path=face_model_path,
                 min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                 min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                 num_hands=2, output_face_blendshapes=True,
                 max_attempts=1, threshold_reduction_factor=0.7, min_threshold=0.2, 
                 frame_step=1, start_time_seconds=0, end_time_seconds=None,
                 save_failure_screenshots=False,
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


    # Use the extracted dominant hand instead of the parameter
    dominand_hand = extracted_dominant_hand


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
    

    
    # When in batch mode, conserve resources even more and limit output
    if batch_mode:
        # Further reduce threads in batch mode to ensure stability
        print(f"[{os.path.basename(str(video_path))}] Using {num_workers} worker threads")
    else:
        print(f"Video: {video_name}")
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







def batch_call_process_video_validated(video_path, start_time_seconds=0, end_time_seconds=None):
    """
    Simple wrapper that validates all frames were processed and stops if not
    """
 
   
    print(f"\nStarting processing of video: {os.path.basename(video_path)}", flush=True)
    start_time = time.time()
    
    # Force garbage collection before processing
    gc.collect()
    
    try:
        # Process the video with fewer threads to reduce resource usage
        result = process_video_new_2(
            video_path=video_path,
            adaptive_detect_func=adaptive_detect,
            hand_model_path=hand_model_path,
            face_model_path=face_model_path,
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
            batch_mode=True,
            num_workers=4  # Reduced thread count
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Get the output directory and stats file
        output_dir = result  # process_video returns the output directory path
        stats_file = os.path.join(output_dir, "detection_statistics.json")
        
        # Check that all frames were processed
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
                
            processed_frames = stats["video_info"]["processed_frames"]
            total_frames = stats["video_info"]["total_frames"]
            
            # Verify frames match
            if processed_frames < total_frames:
                error_msg = f"CRITICAL ERROR: Incomplete processing detected - only {processed_frames}/{total_frames} frames were processed"
                print(f"\n{'!'*80}\n{error_msg}\n{'!'*80}\n", flush=True)
                
                # Force garbage collection
                gc.collect()
                
                # Raise an exception to stop the batch process
                sys.exit(1)
            
            print(f"Successfully processed all {processed_frames}/{total_frames} frames in {elapsed_time:.2f} seconds", flush=True)
        else:
            error_msg = f"CRITICAL ERROR: Statistics file not found: {stats_file}"
            print(f"\n{'!'*80}\n{error_msg}\n{'!'*80}\n", flush=True)
            sys.exit(1)
        
        # Force garbage collection after processing
        gc.collect()
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR during processing ({elapsed_time:.2f} seconds): {str(e)}", flush=True)
        
        # Force garbage collection in case of error
        gc.collect()
        
        # Re-raise the exception to stop batch_process_videos
        raise



















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
        video_fps = 15
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
            if dom_hand_rate < current_detection_threshold_dom:
                print(f"Low dominant hand detection rate for {video_name}: Dom={dom_hand_rate:.1f}%")
                print(f"Threshold was: {current_detection_threshold_dom}%")
                print(f"Deleting directory: {output_dir}")

                # Delete the directory
                shutil.rmtree(output_dir)

                # Update statistics
                stats["processing_info"]["directories_deleted"] += 1
                stats["deleted_directories_summary"]["count"] += 1

                video_detail["status"] = "deleted"
                video_detail["reason"] = "low_dominant_hand_detection_rate"


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
    gc.collect()
    video_path, desired_fps = video_info
    try:
        # Get original video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"path: {video_path}, error opening video, message: {str(e)}")
            sys.exit(1)

        
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
                print(f"Couldn't read frame {i} in {video_path}")
                break
                
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Get new video metadata
        new_cap = cv2.VideoCapture(output_path)
        new_frame_count = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_fps = new_cap.get(cv2.CAP_PROP_FPS)
        if round(new_fps) != round(desired_fps):
            print(f"Error lowering fps, desired {desired_fps} but got {new_fps}")
            sys.exit(1)
        new_total_frames = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_duration = new_total_frames / new_fps
        if np.abs(new_duration-duration)>0.3:
            print(f"Too different video duration after lowering fps, before was {duration}, now is {new_duration}")
            sys.exit(1)
        new_cap.release()
        gc.collect()
        return {
            "path": video_path,
            "new_path": output_path,
            "status": "success",
            "frame_count": new_frame_count,
            "fps": new_fps
        }
        
    except Exception as e:
        print(f"path: {video_path}, status: error, message: {str(e)}")
        sys.exit(1)

def resample_videos_in_dataframe(df, desired_fps, batch_size, save_checkpoint=False, checkpoint_file='resampling_progress.csv', file_path_col='file_path', 
                                delete_originals=False):
    """
    Efficiently resamples videos in a dataframe with parallel processing and batching.
    
    Parameters:
        df (pandas.DataFrame): Dataframe containing video paths
        desired_fps (float): The desired FPS for the output videos
        file_path_col (str): Name of the column containing file paths
        delete_originals (bool): Whether to delete original videos after resampling
        batch_size (int): Size of batches for processing
        
    Returns:
        pandas.DataFrame: Updated dataframe with new file paths and metadata
    """
    # Use original dataframe or create a copy
    result_df = df.copy()
    
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
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
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
                                sys.exit(1)
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

def find_landmark_directories(root_path):
    """
    Finds all directories with names ending in '_landmarks' under the given root path.
    
    Parameters:
    root_path (str): The path to the root directory to search in
    
    Returns:
    pandas.DataFrame: A DataFrame with one column 'landmarks_file_path' containing paths to landmark directories
    """
    landmark_paths = []
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check each directory name
        for dirname in dirnames:
            if dirname.endswith("_landmarks"):
                # Add the full path to our list
                dirname = dirname + f"/"
                full_path = os.path.join(dirpath, dirname)
                landmark_paths.append(full_path)
    
    # Create a DataFrame from the list of paths
    df = pd.DataFrame({'landmarks_file_path': landmark_paths})
    
    return df

def remove_low_detection_directories(df, dom_threshold, non_dom_threshold, column_name='landmarks_file_path', base_dir=None):
    """
    Removes directories where hand detection rates fall below a threshold.
    
    Args:
        df (pandas.DataFrame): DataFrame containing directory paths
        threshold (float): Minimum acceptable detection rate (0.0 to 1.0)
        column_name (str): Name of the column containing directory paths
        base_dir (str, optional): Base directory to resolve relative paths. If None, paths are used as-is.
        
    Returns:
        pandas.DataFrame: Updated DataFrame with rows removed for deleted directories
        list: List of removed directory paths
    """
    # Create a copy of the DataFrame to avoid modifying the original during iteration
    updated_df = df.copy()
    total_removed = 0
    
    for index, row in df.iterrows():
        rel_directory_path = row[column_name]
        
        # Resolve the full path if base_dir is provided
        if base_dir:
            directory_path = os.path.join(base_dir, rel_directory_path)
        else:
            directory_path = rel_directory_path
            
        
        directory_path = os.path.normpath(directory_path)
        
        # Make sure the path exists
        if not os.path.exists(directory_path):
            print(f"Warning: Directory not found: {directory_path}")
            sys.exit(1)
            
        # Construct path to the json file
        json_path = os.path.join(directory_path, 'detection_statistics.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found in directory: {json_path}")
            sys.exit(1)
            
        # Load the JSON file
        try:
            with open(json_path, 'r') as f:
                stats = json.load(f)
                
            # Extract detection rates
            dominant_rate = stats['detection_rates']['dominant_hand']['detection_rate']
            non_dominant_rate = stats['detection_rates']['non_dominant_hand']['detection_rate']
            total_frames = stats['video_info']['total_frames']
            processed_frames = stats['video_info']['processed_frames']
            
            # Check if either rate is below threshold
            if (dominant_rate < dom_threshold) or (non_dominant_rate < non_dom_threshold) or (total_frames != processed_frames):
                print(f"Removing directory due to low detection rates: {directory_path}")
                print(f"  Dominant hand: {dominant_rate}, Non-dominant hand: {non_dominant_rate}")
                
                # Remove the directory
                shutil.rmtree(directory_path)
                total_removed += 1
                # Track removed directories

                
                # Mark this row for removal from the DataFrame
                updated_df = updated_df.drop(index)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_path}: {str(e)}")
            sys.exit(1)
    
    print(f"Removed {total_removed} directories")
    
    
    return updated_df


def process_landmarks_dataframe_new(dataframe, interpolate_func, batch_size=4, 
                               report_dir=None, start_from_row=0):
    """
    Process multiple directories from a dataframe with checkpointing and logging.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with a 'landmarks_file_path' column
        max_concurrent_dirs (int): Maximum number of directories to process in parallel
        batch_size (int): Size of directory batches to process
        report_dir (str): Directory to save checkpoint and report files
                         If None, no checkpointing or reporting is done
        start_from_row (int): Row index to start processing from (for resuming)
        
    Returns:
        pd.DataFrame: The input dataframe with an additional 'interpolated_frames' column
    """
    # Setup logging and checkpointing
    checkpoint_file = None
    report_data = {
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "total_directories": 0,
        "processed_directories": 0,
        "successful_interpolations": 0,
        "failed_interpolations": 0,
        "last_processed_row": start_from_row - 1,
        "status": "in_progress"
    }
    missing_statistics_files = []
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(report_dir, f"landmarks_processing_{timestamp}.json")
        
        # Check if we're resuming from a crash
        if start_from_row > 0:
            print(f"Resuming from row {start_from_row}")
            report_data["notes"] = f"Resumed from row {start_from_row}"
        
        # Initial write
        with open(checkpoint_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    # Add directory column and reset index to ensure row tracking works correctly
    dataframe = dataframe.reset_index(drop=True)

    
    # If resuming, apply a filter based on start_from_row
    if start_from_row > 0:
        working_df = dataframe.iloc[start_from_row:].copy()
    else:
        working_df = dataframe.copy()
    
    
    unique_directories = list(working_df['landmarks_file_path'])
    total_dirs = len(unique_directories)
    
    report_data["total_directories"] = total_dirs
    print(f"Found {total_dirs} unique directories to process")
    
    # Limit concurrent directories based on system resources
    max_concurrent_dirs = batch_size
    print(f"Using {max_concurrent_dirs} concurrent processes")
    
    # Store results and track progress
    overall_start_time = time.time()
    processed_dirs = 0
    successful_interpolations = 0
    failed_interpolations = 0
    last_processed_row = start_from_row - 1
    
    # Update checkpoint function
    def update_checkpoint():
        if checkpoint_file:
            current_time = datetime.now()
            elapsed = time.time() - overall_start_time
            
            report_data["processed_directories"] = processed_dirs
            report_data["successful_interpolations"] = successful_interpolations
            report_data["failed_interpolations"] = failed_interpolations
            report_data["last_processed_row"] = last_processed_row
            report_data["duration_seconds"] = elapsed
            report_data["last_updated"] = current_time.isoformat()
            
            # Write to a temp file first then rename to avoid corruption if crashed during write
            temp_file = checkpoint_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Atomic rename operation
            if os.path.exists(temp_file):
                os.replace(temp_file, checkpoint_file)
                
    
    total_interpolated=0
    # Process directories in batches
    try:
        for batch_idx in range(0, len(unique_directories), batch_size):
            batch_directories = unique_directories[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(unique_directories) + batch_size - 1)//batch_size} "
                  f"({len(batch_directories)} directories)")
            
            mem_available = psutil.virtual_memory().available / (1024**3)  # GB
            if mem_available < 2:  # Less than 2GB available
                print(f"Warning: Only {mem_available:.1f}GB memory available. Waiting 1 second...")
                time.sleep(1)  # Wait for other processes to finish
                gc.collect()
            
            # Process this batch of directories in parallel
            with ProcessPoolExecutor(max_workers=max_concurrent_dirs) as executor:
                # Submit all directories in this batch
                future_to_dir = {
                    executor.submit(interpolate_func, dir_path): dir_path 
                    for dir_path in batch_directories
                }
                
                # Process results as they complete
                for future in as_completed(future_to_dir):
                    dir_path = future_to_dir[future]
                    try:
                        # Get interpolation results
                        interpolated_frames, missing_path = future.result()
                        if interpolated_frames==-1:
                            missing_statistics_files.append(missing_path)
                            print(f"Skipping directory due to missing files: {dir_path}")
                            failed_interpolations += 1
                            continue
                        dir_indices = dataframe[dataframe['landmarks_file_path'] == dir_path].index
                        dataframe.loc[dir_indices, 'interpolated_frames'] = interpolated_frames
                        processed_dirs += 1
                        total_interpolated += interpolated_frames
                        
                        # Track successful interpolations
                        if interpolated_frames > 0:
                            successful_interpolations += 1
                        
                        # Find the maximum row index for this directory
                        dir_rows = working_df[working_df['landmarks_file_path'] == dir_path].index
                        if len(dir_rows) > 0:
                            last_processed_row = max(last_processed_row, max(dir_rows) + start_from_row)
                        

                        
                    except Exception as e:
                        print(f"Error processing directory {dir_path}: {str(e)}")
                        failed_interpolations += 1
                        sys.exit(1)
            
            # Update checkpoint after each batch
            update_checkpoint()
            if missing_statistics_files and report_dir:
                missing_files_path = os.path.join(report_dir, )
                try:
                    with open(missing_files_path, 'w') as f:
                        json.dump({"missing_files": missing_statistics_files}, f, indent=2)
                except Exception as e:
                    print(f"Error saving list of missing files: {e}")
            
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if checkpoint_file:
            report_data["status"] = "crashed"
            report_data["error"] = str(e)
            update_checkpoint()
        sys.exit(1)
    
    # Calculate final statistics
    total_elapsed = time.time() - overall_start_time

    
    print(f"\nProcessing completed in {total_elapsed/60:.1f} minutes")
    print(f"Total directories processed: {processed_dirs}/{total_dirs}")
    print(f"Total frames interpolated: {total_interpolated}")
    print(f"Successful directories: {successful_interpolations}")
    print(f"Failed directories: {failed_interpolations}")
    

    

    
    # Final checkpoint update
    if checkpoint_file:
        report_data["status"] = "completed"
        report_data["end_time"] = datetime.now().isoformat()
        report_data["duration_seconds"] = total_elapsed
        report_data["processed_directories"] = processed_dirs
        report_data["successful_interpolations"] = successful_interpolations
        report_data["failed_interpolations"] = failed_interpolations
        report_data["total_interpolated_frames"] = total_interpolated
        update_checkpoint()
    
    return dataframe


def find_latest_checkpoint(report_dir):
    """
    Find the latest checkpoint file in the report directory.
    
    Args:
        report_dir (str): Directory containing checkpoint files
        
    Returns:
        tuple: (latest_file_path, checkpoint_data) or (None, None) if not found
    """
    if not os.path.exists(report_dir):
        return None, None
    
    checkpoint_files = [f for f in os.listdir(report_dir) 
                        if f.startswith("landmarks_processing_") and f.endswith(".json")]
    
    if not checkpoint_files:
        return None, None
    
    # Sort by modification time (most recent first)
    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(report_dir, f)))
    latest_path = os.path.join(report_dir, latest_file)
    
    try:
        with open(latest_path, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data["last_processed_row"], checkpoint_data
    except:
        print("Error loading")
        return 
    


