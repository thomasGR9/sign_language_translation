#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import numpy as np
import os
import random
from typing import Dict, List, Tuple, Any
import json
import glob
import sys
import pandas as pd
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
import signal
import traceback
import sys

def load_frame_data_standardized(npz_path):
    """
    Load saved frame data from an NPZ file.
    
    Args:
        npz_path (str): Path to the saved .npz file
        
    Returns:
        tuple: All the detection results for the frame
    """
    data = np.load(npz_path)
    
    # Extract all arrays from the npz file
    dom_landmarks_standardized = data['dom_landmarks_standardized']
    non_dom_landmarks_standardized = data['non_dom_landmarks_standardized']
    confidence_scores = data['confidence_scores']
    interpolation_scores = data['interpolation_scores']
    detection_status = data['detection_status']
    blendshape_scores_standardized = data['blendshape_scores_standardized']
    face_detected = data['face_detected'].item()  # Convert 0-d array to scalar
    nose_to_wrist_dist_standardized = data['nose_to_wrist_dist_standardized']
    frame_idx = data['frame_idx'].item()
    timestamp_ms = data['timestamp_ms'].item()
    dom_velocity_small_standardized = data['dom_velocity_small_standardized']
    dom_velocity_large_standardized = data['dom_velocity_large_standardized']
    non_dom_velocity_small_standardized = data['non_dom_velocity_small_standardized']
    non_dom_velocity_large_standardized = data['non_dom_velocity_large_standardized']
    velocity_confidence = data['velocity_confidence']
    velocity_calculation_confidence = data['velocity_calculation_confidence']
    nose_to_wrist_velocity_small_standardized = data['wrist_velocity_small_standardized']
    nose_to_wrist_velocity_large_standardized = data['wrist_velocity_large_standardized']
    
    return (dom_landmarks_standardized, non_dom_landmarks_standardized, confidence_scores, interpolation_scores,
            detection_status, blendshape_scores_standardized, face_detected, 
            nose_to_wrist_dist_standardized, frame_idx, timestamp_ms, dom_velocity_small_standardized, dom_velocity_large_standardized, non_dom_velocity_small_standardized, non_dom_velocity_large_standardized, velocity_confidence, velocity_calculation_confidence, nose_to_wrist_velocity_small_standardized, nose_to_wrist_velocity_large_standardized)

def sorted_npz_files_checked_label(directory_path):
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
    
    if statistics_file['video_info']['total_frames'] != (len(npz_files)-1):
        print("npz filepath list contain different amount of items than total frames")
        sys.exit(1)


    frame_to_file = {}
    for file_path in npz_files:
        if os.path.basename(file_path) == 'smooth_labels.npz':
            label_path = file_path
            continue
        try:
            frame_data = load_frame_data_standardized(file_path)
        except Exception as e:
            print(f"Error loading frame with path: {file_path}: {e}")
            sys.exit(1)
            
        frame_idx = frame_data[8]  # Index for frame_idx
        frame_to_file[frame_idx] = file_path

    
    frame_indices = sorted(frame_to_file.keys())
    if not all(frame_indices[i+1] - frame_indices[i] == 1 for i in range(len(frame_indices) - 1)):
        print("Consecutive frames are not different by one frame")
        sys.exit(1)

    

    return frame_to_file, frame_indices, label_path

def load_label(label_path):
    label_data = np.load(label_path)
    L_index = label_data['L_index']
    L_values = label_data['L_values']
    return L_index, L_values




class ASLFrameDataset(Dataset):
    """Dataset for ASL frame data from video clips with feature extraction."""
    def __init__(self, dataframe):
        """
        Initialize the dataset.
        
        Args:
            dataframe: Pandas DataFrame containing 'landmarks_file_path' column
        """
        self.dataframe = dataframe
        self.video_paths = list(dataframe['landmarks_file_path'])
        
    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get data for a complete video with all features."""
        directory_path = self.video_paths[idx]
        
        # Get paths to all frame files in this video
        frame_to_file, frame_indices, label_path = sorted_npz_files_checked_label(directory_path)
        
        # Initialize dictionaries to store all data
        all_data = {
            # Primary features for model input
            'dom_landmarks': [],
            'non_dom_landmarks': [],
            'blendshape_scores': [],
            'nose_to_wrist_dist': [],
            'dom_velocity_small': [],
            'dom_velocity_large': [],
            'non_dom_velocity_small': [],
            'non_dom_velocity_large': [],
            'nose_to_wrist_velocity_small': [],
            'nose_to_wrist_velocity_large': [],
            
            # Additional data for later use
            'confidence_scores': [],
            'interpolation_scores': [],
            'detection_status': [],
            'face_detected': [],
            'frame_idx': [],
            'velocity_confidence': [],
            'velocity_calculation_confidence': []
        }
        
        # Load data from each frame
        for frame_idx in frame_indices:
            file_path = frame_to_file[frame_idx]
            frame_data = load_frame_data_standardized(file_path)
            
            # Unpack frame data
            (dom_landmarks_standardized,
             non_dom_landmarks_standardized,
             confidence_scores,
             interpolation_scores,
             detection_status,
             blendshape_scores_standardized,
             face_detected,
             nose_to_wrist_dist_standardized,
             frame_idx_val,
             timestamp_ms,  # We'll skip this one
             dom_velocity_small_standardized,
             dom_velocity_large_standardized,
             non_dom_velocity_small_standardized,
             non_dom_velocity_large_standardized,
             velocity_confidence,
             velocity_calculation_confidence,
             nose_to_wrist_velocity_small_standardized,
             nose_to_wrist_velocity_large_standardized) = frame_data
            
            # Store primary features for model input
            all_data['dom_landmarks'].append(dom_landmarks_standardized)
            all_data['non_dom_landmarks'].append(non_dom_landmarks_standardized)
            all_data['blendshape_scores'].append(blendshape_scores_standardized)
            all_data['nose_to_wrist_dist'].append(nose_to_wrist_dist_standardized)
            all_data['dom_velocity_small'].append(dom_velocity_small_standardized)
            all_data['dom_velocity_large'].append(dom_velocity_large_standardized)
            all_data['non_dom_velocity_small'].append(non_dom_velocity_small_standardized)
            all_data['non_dom_velocity_large'].append(non_dom_velocity_large_standardized)
            all_data['nose_to_wrist_velocity_small'].append(nose_to_wrist_velocity_small_standardized)
            all_data['nose_to_wrist_velocity_large'].append(nose_to_wrist_velocity_large_standardized)
            
            # Store additional data for later use
            all_data['confidence_scores'].append(confidence_scores)
            all_data['interpolation_scores'].append(interpolation_scores)
            all_data['detection_status'].append(detection_status)
            all_data['face_detected'].append(face_detected)
            all_data['frame_idx'].append(frame_idx_val)
            all_data['velocity_confidence'].append(velocity_confidence)
            all_data['velocity_calculation_confidence'].append(velocity_calculation_confidence)
        
        # Convert lists to numpy arrays
        for key in all_data:
            all_data[key] = np.array(all_data[key])
        
        # Load label data
        L_index, L_values = load_label(label_path)
        all_data['L_index'] = L_index
        all_data['L_values'] = L_values
        
        # Store sequence length and directory path
        all_data['seq_length'] = len(frame_indices)
        all_data['directory_path'] = directory_path
        
        return all_data


class SingleDataFrameBatchSampler(Sampler):
    """
    Custom batch sampler that ensures each batch contains samples 
    from only one dataframe.
    """
    def __init__(self, dataset_sizes: List[int], batch_size: int, drop_last: bool = False):
        """
        Initialize the batch sampler.
        
        Args:
            dataset_sizes: List of sizes for each dataset
            batch_size: Batch size
            drop_last: Whether to drop the last batch if incomplete
        """
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Calculate offsets for indexing into the combined dataset
        self.offsets = [0]
        for size in dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)
    
    def __iter__(self):
        """Generate batches of indices, ensuring each batch comes from one dataset."""
        # Create index lists for each dataset
        all_indices = []
        for dataset_idx, size in enumerate(self.dataset_sizes):
            offset = self.offsets[dataset_idx]
            indices = list(range(offset, offset + size))
            random.shuffle(indices)
            all_indices.append(indices)
            
        # Create batches for each dataset
        all_batches = []
        for dataset_idx, indices in enumerate(all_indices):
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:min(i + self.batch_size, len(indices))]
                
                # Skip last incomplete batch if drop_last is True
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                all_batches.append(batch)
        
        # Shuffle the order of batches
        random.shuffle(all_batches)
        
        # Yield batches one at a time
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return sum(size // self.batch_size for size in self.dataset_sizes)
        else:
            return sum((size + self.batch_size - 1) // self.batch_size for size in self.dataset_sizes)

def collate_with_dynamic_padding(batch, device='cuda'):
    """
    Custom collate function that handles variable-length sequences and label data.
    """
    # Find the maximum sequence length in this batch
    max_seq_length = max(sample['seq_length'] for sample in batch)
    batch_size = len(batch)
    
    # Initialize the result dictionary
    result = {
        'directory_paths': [],
        'seq_lengths': []
    }
    
    # Store directory paths and sequence lengths
    for sample in batch:
        result['directory_paths'].append(sample['directory_path'])
        result['seq_lengths'].append(sample['seq_length'])
    
    result['seq_lengths'] = torch.tensor(result['seq_lengths'], dtype=torch.long, device=device)
    
    # Create mask tensor for frames [batch_size, max_seq_length]
    frame_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=device)
    
    # Handle variable-sized label data
    # Find maximum dimensions for L_index and L_values
    max_tokens = max(sample['L_index'].shape[0] for sample in batch)
    token_width = batch[0]['L_index'].shape[1]  # Assuming all have same width (6)
    
    # Create padded tensors for labels - using appropriate dtypes
    L_index_padded = torch.zeros((batch_size, max_tokens, token_width), dtype=torch.long, device=device)
    L_values_padded = torch.zeros((batch_size, max_tokens, token_width), dtype=torch.float32, device=device)
    label_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    
    # Fill in label data
    for i, sample in enumerate(batch):
        num_tokens = sample['L_index'].shape[0]
        L_index_padded[i, :num_tokens] = torch.tensor(sample['L_index'], dtype=torch.long, device=device)
        L_values_padded[i, :num_tokens] = torch.tensor(sample['L_values'], dtype=torch.float32, device=device)
        label_mask[i, :num_tokens] = True
    
    result['L_index'] = L_index_padded
    result['L_values'] = L_values_padded
    result['label_mask'] = label_mask
    
    # Process feature data with consistent dimensions
    feature_keys = [
        # Primary features for model input
        'dom_landmarks', 'non_dom_landmarks', 'blendshape_scores',
        'nose_to_wrist_dist', 'dom_velocity_small', 'dom_velocity_large',
        'non_dom_velocity_small', 'non_dom_velocity_large',
        'nose_to_wrist_velocity_small', 'nose_to_wrist_velocity_large',
        
        # Additional data for later use
        'confidence_scores', 'interpolation_scores', 'detection_status',
        'frame_idx', 'velocity_confidence',
        'velocity_calculation_confidence'
    ]
    
    # Process all standard features
    for key in feature_keys:
        try:
            # Get the sample feature
            sample_feature = batch[0][key]
            feature_shape = sample_feature.shape[1:] if len(sample_feature.shape) > 1 else ()
            
            # Create padded tensor [batch_size, max_seq_length, *feature_shape]
            padded_tensor = torch.zeros((batch_size, max_seq_length) + feature_shape, dtype=torch.float32, device=device)
            
            # Fill in the actual data and update the mask
            for i, sample in enumerate(batch):
                seq_length = sample['seq_length']
                feature_data = sample[key]
                padded_tensor[i, :seq_length] = torch.tensor(feature_data, dtype=torch.float32, device=device)
                frame_mask[i, :seq_length] = True
                
            # Add to result
            result[key] = padded_tensor
            
        except Exception as e:
            print(f"Error processing feature '{key}': {e}")
            print(f"  Shape in first sample: {np.array(batch[0][key]).shape}")
            if i > 0:
                print(f"  Shape in problematic sample {i}: {np.array(sample[key]).shape}")
    
    # Process face_detected separately with proper reshaping
    try:
        # Create a tensor specifically for face_detected (which needs special handling)
        face_detected_tensor = torch.zeros((batch_size, max_seq_length), dtype=torch.float32, device=device)
        
        for i, sample in enumerate(batch):
            seq_length = sample['seq_length']
            face_data = sample['face_detected']
            
            # Convert to tensor and ensure it's 1D
            face_tensor = torch.tensor(face_data, dtype=torch.float32, device=device)
            
            # Assign directly without reshaping
            face_detected_tensor[i, :seq_length] = face_tensor
            
        result['face_detected'] = face_detected_tensor
        
    except Exception as e:
        print(f"Error processing face_detected: {e}")
        print(f"  Shape: {np.array(batch[0]['face_detected']).shape}")
    
    # Add the frame mask
    result['mask'] = frame_mask
    
    return result


def create_asl_dataloader(low_df, mid_df, high_df, batch_size=16, num_workers=4, drop_last=False, device='cuda'):
    """
    Create a data loader for ASL data that ensures batches only contain samples from one dataframe.
    
    Args:
        low_df: DataFrame with low frame count videos
        mid_df: DataFrame with medium frame count videos
        high_df: DataFrame with high frame count videos
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last batch if incomplete
        
    Returns:
        A DataLoader that yields batches from the three dataframes
    """
    # Create datasets for each dataframe
    low_dataset = ASLFrameDataset(low_df)
    mid_dataset = ASLFrameDataset(mid_df)
    high_dataset = ASLFrameDataset(high_df)
    
    # Get dataset sizes
    dataset_sizes = [len(low_dataset), len(mid_dataset), len(high_dataset)]
    if drop_last:
        expected_batches = sum(size // batch_size for size in dataset_sizes)
    else:
        expected_batches = sum((size + batch_size - 1) // batch_size for size in dataset_sizes)
    # Combine datasets
    combined_dataset = ConcatDataset([low_dataset, mid_dataset, high_dataset])
    
    # Create a batch sampler that ensures batches only contain samples from one dataframe
    batch_sampler = SingleDataFrameBatchSampler(dataset_sizes, batch_size, drop_last)
    
    # Create the data loader
    data_loader = DataLoader(
        combined_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda b: collate_with_dynamic_padding(b, device=device)
    )
    
    return data_loader, expected_batches



class LandmarkEmbedding(nn.Module):
    """
    Creates learnable embeddings for hand landmarks.
    
    This module maps each landmark (across both hands) to a unique 
    embedding vector that encodes its semantic meaning.
    """
    def __init__(self, embedding_dim, num_landmarks_per_hand=21):
        """
        Initialize the landmark embedding module.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            num_landmarks_per_hand: Number of landmarks per hand (default: 21)
        """
        super(LandmarkEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_landmarks_per_hand = num_landmarks_per_hand
        self.total_landmarks = 2 * num_landmarks_per_hand  # Both hands
        
        # Create the embedding table: [total_landmarks, embedding_dim]
        self.embedding_table = nn.Embedding(
            num_embeddings=self.total_landmarks,
            embedding_dim=embedding_dim
        )
        
        # Initialize the embeddings with a normal distribution
        nn.init.uniform_(self.embedding_table.weight, a=-0.05, b=0.05)
    
    def forward(self, landmark_indices=None):
        """
        Get embeddings for landmarks.
        
        Args:
            landmark_indices: Optional tensor of landmark indices to retrieve.
                             If None, returns all landmark embeddings.
        
        Returns:
            Tensor of landmark embeddings
        """
        if landmark_indices is None:
            # Return all landmark embeddings
            # Create indices for all landmarks: 0 to total_landmarks-1
            landmark_indices = torch.arange(self.total_landmarks, device=self.embedding_table.weight.device)
        
        # Get the embeddings for the specified indices
        embeddings = self.embedding_table(landmark_indices)
        return embeddings
    

class LandmarkSpatialEncoder(nn.Module):
    """
    Encodes the spatial information (x,y,z coordinates) of individual hand landmarks.
    
    This module transforms the 3D coordinates of each landmark into a higher-dimensional
    representation that captures the 'where' aspect of the landmark.
    """
    def __init__(self, 
                 embedding_dim, 
                 hidden_dims=None, 
                 num_layers=2,
                 activation='relu',
                 init_method='kaiming_normal',
                 init_gain=1.0,
                 init_nonlinearity='relu'):
        """
        Initialize the spatial encoder with customizable architecture.
        
        Args:
            embedding_dim: Base dimension for the model
            hidden_dims: List of hidden layer dimensions. If None, uses [4*embedding_dim] * num_layers
            num_layers: Number of hidden layers (default: 2)
            activation: Activation function to use ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', etc.)
            init_method: Weight initialization method ('kaiming_normal', 'kaiming_uniform', 
                        'xavier_normal', 'xavier_uniform', 'normal', 'uniform')
            init_gain: Gain parameter for certain initialization methods
            init_nonlinearity: Nonlinearity parameter for certain initialization methods
        """
        super(LandmarkSpatialEncoder, self).__init__()
        
        # The output dimension will be 2*embedding_dim as requested
        self.output_dim = 2 * embedding_dim
        
        # If hidden_dims not provided, create default configuration
        if hidden_dims is None:
            hidden_dims = [4 * embedding_dim] * num_layers
        
        # Get the activation function
        self.activation_fn = self._get_activation(activation)
        
        # Create layers list starting with input layer
        layers = []
        
        # Input layer
        layers.append(nn.Linear(3, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-5))
        layers.append(self.activation_fn)
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i], eps=1e-5))
            layers.append(self.activation_fn)
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Create the feed-forward network
        self.spatial_encoder = nn.Sequential(*layers)
        
        # Initialize weights using the specified method
        self._init_weights(init_method, init_gain, init_nonlinearity)
    
    def _get_activation(self, activation_name):
        """Get the activation function based on name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Activation function '{activation_name}' not supported. "
                           f"Choose from: {', '.join(activations.keys())}")
        
        return activations[activation_name.lower()]
    
    def _init_weights(self, init_method, gain, nonlinearity):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
                
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Initialization method '{init_method}' not supported.")
            
            # Initialize bias if it exists
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, landmarks):
        """
        Encode the spatial coordinates of landmarks.
        
        Args:
            landmarks: Tensor of shape [..., 3] containing x,y,z coordinates
                       The leading dimensions can be anything (batch, sequence, landmark)
        
        Returns:
            Tensor of shape [..., output_dim] with the spatial encodings
        """
        # Get the original shape to reshape the output later
        original_shape = landmarks.shape
        
        # Reshape to [-1, 3] to process all landmarks in parallel
        flat_landmarks = landmarks.reshape(-1, 3)
        
        # Apply the spatial encoder
        encoded = self.spatial_encoder(flat_landmarks)
        
        # Reshape back to original dimensions but with output_dim as the last dimension
        reshaped_encoded = encoded.reshape(*original_shape[:-1], self.output_dim)
        
        return reshaped_encoded
    
def combine_spatial_and_semantic_features(spatial_features, semantic_features):
    """
    Combines the spatial encoder output with the semantic embedding features.
    
    This function concatenates the "where" (spatial) information with the "what" 
    (semantic) information to create a comprehensive landmark representation.
    
    Args:
        spatial_features: Tensor of shape [..., n_spatial_encode] where
                         n_spatial_encode = 2*embedding_dim
        semantic_features: Tensor of shape [..., embedding_dim]
    
    Returns:
        Tensor of shape [..., 3*embedding_dim] containing the combined representation
    """

    batch_dims = spatial_features.shape[:-2]
    expanded_embeddings = semantic_features.expand(*batch_dims, -1, -1)
    # Verify that the batch dimensions match
    assert spatial_features.shape[:-1] == expanded_embeddings.shape[:-1], \
        "Batch dimensions of spatial and semantic features must match"
    
    # Concatenate along the last dimension
    combined_features = torch.cat([expanded_embeddings, spatial_features], dim=-1)
    
    return combined_features

class WristSpatialEncoder(nn.Module):
    """
    Encodes the spatial information of wrist landmarks relative to the nose.
    
    This module processes the 2D coordinates (x,y) of each wrist independently
    but in parallel, using shared weights across both wrists.
    """
    def __init__(self, 
                 embedding_dim, 
                 hidden_dims=None, 
                 num_layers=2,
                 activation='relu',
                 init_method='kaiming_normal',
                 init_gain=1.0,
                 init_nonlinearity='relu'):
        """
        Initialize the wrist spatial encoder with customizable architecture.
        
        Args:
            embedding_dim: Base dimension for the model
            hidden_dims: List of hidden layer dimensions. If None, uses [4*embedding_dim] * num_layers
            num_layers: Number of hidden layers (default: 2)
            activation: Activation function to use ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', etc.)
            init_method: Weight initialization method ('kaiming_normal', 'kaiming_uniform', 
                        'xavier_normal', 'xavier_uniform', 'normal', 'uniform')
            init_gain: Gain parameter for certain initialization methods
            init_nonlinearity: Nonlinearity parameter for certain initialization methods
        """
        super(WristSpatialEncoder, self).__init__()
        
        # The output dimension will be 2*embedding_dim as requested
        self.output_dim = 2 * embedding_dim
        
        # If hidden_dims not provided, create default configuration
        if hidden_dims is None:
            hidden_dims = [4 * embedding_dim] * num_layers
        
        # Get the activation function
        self.activation_fn = self._get_activation(activation)
        
        # Create layers list starting with input layer
        layers = []
        
        # Input layer (2D coordinates instead of 3D)
        layers.append(nn.Linear(2, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-5))
        layers.append(self.activation_fn)
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i], eps=1e-5))
            layers.append(self.activation_fn)
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Create the feed-forward network
        self.wrist_encoder = nn.Sequential(*layers)
        
        # Initialize weights using the specified method
        self._init_weights(init_method, init_gain, init_nonlinearity)
    
    def _get_activation(self, activation_name):
        """Get the activation function based on name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Activation function '{activation_name}' not supported. "
                           f"Choose from: {', '.join(activations.keys())}")
        
        return activations[activation_name.lower()]
    
    def _init_weights(self, init_method, gain, nonlinearity):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
                
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Initialization method '{init_method}' not supported.")
            
            # Initialize bias if it exists
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, wrist_coordinates):
        """
        Encode the spatial coordinates of wrist landmarks.
        
        Args:
            wrist_coordinates: Tensor of shape [..., 2, 2] containing x,y coordinates
                              for both wrists. Leading dimensions can be anything
                              (batch, sequence), and the last two dimensions are:
                              - Dimension -2: Wrist index (0=dominant, 1=non-dominant)
                              - Dimension -1: Coordinates (x,y)
        
        Returns:
            Tensor of shape [..., 2, output_dim] with the spatial encodings for each wrist
        """
        # Get the original shape to reshape the output later
        original_shape = wrist_coordinates.shape
        
        # Reshape to [-1, 2] to process all wrist coordinates in parallel
        # This flattens all leading dimensions and processes each (x,y) pair independently
        flat_wrists = wrist_coordinates.reshape(-1, 2)
        
        # Apply the wrist encoder
        encoded = self.wrist_encoder(flat_wrists)
        
        # Reshape back to original dimensions but with output_dim as the last dimension
        # Replace the coordinate dimension (2) with output_dim
        new_shape = original_shape[:-1] + (self.output_dim,)
        reshaped_encoded = encoded.reshape(new_shape)
        
        return reshaped_encoded
    
def combine_wrist_embedding_and_spatial(wrist_embeddings, wrist_spatial_features):
    """
    Combines wrist semantic embeddings with their spatial features.
    
    This function integrates:
    1. The semantic meaning of each wrist (from embeddings)
    2. The spatial position of each wrist (from the WristSpatialEncoder)
    
    Args:
        wrist_embeddings: Tensor of shape [2, embedding_dim] with wrist embeddings
                         where [0] is dom wrist and [1] is non-dom wrist
        wrist_spatial_features: Tensor of shape [..., 2, 2*embedding_dim] 
                               from the WristSpatialEncoder
    
    Returns:
        Tensor of shape [..., 2, 3*embedding_dim] with the combined representation
    """
    # Get the batch dimensions from the spatial features tensor
    batch_dims = wrist_spatial_features.shape[:-2]
    
    # Expand wrist embeddings to match the batch dimensions
    # From [2, embedding_dim] to [..., 2, embedding_dim]
    expanded_embeddings = wrist_embeddings.expand(*batch_dims, -1, -1)
    
    # Verify that the shapes are compatible for concatenation
    assert expanded_embeddings.shape[:-1] == wrist_spatial_features.shape[:-1], \
        "Batch dimensions of embeddings and spatial features must match"
    
    # Concatenate along the last dimension
    combined_features = torch.cat([
        expanded_embeddings,     # Wrist identity (what)
        wrist_spatial_features   # Wrist position (where)
    ], dim=-1)
    
    return combined_features

class BlendshapeEncoder(nn.Module):
    """
    Encodes facial blendshape scores into a higher-dimensional representation.
    
    This network processes the 52 facial blendshape parameters that capture
    expressions and face movements relevant to ASL interpretation.
    """
    def __init__(self, 
                 embedding_dim, 
                 hidden_dims=None, 
                 num_layers=2,
                 activation='relu',
                 init_method='kaiming_normal',
                 init_gain=1.0,
                 init_nonlinearity='relu'):
        """
        Initialize the blendshape encoder with customizable architecture.
        
        Args:
            embedding_dim: Base dimension for the model
            hidden_dims: List of hidden layer dimensions. If None, uses [4*embedding_dim] * num_layers
            num_layers: Number of hidden layers (default: 2)
            activation: Activation function to use ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', etc.)
            init_method: Weight initialization method ('kaiming_normal', 'kaiming_uniform', 
                        'xavier_normal', 'xavier_uniform', 'normal', 'uniform')
            init_gain: Gain parameter for certain initialization methods
            init_nonlinearity: Nonlinearity parameter for certain initialization methods
        """
        super(BlendshapeEncoder, self).__init__()
        
        # The output dimension will be 2*embedding_dim as requested
        self.output_dim = 2 * embedding_dim
        
        # Input dimension for blendshape scores
        self.input_dim = 52
        
        # If hidden_dims not provided, create default configuration
        if hidden_dims is None:
            hidden_dims = [4 * embedding_dim] * num_layers
        
        # Get the activation function
        self.activation_fn = self._get_activation(activation)
        
        # Create layers list starting with input layer
        layers = []
        
        # Input layer (52 blendshape scores)
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-5))
        layers.append(self.activation_fn)
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i], eps=1e-5))
            layers.append(self.activation_fn)
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Create the feed-forward network
        self.blendshape_encoder = nn.Sequential(*layers)
        
        # Initialize weights using the specified method
        self._init_weights(init_method, init_gain, init_nonlinearity)
    
    def _get_activation(self, activation_name):
        """Get the activation function based on name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Activation function '{activation_name}' not supported. "
                           f"Choose from: {', '.join(activations.keys())}")
        
        return activations[activation_name.lower()]
    
    def _init_weights(self, init_method, gain, nonlinearity):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
                
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Initialization method '{init_method}' not supported.")
            
            # Initialize bias if it exists
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, blendshape_scores):
        """
        Encode the facial blendshape scores.
        
        Args:
            blendshape_scores: Tensor of shape [..., 52] containing facial expression parameters.
                              Leading dimensions can be anything (batch, sequence).
        
        Returns:
            Tensor of shape [..., output_dim] with the encoded facial features
        """
        # Get the original shape to reshape the output later
        original_shape = blendshape_scores.shape
        
        # Reshape to [-1, 52] to process all blendshape scores in parallel
        flat_blendshapes = blendshape_scores.reshape(-1, self.input_dim)
        
        # Apply the blendshape encoder
        encoded = self.blendshape_encoder(flat_blendshapes)
        
        # Reshape back to original dimensions but with output_dim as the last dimension
        # Replace the blendshape dimension (52) with output_dim
        new_shape = original_shape[:-1] + (self.output_dim,)
        reshaped_encoded = encoded.reshape(new_shape)
        
        return reshaped_encoded
    
class VelocityEncoder(nn.Module):
    """
    Encodes velocity features of hand landmarks into a higher-dimensional representation.
    
    This network processes the 5 spherical coordinate velocity features for each landmark
    independently but in parallel, using the same weights across all landmarks, hands,
    and velocity windows (small and large).
    """
    def __init__(self, 
                 n_velocity_encoding, 
                 hidden_dims=None, 
                 num_layers=2,
                 activation='relu',
                 init_method='kaiming_normal',
                 init_gain=1.0,
                 init_nonlinearity='relu'):
        """
        Initialize the velocity encoder with customizable architecture.
        
        Args:
            n_velocity_encoding: Output dimension for each landmark's velocity encoding
            hidden_dims: List of hidden layer dimensions. If None, uses [4*n_velocity_encoding] * num_layers
            num_layers: Number of hidden layers (default: 2)
            activation: Activation function to use ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', etc.)
            init_method: Weight initialization method ('kaiming_normal', 'kaiming_uniform', 
                        'xavier_normal', 'xavier_uniform', 'normal', 'uniform')
            init_gain: Gain parameter for certain initialization methods
            init_nonlinearity: Nonlinearity parameter for certain initialization methods
        """
        super(VelocityEncoder, self).__init__()
        
        # The output dimension as specified
        self.output_dim = n_velocity_encoding
        
        # Input dimension for velocity features (spherical coordinates)
        self.input_dim = 5
        
        # If hidden_dims not provided, create default configuration
        if hidden_dims is None:
            hidden_dims = [4 * n_velocity_encoding] * num_layers
        
        # Get the activation function
        self.activation_fn = self._get_activation(activation)
        
        # Create layers list starting with input layer
        layers = []
        
        # Input layer (5 velocity features in spherical coordinates)
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-5))
        layers.append(self.activation_fn)
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i], eps=1e-5))
            layers.append(self.activation_fn)
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Create the feed-forward network
        self.velocity_encoder = nn.Sequential(*layers)
        
        # Initialize weights using the specified method
        self._init_weights(init_method, init_gain, init_nonlinearity)
    
    def _get_activation(self, activation_name):
        """Get the activation function based on name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Activation function '{activation_name}' not supported. "
                           f"Choose from: {', '.join(activations.keys())}")
        
        return activations[activation_name.lower()]
    
    def _init_weights(self, init_method, gain, nonlinearity):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
                
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Initialization method '{init_method}' not supported.")
            
            # Initialize bias if it exists
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, velocity_features):
        """
        Encode the velocity features for hand landmarks.
        
        Args:
            velocity_features: Tensor of shape [..., 5] containing velocity features
                              in spherical coordinates. Leading dimensions can be anything
                              (batch, sequence, landmark).
        
        Returns:
            Tensor of shape [..., output_dim] with the encoded velocity features
        """
        # Get the original shape to reshape the output later
        original_shape = velocity_features.shape
        
        # Reshape to [-1, 5] to process all velocity features in parallel
        flat_velocities = velocity_features.reshape(-1, self.input_dim)
        
        # Apply the velocity encoder
        encoded = self.velocity_encoder(flat_velocities)
        
        # Reshape back to original dimensions but with output_dim as the last dimension
        # Replace the velocity dimension (5) with output_dim
        new_shape = original_shape[:-1] + (self.output_dim,)
        reshaped_encoded = encoded.reshape(new_shape)
        
        return reshaped_encoded
    
    def encode_all_velocity_windows(self, dom_vel_small, dom_vel_large, non_dom_vel_small, non_dom_vel_large):
        """
        Encode all four velocity window tensors using the same encoder.
        
        Args:
            dom_vel_small: Dominant hand small window velocities [batch_size, seq_len, 20, 5]
            dom_vel_large: Dominant hand large window velocities [batch_size, seq_len, 20, 5]
            non_dom_vel_small: Non-dominant hand small window velocities [batch_size, seq_len, 20, 5]
            non_dom_vel_large: Non-dominant hand large window velocities [batch_size, seq_len, 20, 5]
            
        Returns:
            Dictionary containing encoded velocity features for all windows
        """
        # Process each velocity window
        dom_small_encoded = self.forward(dom_vel_small)  # [batch_size, seq_len, 20, output_dim]
        dom_large_encoded = self.forward(dom_vel_large)  # [batch_size, seq_len, 20, output_dim]
        non_dom_small_encoded = self.forward(non_dom_vel_small)  # [batch_size, seq_len, 20, output_dim]
        non_dom_large_encoded = self.forward(non_dom_vel_large)  # [batch_size, seq_len, 20, output_dim]
        
        return {
            'dom_velocity_small_encoded': dom_small_encoded,
            'dom_velocity_large_encoded': dom_large_encoded,
            'non_dom_velocity_small_encoded': non_dom_small_encoded,
            'non_dom_velocity_large_encoded': non_dom_large_encoded
        }
    

def combine_semantic_and_velocity_features(semantic_features, velocity_small_features, velocity_large_features):
    """
    Combines landmark semantic embeddings with velocity features from both time windows.
    
    This function concatenates:
    1. The "what" (semantic embedding) of each landmark
    2. The "how fast small window" (small window velocity encoding)
    3. The "how fast large window" (large window velocity encoding)
    
    Args:
        semantic_features: Tensor of shape [..., embedding_dim] containing landmark embeddings
        velocity_small_features: Tensor of shape [..., n_velocity_encoding] from small window
        velocity_large_features: Tensor of shape [..., n_velocity_encoding] from large window
    
    Returns:
        Tensor of shape [..., embedding_dim + 2*n_velocity_encoding] with the combined representation
    """
    batch_shape = velocity_small_features.shape[:-2]
    semantic_features_expanded = semantic_features.expand(*batch_shape, -1, -1)
    # Verify that the batch dimensions match
    assert semantic_features_expanded.shape[:-1] == velocity_small_features.shape[:-1] == velocity_large_features.shape[:-1], \
        "Batch dimensions of semantic and velocity features must match"
    
    # Concatenate all three feature types along the last dimension
    combined_features = torch.cat([
        semantic_features_expanded,        # Landmark identity (what)
        velocity_small_features,  # Short-term movement (how fast recently)
        velocity_large_features   # Long-term movement (how fast overall)
    ], dim=-1)
    
    return combined_features


class WristVelocityEncoder(nn.Module):
    """
    Encodes velocity features of wrist landmarks relative to the nose.
    
    This network processes the 3 polar coordinate velocity features for each wrist
    independently but in parallel, using the same weights across both wrists
    and both velocity windows (small and large).
    """
    def __init__(self, 
                 n_velocity_encoding, 
                 hidden_dims=None, 
                 num_layers=2,
                 activation='relu',
                 init_method='kaiming_normal',
                 init_gain=1.0,
                 init_nonlinearity='relu'):
        """
        Initialize the wrist velocity encoder with customizable architecture.
        
        Args:
            n_velocity_encoding: Output dimension for each wrist's velocity encoding
            hidden_dims: List of hidden layer dimensions. If None, uses [4*n_velocity_encoding] * num_layers
            num_layers: Number of hidden layers (default: 2)
            activation: Activation function to use ('relu', 'leaky_relu', 'gelu', 'silu', 'tanh', etc.)
            init_method: Weight initialization method ('kaiming_normal', 'kaiming_uniform', 
                        'xavier_normal', 'xavier_uniform', 'normal', 'uniform')
            init_gain: Gain parameter for certain initialization methods
            init_nonlinearity: Nonlinearity parameter for certain initialization methods
        """
        super(WristVelocityEncoder, self).__init__()
        
        # The output dimension as specified
        self.output_dim = n_velocity_encoding
        
        # Input dimension for wrist velocity features (polar coordinates)
        self.input_dim = 3
        
        # If hidden_dims not provided, create default configuration
        if hidden_dims is None:
            hidden_dims = [4 * n_velocity_encoding] * num_layers
        
        # Get the activation function
        self.activation_fn = self._get_activation(activation)
        
        # Create layers list starting with input layer
        layers = []
        
        # Input layer (3 velocity features in polar coordinates)
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0], eps=1e-5))
        layers.append(self.activation_fn)
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LayerNorm(hidden_dims[i], eps=1e-5))
            layers.append(self.activation_fn)
        
        # Add output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Create the feed-forward network
        self.wrist_velocity_encoder = nn.Sequential(*layers)
        
        # Initialize weights using the specified method
        self._init_weights(init_method, init_gain, init_nonlinearity)
    
    def _get_activation(self, activation_name):
        """Get the activation function based on name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Activation function '{activation_name}' not supported. "
                           f"Choose from: {', '.join(activations.keys())}")
        
        return activations[activation_name.lower()]
    
    def _init_weights(self, init_method, gain, nonlinearity):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
                
            if init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            else:
                raise ValueError(f"Initialization method '{init_method}' not supported.")
            
            # Initialize bias if it exists
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, wrist_velocity_features):
        """
        Encode the velocity features for wrist landmarks.
        
        Args:
            wrist_velocity_features: Tensor of shape [..., 3] containing velocity features
                                    in polar coordinates. Leading dimensions can be anything
                                    (batch, sequence, wrist).
        
        Returns:
            Tensor of shape [..., output_dim] with the encoded velocity features
        """
        # Get the original shape to reshape the output later
        original_shape = wrist_velocity_features.shape
        
        # Reshape to [-1, 3] to process all velocity features in parallel
        flat_velocities = wrist_velocity_features.reshape(-1, self.input_dim)
        
        # Apply the wrist velocity encoder
        encoded = self.wrist_velocity_encoder(flat_velocities)
        
        # Reshape back to original dimensions but with output_dim as the last dimension
        # Replace the velocity dimension (3) with output_dim
        new_shape = original_shape[:-1] + (self.output_dim,)
        reshaped_encoded = encoded.reshape(new_shape)
        
        return reshaped_encoded
    
    def encode_both_velocity_windows(self, wrist_vel_small, wrist_vel_large):
        """
        Encode both velocity window tensors for wrists using the same encoder.
        
        Args:
            wrist_vel_small: Wrist small window velocities [batch_size, seq_len, 2, 3]
            wrist_vel_large: Wrist large window velocities [batch_size, seq_len, 2, 3]
            
        Returns:
            Dictionary containing encoded velocity features for both windows
        """
        # Process each velocity window
        small_window_encoded = self.forward(wrist_vel_small)  # [batch_size, seq_len, 2, output_dim]
        large_window_encoded = self.forward(wrist_vel_large)  # [batch_size, seq_len, 2, output_dim]
        
        return {
            'wrist_velocity_small_encoded': small_window_encoded,
            'wrist_velocity_large_encoded': large_window_encoded
        }
    

def combine_wrist_embedding_and_velocity(wrist_embeddings, wrist_velocity_small, wrist_velocity_large):
    """
    Combines wrist semantic embeddings with velocity features from both time windows.
    
    This function handles the specific arrangement of wrist data in your model:
    - In embeddings: Wrists are at indices 20 (dom) and 41 (non-dom) in the embedding table
    - In velocity tensors: Wrists are at indices 0 (dom) and 1 (non-dom)
    
    Args:
        wrist_embeddings: Tensor of shape [2, embedding_dim] with wrist embeddings
                         where [0] is dom wrist and [1] is non-dom wrist
        wrist_velocity_small: Tensor of shape [..., 2, n_velocity_encoding] 
                             from small window velocity encoder
        wrist_velocity_large: Tensor of shape [..., 2, n_velocity_encoding] 
                             from large window velocity encoder
    
    Returns:
        Tensor of shape [..., 2, embedding_dim + 2*n_velocity_encoding] 
        with the combined representation for both wrists
    """
    # Get the batch dimensions from the velocity tensors
    batch_dims = wrist_velocity_small.shape[:-2]
    
    # Expand wrist embeddings to match the batch dimensions
    # From [2, embedding_dim] to [..., 2, embedding_dim]
    expanded_embeddings = wrist_embeddings.expand(*batch_dims, -1, -1)
    
    # Verify that the shapes are compatible for concatenation
    assert expanded_embeddings.shape[:-1] == wrist_velocity_small.shape[:-1] == wrist_velocity_large.shape[:-1], \
        "Batch dimensions of embeddings and velocity features must match"
    
    # Concatenate along the last dimension
    combined_features = torch.cat([
        expanded_embeddings,      # Wrist identity (what)
        wrist_velocity_small,     # Short-term movement (how fast recently)
        wrist_velocity_large      # Long-term movement (how fast overall)
    ], dim=-1)
    
    return combined_features


class LandmarkTransformerEncoder(nn.Module):
    """
    Transformer encoder for processing hand landmarks and learning contextual relationships.
    
    This module treats the set of landmarks as a sequence and applies self-attention
    to learn the relationships between different parts of the hand.
    """
    def __init__(self, 
                 input_dim, 
                 num_layers=2,
                 num_heads=8,
                 hidden_dim=None,
                 ff_dim=None,
                 prenorm=True,
                 activation='gelu',
                 init_method='xavier_uniform',
                 init_gain=1.0):
        """
        Initialize the landmark transformer encoder.
        
        Args:
            input_dim: Dimension of input features per landmark (3*embedding_dim)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension size (if None, uses input_dim)
            ff_dim: Feed-forward dimension (if None, uses 4*hidden_dim)
            prenorm: Whether to use pre-norm (True) or post-norm (False) architecture
            activation: Activation function in feed-forward network
            init_method: Weight initialization method
            init_gain: Gain parameter for initialization
        """
        super(LandmarkTransformerEncoder, self).__init__()
        
        # Set dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * self.hidden_dim
        
        # Input projection if needed
        self.input_projection = None
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Create transformer encoder layers
        encoder_layer = LandmarkTransformerLayer(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            ff_dim=self.ff_dim,
            prenorm=prenorm,
            activation=activation
        )
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
        # Final normalization
        self.norm = nn.LayerNorm(self.hidden_dim, eps=1e-5)
        
        # Initialize weights
        self._init_weights(init_method, init_gain)
    
    def _init_weights(self, init_method, gain):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in')
                elif init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                else:
                    raise ValueError(f"Initialization method '{init_method}' not supported.")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Process hand landmarks through the transformer.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, 20, input_dim]
               where 20 is the number of landmarks and input_dim is 3*embedding_dim
        
        Returns:
            Tensor of shape [batch_size, seq_len, 20, hidden_dim]
            with contextually enriched landmark representations
        """

        # Get original shape
        batch_size, seq_len, num_landmarks, _ = x.shape
        
        # Reshape to process each frame separately
        # [batch_size * seq_len, 20, input_dim]
        x_reshaped = x.reshape(-1, num_landmarks, self.input_dim)
        
        # Apply input projection if needed
        if self.input_projection is not None:
            x_reshaped = self.input_projection(x_reshaped)
        
        # Process through transformer layers
        for layer in self.layers:
            x_reshaped = layer(x_reshaped)
        
        # Apply final normalization
        x_reshaped = self.norm(x_reshaped)
        
        # Reshape back to original dimensions
        # [batch_size, seq_len, 20, hidden_dim]
        output = x_reshaped.reshape(batch_size, seq_len, num_landmarks, self.hidden_dim)
        
        return output


class LandmarkTransformerLayer(nn.Module):
    """
    Single transformer encoder layer for landmark processing.
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, prenorm=True, activation='gelu'):
        """
        Initialize a transformer encoder layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            prenorm: Whether to use pre-norm (True) or post-norm (False)
            activation: Activation function in feed-forward network
        """
        super(LandmarkTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            self._get_activation(activation),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'silu' or name.lower() == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Activation function '{name}' not supported.")
    
    def forward(self, x):
        """
        Process landmarks through a transformer layer.
        
        Args:
            x: Tensor of shape [batch_size*seq_len, 20, hidden_dim]
               representing landmarks in a single frame
        
        Returns:
            Tensor of same shape with contextualized representations
        """
        # Pre-norm or post-norm architecture

        if self.prenorm:
            # Pre-norm: Apply normalization before attention
            norm_x = self.norm1(x)
            attn_output, _ = self.self_attention(norm_x, norm_x, norm_x)
            x = x + attn_output  # Residual connection
            
            # Feed-forward with normalization
            norm_x = self.norm2(x)
            ff_output = self.ff_network(norm_x)
            x = x + ff_output  # Residual connection
        else:
            # Post-norm: Apply attention then normalization
            attn_output, _ = self.self_attention(x, x, x)
            x = self.norm1(x + attn_output)  # Residual connection and norm
            
            # Feed-forward and normalization
            ff_output = self.ff_network(x)
            x = self.norm2(x + ff_output)  # Residual connection and norm

        return x
    

class LandmarkAttentionPooling(nn.Module):
    """
    Applies attention pooling over landmarks using PyTorch's MultiheadAttention.
    """
    def __init__(self, input_dim, output_dim, 
                init_method='xavier_uniform',
                gain=1.0,
                bias_init='zeros',
                **init_kwargs):
        """
        Initialize the attention pooling module.
        
        Args:
            input_dim: Dimension of input features per landmark
            output_dim: Dimension of the output representation
        """
        super(LandmarkAttentionPooling, self).__init__()
        
        # Using PyTorch's built-in attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=1,  # Single head is sufficient for pooling
            batch_first=True,
            dropout=0.1
        )
        self.output_dim = output_dim
        # Learnable query vector
        query_init = torch.zeros(1, 1, input_dim)
        self.query = nn.Parameter(query_init)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, self.output_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-5)

        self._apply_init(self.query, init_method, gain=gain, **init_kwargs)
        
        # Initialize attention weights
        if hasattr(self.attention, 'in_proj_weight') and self.attention.in_proj_weight is not None:
            self._apply_init(self.attention.in_proj_weight, init_method, gain=gain, **init_kwargs)
        
        if hasattr(self.attention, 'out_proj'):
            self._apply_init(self.attention.out_proj.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize output projection
        self._apply_init(self.output_projection.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize all bias terms
        if hasattr(self.attention, 'in_proj_bias') and self.attention.in_proj_bias is not None:
            self._apply_init(self.attention.in_proj_bias, bias_init, **init_kwargs)
            
        if hasattr(self.attention, 'out_proj') and self.attention.out_proj.bias is not None:
            self._apply_init(self.attention.out_proj.bias, bias_init, **init_kwargs)
        
        if self.output_projection.bias is not None:
            self._apply_init(self.output_projection.bias, bias_init, **init_kwargs)
    
    def _apply_init(self, tensor, init_type, **kwargs):
        """
        Apply the specified initialization to a tensor.

        Args:
            tensor: The tensor to initialize
            init_type: Initialization method name
            **kwargs: Additional parameters for initialization
        """
        # Check tensor dimensions
        if len(tensor.shape) < 2:
            # For 1D tensors (biases, layer norm weights), use simpler initialization
            if init_type in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']:
                # For these methods that require 2D+ tensors, fall back to a simpler method
                if bias_init := kwargs.get('bias_init', 'zeros'):
                    if bias_init == 'zeros':
                        nn.init.zeros_(tensor)
                    elif bias_init == 'ones':
                        nn.init.ones_(tensor)
                    elif bias_init == 'constant':
                        val = kwargs.get('val', 0.0)
                        nn.init.constant_(tensor, val=val)
                    elif bias_init == 'normal':
                        mean = kwargs.get('mean', 0.0)
                        std = kwargs.get('std', 0.01)
                        nn.init.normal_(tensor, mean=mean, std=std)
                    elif bias_init == 'uniform':
                        a = kwargs.get('a', -0.1)
                        b = kwargs.get('b', 0.1)
                        nn.init.uniform_(tensor, a=a, b=b)
                    else:
                        nn.init.zeros_(tensor)  # Default fallback
                else:
                    nn.init.zeros_(tensor)  # Default fallback
            else:
                # For other initializations, proceed with the specified method
                if init_type == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif init_type == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                elif init_type == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif init_type == 'zeros':
                    nn.init.zeros_(tensor)
                elif init_type == 'ones':
                    nn.init.ones_(tensor)
        else:
            # For 2D+ tensors, use the specified initialization method
            if init_type == 'xavier_uniform':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_uniform_(tensor, gain=gain)

            elif init_type == 'xavier_normal':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_normal_(tensor, gain=gain)

            elif init_type == 'kaiming_uniform':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'kaiming_normal':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'normal':
                mean = kwargs.get('mean', 0.0)
                std = kwargs.get('std', 0.01)
                nn.init.normal_(tensor, mean=mean, std=std)

            elif init_type == 'uniform':
                a = kwargs.get('a', -0.1)
                b = kwargs.get('b', 0.1)
                nn.init.uniform_(tensor, a=a, b=b)

            elif init_type == 'constant':
                val = kwargs.get('val', 0.0)
                nn.init.constant_(tensor, val=val)

            elif init_type == 'zeros':
                nn.init.zeros_(tensor)

            elif init_type == 'ones':
                nn.init.ones_(tensor)
    
    def forward(self, x):
        """
        Apply attention pooling over landmarks.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, num_landmarks, input_dim]
        
        Returns:
            Tensor of shape [batch_size, seq_len, output_dim]
        """

        batch_size, seq_len, num_landmarks, input_dim = x.shape
        
        # Reshape to process each sequence element separately
        x_reshaped = x.reshape(batch_size * seq_len, num_landmarks, input_dim)
        
        # Apply layer normalization
        x_norm = self.layer_norm(x_reshaped)
        
        # Expand query to match the batch size
        query = self.query.expand(batch_size * seq_len, -1, -1)
        
        # Apply attention
        # The query attends to all landmarks (keys and values are the same: x_norm)
        pooled, _ = self.attention(query, x_norm, x_norm)
        
        # Remove the sequence dimension (which is 1 for the query)
        pooled = pooled.squeeze(1)  # [batch_size * seq_len, input_dim]
        
        # Project to output dimension
        output = self.output_projection(pooled)  # [batch_size * seq_len, output_dim]
        
        # Reshape back to [batch_size, seq_len, output_dim]
        output = output.reshape(batch_size, seq_len, -1)
        

        return output
    

def concat_pooled_wrists(pooled, wrist):
# Verify that the shapes are compatible for concatenation
    assert pooled.shape[:-1] == wrist.shape[:-1], \
        "Batch dimensions of embeddings and spatial features must match"
    
    # Concatenate along the last dimension
    combined_features = torch.cat([
        pooled,     # Wrist identity (what)
        wrist   # Wrist position (where)
    ], dim=-1)
    
    return combined_features



class ConfidenceWeightedTransformerEncoder(nn.Module):
    """
    Transformer encoder that incorporates confidence scores into attention calculations.
    
    This second-stage transformer learns relationships between the two hands while
    taking into account confidence and interpolation scores from both spatial and
    velocity features.
    """
    def __init__(self, 
                 input_dim, 
                 num_layers=2,
                 num_heads=8,
                 hidden_dim=None,
                 ff_dim=None,
                 prenorm=True,
                 activation='gelu',
                 init_method='xavier_uniform',
                 init_gain=1.0):
        """
        Initialize the confidence-weighted transformer encoder.
        
        Args:
            input_dim: Dimension of input features per hand
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension size (if None, uses input_dim)
            ff_dim: Feed-forward dimension (if None, uses 4*hidden_dim)
            prenorm: Whether to use pre-norm (True) or post-norm (False) architecture
            activation: Activation function in feed-forward network
            init_method: Weight initialization method
            init_gain: Gain parameter for initialization
        """
        super(ConfidenceWeightedTransformerEncoder, self).__init__()
        
        # Set dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * self.hidden_dim
        
        # Input projection if needed
        self.input_projection = None
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Create transformer encoder layers with confidence weighting
        layers = []
        for _ in range(num_layers):
            layers.append(
                ConfidenceWeightedTransformerLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ff_dim=self.ff_dim,
                    prenorm=prenorm,
                    activation=activation
                )
            )
        self.layers = nn.ModuleList(layers)
        
        # Final normalization
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize weights
        self._init_weights(init_method, init_gain)
    
    def _init_weights(self, init_method, gain):
        """Initialize the weights using the specified method."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in')
                elif init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                else:
                    raise ValueError(f"Initialization method '{init_method}' not supported.")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, confidence_scores):
        """
        Process hand features through the transformer with confidence weighting.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, 2, input_dim]
               where 2 represents the dom and non-dom hands
            confidence_scores: Dictionary containing:
                - Cd_spatial: [batch_size, seq_len, 2] confidence scores
                - Ci_spatial: [batch_size, seq_len, 2] interpolation scores
                - Cd_velocity: [batch_size, seq_len, 2] velocity calculation confidence
                - Ci_velocity: [batch_size, seq_len, 2] velocity confidence
        
        Returns:
            Tensor of shape [batch_size, seq_len, 2, hidden_dim]
            with confidence-weighted contextual representations
        """
        # Get original shape
        batch_size, seq_len, num_hands, _ = x.shape
        
        # Reshape to process each frame separately
        # [batch_size * seq_len, 2, input_dim]
        x_reshaped = x.reshape(-1, num_hands, self.input_dim)
        
        # Apply input projection if needed
        if self.input_projection is not None:
            x_reshaped = self.input_projection(x_reshaped)
        
        # Reshape confidence scores for per-frame processing
        conf_scores_reshaped = {}
        for key, tensor in confidence_scores.items():
            conf_scores_reshaped[key] = tensor.reshape(-1, num_hands)
        
        # Process through transformer layers
        for layer in self.layers:
            x_reshaped = layer(x_reshaped, conf_scores_reshaped)
        
        # Apply final normalization
        x_reshaped = self.norm(x_reshaped)
        
        # Reshape back to original dimensions
        # [batch_size, seq_len, 2, hidden_dim]
        output = x_reshaped.reshape(batch_size, seq_len, num_hands, self.hidden_dim)
        
        return output


class ConfidenceWeightedTransformerLayer(nn.Module):
    """
    Transformer encoder layer with confidence-weighted attention.
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, prenorm=True, activation='gelu'):
        """
        Initialize a confidence-weighted transformer encoder layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            prenorm: Whether to use pre-norm (True) or post-norm (False)
            activation: Activation function in feed-forward network
        """
        super(ConfidenceWeightedTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        self.num_heads = num_heads
        
        # Custom attention with confidence weighting
        self.self_attention = ConfidenceWeightedAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            self._get_activation(activation),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'gelu':
            return nn.GELU()
        elif name.lower() == 'silu' or name.lower() == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Activation function '{name}' not supported.")
    
    def forward(self, x, confidence_scores):
        """
        Process through a transformer layer with confidence-weighted attention.
        
        Args:
            x: Tensor of shape [batch_size*seq_len, 2, hidden_dim]
            confidence_scores: Dictionary of confidence scores
            
        Returns:
            Tensor of same shape with contextualized representations
        """
        # Pre-norm or post-norm architecture
        if self.prenorm:
            # Pre-norm: Apply normalization before attention
            norm_x = self.norm1(x)
            attn_output = self.self_attention(norm_x, norm_x, norm_x, confidence_scores)
            x = x + attn_output  # Residual connection
            
            # Feed-forward with normalization
            norm_x = self.norm2(x)
            ff_output = self.ff_network(norm_x)
            x = x + ff_output  # Residual connection
        else:
            # Post-norm: Apply attention then normalization
            attn_output = self.self_attention(x, x, x, confidence_scores)
            x = self.norm1(x + attn_output)  # Residual connection and norm
            
            # Feed-forward and normalization
            ff_output = self.ff_network(x)
            x = self.norm2(x + ff_output)  # Residual connection and norm
        
        return x


class ConfidenceWeightedAttention(nn.Module):
    """
    Multi-head attention with confidence weighting.
    
    This applies the formula:
    Attention(Q,K,V,Cd_spatial,Ci_spatial,Cd_velocity,Ci_velocity) = 
        softmax(QK^T/sqrt(dk) + f(Cd_spatial,Ci_spatial,Cd_velocity,Ci_velocity))V
    """
    def __init__(self, embed_dim, num_heads):
        super(ConfidenceWeightedAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable parameters for confidence weighting
        self.a = nn.Parameter(torch.tensor([0.0]))  # For Cd_spatial
        self.b = nn.Parameter(torch.tensor([0.0]))  # For Cd_velocity
        self.c = nn.Parameter(torch.tensor([0.0]))  # For Ci_spatial
        self.d = nn.Parameter(torch.tensor([0.0]))  # For Ci_velocity
        
        # Small epsilon to avoid log(0)
        self.epsilon = 0.1
    
  
    def compute_confidence_weights(self, confidence_scores):
        """
        Compute the confidence weighting matrix f(Cd_spatial, Ci_spatial, Cd_velocity, Ci_velocity).

        Args:
            confidence_scores: Dictionary with confidence score tensors of shape [flattened_batch_size, 2]

        Returns:
            Tensor of shape [flattened_batch_size, 2, 2] for weighting attention scores
        """
        Cd_spatial = confidence_scores['Cd_spatial']
        Ci_spatial = confidence_scores['Ci_spatial']
        Cd_velocity = confidence_scores['Cd_velocity']
        Ci_velocity = confidence_scores['Ci_velocity']

        
        # These tensors have shape [flattened_batch_size, 2]
        flattened_batch_size, num_hands = Cd_spatial.shape

        # Apply the confidence weighting formula
        f_values = (
            torch.log2(self.epsilon + Cd_spatial) * torch.sigmoid(self.a) * 0.25 +
            torch.log2(self.epsilon + Cd_velocity) * torch.sigmoid(self.b) * 0.25 +
            torch.log2(self.epsilon + Ci_spatial) * torch.sigmoid(self.c) * 0.5 +
            torch.log2(self.epsilon + Ci_velocity) * torch.sigmoid(self.d) * 0.5
        )
    
        scale_factor = math.sqrt(self.head_dim)
    
    # Scale the confidence weights
        f_values = f_values / scale_factor
        # Create the 2x2 matrix for each batch item where columns have same values
        confidence_matrix = f_values.unsqueeze(1).expand(-1, num_hands, -1)
        confidence_matrix = torch.clamp(confidence_matrix, min=-1e9, max=1e9)
        
        return confidence_matrix
    
    def forward(self, query, key, value, confidence_scores):
        """
        Apply confidence-weighted attention.
        
        Args:
            query, key, value: Tensors of shape [batch_size*seq_len, num_hands, embed_dim]
                              where batch_size*seq_len represents flattened batch and sequence dimensions
            confidence_scores: Dictionary of confidence scores
            
        Returns:
            Attention output tensor of same shape
        """

        # Get the shape components - note there's no separate sequence dimension here!
        flattened_batch_size, num_hands, embed_dim = query.shape
        
        # Linear projections
        q = self.q_proj(query)  # [flattened_batch_size, num_hands, embed_dim]
        k = self.k_proj(key)    # [flattened_batch_size, num_hands, embed_dim]
        v = self.v_proj(value)  # [flattened_batch_size, num_hands, embed_dim]
        
        # Compute confidence weights
        # This should return: [flattened_batch_size, num_hands, num_hands]
        confidence_weights = self.compute_confidence_weights(confidence_scores)

        # Reshape for multi-head attention
        # Split embed_dim into num_heads × head_dim
        q = q.reshape(flattened_batch_size, num_hands, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [flattened_batch_size, num_heads, num_hands, head_dim]
        
        k = k.reshape(flattened_batch_size, num_hands, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [flattened_batch_size, num_heads, num_hands, head_dim]
        
        v = v.reshape(flattened_batch_size, num_hands, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [flattened_batch_size, num_heads, num_hands, head_dim]
        
        # Calculate attention scores
        # [flattened_batch_size, num_heads, num_hands, num_hands]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add confidence weights to attention scores
        # Expand confidence_weights for all heads
        # [flattened_batch_size, 1, num_hands, num_hands]
        confidence_weights = confidence_weights.unsqueeze(1)
        
        # Add confidence weights to attention scores
        attention_scores = attention_scores + confidence_weights
        attention_scores = torch.clamp(attention_scores, min=-20, max=20)

        

        attn_max, _ = torch.max(attention_scores, dim=-1, keepdim=True)
        attention_scores = attention_scores - attn_max.detach()
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        # [flattened_batch_size, num_heads, num_hands, head_dim]
        context = torch.matmul(attention_probs, v)
        
        # Reshape back
        context = context.permute(0, 2, 1, 3)  # [flattened_batch_size, num_hands, num_heads, head_dim]
        context = context.reshape(flattened_batch_size, num_hands, embed_dim)
        
        # Final projection
        output = self.out_proj(context)  # [flattened_batch_size, num_hands, embed_dim]

        return output
    
class TemporalDownsampler(nn.Module):
    """
    Reduces frame count using 1D convolution with configurable parameters.
    
    This module applies a 1D convolution across the temporal dimension,
    effectively reducing the number of frames while preserving important
    temporal information through learned filters.
    """
    def __init__(self, 
                 input_dim, 
                 output_channels=None, 
                 kernel_size=3, 
                 stride=2,
                 activation='relu',
                 norm_layer=True,
                 init_method='xavier_uniform',
                 gain=1.0,
                 bias_init='zeros',
                 **init_kwargs):
        """
        Initialize the temporal downsampler.
        
        Args:
            input_dim: Input dimension (d) - feature size per frame
            output_channels: Number of convolutional filters (C), defaults to input_dim
            kernel_size: Size of the convolutional kernel (k)
            stride: Stride of the convolution, controls downsampling factor
            activation: Activation function ('relu', 'gelu', None)
            norm_layer: Whether to include layer normalization after convolution
        """
        super(TemporalDownsampler, self).__init__()
        
        # Default output channels to input dimension if not specified
        self.output_channels = input_dim if output_channels is None else output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Calculate padding to maintain temporal alignment
        # For even kernel sizes, we'll use asymmetric padding later
        self.padding = (kernel_size - 1) // 2
        self.is_even_kernel = (kernel_size % 2 == 0)
        
        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,  # This will be adjusted for even kernels
            bias=True
        )
        
        # Normalization layer
        self.norm = nn.LayerNorm(self.output_channels) if norm_layer else None
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    # Initialize weights
        self._apply_init(self.conv.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize bias
        if self.conv.bias is not None:
            self._apply_init(self.conv.bias, bias_init, **init_kwargs)
        
        # Initialize norm layer weights if present
        if self.norm is not None:
            if hasattr(self.norm, 'weight') and self.norm.weight is not None:
                self._apply_init(self.norm.weight, init_method, gain=gain, **init_kwargs)
            if hasattr(self.norm, 'bias') and self.norm.bias is not None:
                self._apply_init(self.norm.bias, bias_init, **init_kwargs)
    
    def _apply_init(self, tensor, init_type, **kwargs):
        """
        Apply the specified initialization to a tensor.

        Args:
            tensor: The tensor to initialize
            init_type: Initialization method name
            **kwargs: Additional parameters for initialization
        """
        # Check tensor dimensions
        if len(tensor.shape) < 2:
            # For 1D tensors (biases, layer norm weights), use simpler initialization
            if init_type in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']:
                # For these methods that require 2D+ tensors, fall back to a simpler method
                if bias_init := kwargs.get('bias_init', 'zeros'):
                    if bias_init == 'zeros':
                        nn.init.zeros_(tensor)
                    elif bias_init == 'ones':
                        nn.init.ones_(tensor)
                    elif bias_init == 'constant':
                        val = kwargs.get('val', 0.0)
                        nn.init.constant_(tensor, val=val)
                    elif bias_init == 'normal':
                        mean = kwargs.get('mean', 0.0)
                        std = kwargs.get('std', 0.01)
                        nn.init.normal_(tensor, mean=mean, std=std)
                    elif bias_init == 'uniform':
                        a = kwargs.get('a', -0.1)
                        b = kwargs.get('b', 0.1)
                        nn.init.uniform_(tensor, a=a, b=b)
                    else:
                        nn.init.zeros_(tensor)  # Default fallback
                else:
                    nn.init.zeros_(tensor)  # Default fallback
            else:
                # For other initializations, proceed with the specified method
                if init_type == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif init_type == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                elif init_type == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif init_type == 'zeros':
                    nn.init.zeros_(tensor)
                elif init_type == 'ones':
                    nn.init.ones_(tensor)
        else:
            # For 2D+ tensors, use the specified initialization method
            if init_type == 'xavier_uniform':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_uniform_(tensor, gain=gain)

            elif init_type == 'xavier_normal':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_normal_(tensor, gain=gain)

            elif init_type == 'kaiming_uniform':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'kaiming_normal':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'normal':
                mean = kwargs.get('mean', 0.0)
                std = kwargs.get('std', 0.01)
                nn.init.normal_(tensor, mean=mean, std=std)

            elif init_type == 'uniform':
                a = kwargs.get('a', -0.1)
                b = kwargs.get('b', 0.1)
                nn.init.uniform_(tensor, a=a, b=b)

            elif init_type == 'constant':
                val = kwargs.get('val', 0.0)
                nn.init.constant_(tensor, val=val)

            elif init_type == 'zeros':
                nn.init.zeros_(tensor)

            elif init_type == 'ones':
                nn.init.ones_(tensor)

    def forward(self, x):
        """
        Apply temporal downsampling to the input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, n_frames, input_dim]
            
        Returns:
            Tensor of shape [batch_size, n_frames/stride, output_channels]
        """
        batch_size, n_frames, input_dim = x.shape
        
        # Reshape for conv1d which expects [batch_size, channels, length]
        x = x.permute(0, 2, 1)  # -> [batch_size, input_dim, n_frames]
        
        # Handle even-sized kernels with asymmetric padding if needed
        if self.is_even_kernel:
            # For even kernels, PyTorch padding is not symmetric
            # We'll pad manually to handle this
            pad_size = (self.kernel_size - 1) // 2
            x = nn.functional.pad(x, (pad_size, pad_size+1), mode='constant', value=0)
            
        # Apply convolution
        x = self.conv(x)  # -> [batch_size, output_channels, n_frames/stride]
        
        # Reshape back to [batch_size, n_frames/stride, output_channels]
        x = x.permute(0, 2, 1)
        
        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
        
        # Apply activation if specified
        if self.activation is not None:
            x = self.activation(x)
        
        return x
    
    def compute_output_shape(self, input_length):
        """
        Calculate the output sequence length given the input length.
        
        Args:
            input_length: Length of the input sequence (n_frames)
            
        Returns:
            Length of the output sequence
        """
        # For even kernels with our manual padding
        if self.is_even_kernel:
            padding = self.padding + 1
        else:
            padding = self.padding
        
        # Standard formula for conv output shape
        return math.floor((input_length + 2 * padding - self.kernel_size) / self.stride + 1)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create fixed positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x, scale=1.0):
        """
        Add positional encodings to the input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            scale: Scaling factor for the positional encodings
            
        Returns:
            Tensor with added positional encodings
        """
        x = x + (self.pe[:, :x.size(1), :] * scale)
        return x
    
class MultiScaleTemporalTransformer(nn.Module):
    """
    Transformer that processes sequences with multi-scale temporal attention.
    
    Uses exactly three attention heads:
    - Short-term head: Attends to frames within ±5 frames
    - Medium-term head: Attends to frames within ±15 frames
    - Long-term head: Attends to frames within ±45 frames
    """
    def __init__(self, 
                 d_model, 
                 num_layers=4,
                 short_range=5,
                 medium_range=15,
                 long_range=45,
                 dim_feedforward=2048,
                 activation='gelu',
                 stride=2,
                 init_method='xavier_uniform',
                 gain=1.0,
                 bias_init='zeros',
                 **init_kwargs):  
        """
        Initialize the multi-scale temporal transformer.
        
        Args:
            d_model: Model dimension / feature size
            num_layers: Number of transformer encoder layers
            short_range: Range for short-term attention (±frames)
            medium_range: Range for medium-term attention (±frames)
            long_range: Range for long-term attention (±frames)
            dim_feedforward: Dimension of feedforward network
            activation: Activation function type
            stride: Stride used in downsampling (needed for mask adjustment)
        """
        super(MultiScaleTemporalTransformer, self).__init__()
        
        # Store parameters
        self.d_model = d_model
        self.total_heads = 3  # Exactly 3 heads
        self.head_ranges = {
            'short': short_range,
            'medium': medium_range,
            'long': long_range
        }
        self.stride = stride
        
        # Create transformer layers
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                MultiScaleTransformerEncoderLayer(
                    d_model=d_model,
                    head_ranges=self.head_ranges,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    init_method=init_method,
                    gain=gain,
                    bias_init=bias_init,
                    **init_kwargs
                    )
            )
        self.layers = nn.ModuleList(encoder_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, mask=None):
        """
        Process the input sequence through the transformer.
        
        Args:
            src: Input tensor [batch_size, seq_len_downsampled, d_model]
            mask: Boolean mask [batch_size, seq_len_original] where True indicates valid frames
                 and False indicates padding frames
            
        Returns:
            Output tensor of same shape as input with multi-scale temporal context
        """
        output = src
        
        # Adjust mask for downsampled sequence length
        if mask is not None:
            # Subsample the mask to match downsampled sequence
            # Take every stride-th element, starting from 0
            # This accounts for how conv1d downsampling affects the sequence length
            downsample_mask = mask[:, ::self.stride]
            
            # Make sure downsampled mask matches sequence length
            # It might be off by 1 due to padding in conv1d
            if downsample_mask.shape[1] > src.shape[1]:
                downsample_mask = downsample_mask[:, :src.shape[1]]
            elif downsample_mask.shape[1] < src.shape[1]:
                # This shouldn't normally happen, but just in case
                pad_size = src.shape[1] - downsample_mask.shape[1]
                pad = torch.zeros((downsample_mask.shape[0], pad_size), dtype=torch.bool, device=mask.device)
                downsample_mask = torch.cat([downsample_mask, pad], dim=1)
            
            # Convert from True=valid to True=padding format used by transformer
            padding_mask = ~downsample_mask
        else:
            padding_mask = None
        
        # Pass through each transformer layer
        for layer in self.layers:
            output = layer(output, padding_mask=padding_mask)
        
        # Apply final normalization
        output = self.norm(output)

        
        return output


class MultiScaleTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-scale temporal attention.
    """
    def __init__(self, 
                 d_model, 
                 head_ranges,
                 dim_feedforward=2048, 
                 activation="gelu",
                 init_method='xavier_uniform',
                 gain=1.0,
                 bias_init='zeros',
                 **init_kwargs):
        super(MultiScaleTransformerEncoderLayer, self).__init__()
        
        # Multi-scale attention
        self.self_attn = MultiScaleAttention(
            embed_dim=d_model,
            head_ranges=head_ranges
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize feed-forward weights
        self._apply_init(self.linear1.weight, init_method, gain=gain, **init_kwargs)
        self._apply_init(self.linear2.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize feed-forward biases
        if self.linear1.bias is not None:
            self._apply_init(self.linear1.bias, bias_init, **init_kwargs)
        if self.linear2.bias is not None:
            self._apply_init(self.linear2.bias, bias_init, **init_kwargs)
        
        # Initialize layer norm parameters
        if hasattr(self.norm1, 'weight') and self.norm1.weight is not None:
            self._apply_init(self.norm1.weight, init_method, gain=gain, **init_kwargs)
        if hasattr(self.norm1, 'bias') and self.norm1.bias is not None:
            self._apply_init(self.norm1.bias, bias_init, **init_kwargs)
        
        if hasattr(self.norm2, 'weight') and self.norm2.weight is not None:
            self._apply_init(self.norm2.weight, init_method, gain=gain, **init_kwargs)
        if hasattr(self.norm2, 'bias') and self.norm2.bias is not None:
            self._apply_init(self.norm2.bias, bias_init, **init_kwargs)
    
    def _apply_init(self, tensor, init_type, **kwargs):
        """
        Apply the specified initialization to a tensor.

        Args:
            tensor: The tensor to initialize
            init_type: Initialization method name
            **kwargs: Additional parameters for initialization
        """
        # Check tensor dimensions
        if len(tensor.shape) < 2:
            # For 1D tensors (biases, layer norm weights), use simpler initialization
            if init_type in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']:
                # For these methods that require 2D+ tensors, fall back to a simpler method
                if bias_init := kwargs.get('bias_init', 'zeros'):
                    if bias_init == 'zeros':
                        nn.init.zeros_(tensor)
                    elif bias_init == 'ones':
                        nn.init.ones_(tensor)
                    elif bias_init == 'constant':
                        val = kwargs.get('val', 0.0)
                        nn.init.constant_(tensor, val=val)
                    elif bias_init == 'normal':
                        mean = kwargs.get('mean', 0.0)
                        std = kwargs.get('std', 0.01)
                        nn.init.normal_(tensor, mean=mean, std=std)
                    elif bias_init == 'uniform':
                        a = kwargs.get('a', -0.1)
                        b = kwargs.get('b', 0.1)
                        nn.init.uniform_(tensor, a=a, b=b)
                    else:
                        nn.init.zeros_(tensor)  # Default fallback
                else:
                    nn.init.zeros_(tensor)  # Default fallback
            else:
                # For other initializations, proceed with the specified method
                if init_type == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif init_type == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                elif init_type == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif init_type == 'zeros':
                    nn.init.zeros_(tensor)
                elif init_type == 'ones':
                    nn.init.ones_(tensor)
        else:
            # For 2D+ tensors, use the specified initialization method
            if init_type == 'xavier_uniform':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_uniform_(tensor, gain=gain)

            elif init_type == 'xavier_normal':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_normal_(tensor, gain=gain)

            elif init_type == 'kaiming_uniform':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'kaiming_normal':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'normal':
                mean = kwargs.get('mean', 0.0)
                std = kwargs.get('std', 0.01)
                nn.init.normal_(tensor, mean=mean, std=std)

            elif init_type == 'uniform':
                a = kwargs.get('a', -0.1)
                b = kwargs.get('b', 0.1)
                nn.init.uniform_(tensor, a=a, b=b)

            elif init_type == 'constant':
                val = kwargs.get('val', 0.0)
                nn.init.constant_(tensor, val=val)

            elif init_type == 'zeros':
                nn.init.zeros_(tensor)

            elif init_type == 'ones':
                nn.init.ones_(tensor)
    
    def forward(self, src, padding_mask=None):
        """
        Forward pass through the transformer encoder layer.
        
        Args:
            src: Input tensor [batch_size, seq_len_downsampled, d_model]
            padding_mask: Boolean mask [batch_size, seq_len_downsampled] 
                         where True indicates padding
            
        Returns:
            Output tensor of the same shape
        """
        # Multi-scale attention with residual connection
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, padding_mask=padding_mask)
        src = src + src2
        
        # Feed-forward network with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + src2
        
        return src


class MultiScaleAttention(nn.Module):
    """
    Multi-head attention where different heads attend to different temporal ranges.
    Uses exactly 3 heads: short, medium, and long-term.
    """
    def __init__(self, embed_dim, head_ranges, init_method='xavier_uniform', gain=1.0, bias_init='zeros', **init_kwargs):
        super(MultiScaleAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.head_ranges = head_ranges
        self.total_heads = 3  # Fixed: one head per range
        
        assert embed_dim % self.total_heads == 0, "embed_dim must be divisible by 3"
        self.head_dim = embed_dim // self.total_heads
        
        # Create linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm for additional stability
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Head indices (fixed for 3 heads)
        self.head_indices = {
            'short': (0, 1),
            'medium': (1, 2),
            'long': (2, 3)
        }
        self._apply_init(self.q_proj.weight, init_method, gain=gain, **init_kwargs)
        self._apply_init(self.k_proj.weight, init_method, gain=gain, **init_kwargs)
        self._apply_init(self.v_proj.weight, init_method, gain=gain, **init_kwargs)
        self._apply_init(self.out_proj.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize all biases
        if self.q_proj.bias is not None:
            self._apply_init(self.q_proj.bias, bias_init, **init_kwargs)
        if self.k_proj.bias is not None:
            self._apply_init(self.k_proj.bias, bias_init, **init_kwargs)
        if self.v_proj.bias is not None:
            self._apply_init(self.v_proj.bias, bias_init, **init_kwargs)
        if self.out_proj.bias is not None:
            self._apply_init(self.out_proj.bias, bias_init, **init_kwargs)
        
        # Initialize layer norm parameters
        if hasattr(self.layer_norm, 'weight') and self.layer_norm.weight is not None:
            self._apply_init(self.layer_norm.weight, init_method, gain=gain, **init_kwargs)
        if hasattr(self.layer_norm, 'bias') and self.layer_norm.bias is not None:
            self._apply_init(self.layer_norm.bias, bias_init, **init_kwargs)
    
    def _apply_init(self, tensor, init_type, **kwargs):
        """
        Apply the specified initialization to a tensor.

        Args:
            tensor: The tensor to initialize
            init_type: Initialization method name
            **kwargs: Additional parameters for initialization
        """
        # Check tensor dimensions
        if len(tensor.shape) < 2:
            # For 1D tensors (biases, layer norm weights), use simpler initialization
            if init_type in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']:
                # For these methods that require 2D+ tensors, fall back to a simpler method
                if bias_init := kwargs.get('bias_init', 'zeros'):
                    if bias_init == 'zeros':
                        nn.init.zeros_(tensor)
                    elif bias_init == 'ones':
                        nn.init.ones_(tensor)
                    elif bias_init == 'constant':
                        val = kwargs.get('val', 0.0)
                        nn.init.constant_(tensor, val=val)
                    elif bias_init == 'normal':
                        mean = kwargs.get('mean', 0.0)
                        std = kwargs.get('std', 0.01)
                        nn.init.normal_(tensor, mean=mean, std=std)
                    elif bias_init == 'uniform':
                        a = kwargs.get('a', -0.1)
                        b = kwargs.get('b', 0.1)
                        nn.init.uniform_(tensor, a=a, b=b)
                    else:
                        nn.init.zeros_(tensor)  # Default fallback
                else:
                    nn.init.zeros_(tensor)  # Default fallback
            else:
                # For other initializations, proceed with the specified method
                if init_type == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif init_type == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                elif init_type == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif init_type == 'zeros':
                    nn.init.zeros_(tensor)
                elif init_type == 'ones':
                    nn.init.ones_(tensor)
        else:
            # For 2D+ tensors, use the specified initialization method
            if init_type == 'xavier_uniform':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_uniform_(tensor, gain=gain)

            elif init_type == 'xavier_normal':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_normal_(tensor, gain=gain)

            elif init_type == 'kaiming_uniform':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'kaiming_normal':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

            elif init_type == 'normal':
                mean = kwargs.get('mean', 0.0)
                std = kwargs.get('std', 0.01)
                nn.init.normal_(tensor, mean=mean, std=std)

            elif init_type == 'uniform':
                a = kwargs.get('a', -0.1)
                b = kwargs.get('b', 0.1)
                nn.init.uniform_(tensor, a=a, b=b)

            elif init_type == 'constant':
                val = kwargs.get('val', 0.0)
                nn.init.constant_(tensor, val=val)

            elif init_type == 'zeros':
                nn.init.zeros_(tensor)

            elif init_type == 'ones':
                nn.init.ones_(tensor)
    def forward(self, query, key, value, padding_mask=None):
        """
        Apply multi-scale attention with enhanced numerical stability.
        """
        # Safety check - replace NaNs with zeros in inputs

        
        # Apply layer norm for additional stability
        query = self.layer_norm(query)
        
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        try:
            # Linear projections and reshape for multi-head attention
            q = self.q_proj(query).view(batch_size, tgt_len, self.total_heads, self.head_dim)
            k = self.k_proj(key).view(batch_size, src_len, self.total_heads, self.head_dim)
            v = self.v_proj(value).view(batch_size, src_len, self.total_heads, self.head_dim)
            

            
            # Transpose for attention computation
            q = q.transpose(1, 2)  # [batch_size, total_heads, tgt_len, head_dim]
            k = k.transpose(1, 2)  # [batch_size, total_heads, src_len, head_dim]
            v = v.transpose(1, 2)  # [batch_size, total_heads, src_len, head_dim]
            
            # Compute attention scores with safety checks
            attn_output = self._multi_scale_attention(q, k, v, tgt_len, src_len, padding_mask)

            
            # Reshape and apply final projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
            output = self.out_proj(attn_output)
            
            # Final output safety check
            output = torch.clamp(output, min=-1e4, max=1e4)  # Prevent extreme values
            
            return output
            
        except Exception as e:
            print(f"Error in MultiScaleAttention: {str(e)}")
            # Fall back to original input in case of any error
            return query
    
    def _multi_scale_attention(self, q, k, v, tgt_len, src_len, padding_mask):
        """
        Apply attention with different temporal ranges for different heads.
        """
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create temporal range masks for each head type
        temporal_masks = self._create_temporal_masks(tgt_len, src_len, device=q.device)

        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            # Use a large negative number instead of -inf
            attn_weights = attn_weights.masked_fill(padding_mask, -1e6)

        # Apply the temporal masks with a large negative number instead of -inf
        for scale, (start_idx, end_idx) in self.head_indices.items():
            mask = temporal_masks[scale]
            attn_weights[:, start_idx:end_idx] = attn_weights[:, start_idx:end_idx].masked_fill(mask, -1e6)

        # Check if any row is completely masked
        # If a row has all -1e9, add a small value to the first element
        mask_check = (attn_weights <= -1e6).all(dim=-1, keepdim=True)
        attn_weights = attn_weights.masked_fill(mask_check, 0.0)

        attn_weights = torch.clamp(attn_weights, min=-20, max=20)
        # Apply softmax with added stability
        # Subtract max value for numerical stability
        attn_weights_max, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        attn_weights = attn_weights - attn_weights_max.detach()
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)

        return output
    
    def _create_temporal_masks(self, tgt_len, src_len, device):
        """
        Create masks to restrict attention to specific temporal ranges.
        """
        try:
            temporal_masks = {}
            
            # Create position indices
            pos_i = torch.arange(tgt_len, device=device).unsqueeze(1)
            pos_j = torch.arange(src_len, device=device).unsqueeze(0)
            
            # Calculate distance between positions
            dist = torch.abs(pos_i - pos_j)  # [tgt_len, src_len]
            
            # Create masks for each temporal range
            for scale, range_val in self.head_ranges.items():
                # True where attention should be blocked (outside of the range)
                mask = dist > range_val
                
                # Safety check - ensure not all positions are masked
                if mask.all():
                    # If all positions would be masked, allow attention to the closest position
                    closest_pos = torch.argmin(dist, dim=1, keepdim=True)
                    mask.scatter_(1, closest_pos, False)
                
                # Expand for batch dimension and appropriate number of heads
                # Shape: [1, 1, tgt_len, src_len]
                temporal_masks[scale] = mask.unsqueeze(0).unsqueeze(0)
            
            return temporal_masks
            
        except Exception as e:
            print(f"Error in _create_temporal_masks: {str(e)}")
            # Return empty masks as fallback
            return {scale: torch.zeros((1, 1, tgt_len, src_len), dtype=torch.bool, device=device) 
                   for scale in self.head_ranges.keys()}
    
def compute_minimum_loss(target_labels, label_mask=None):
    """
    Compute the theoretical minimum loss (entropy of target distribution)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Calculate entropy for each position: -∑(p_i * log(p_i))
    position_entropy = -(target_labels * torch.log(target_labels + epsilon)).sum(dim=2)
    
    # Apply mask if provided
    if label_mask is not None:
        position_entropy = position_entropy * label_mask.float()
        # Average entropy over valid tokens
        min_loss = position_entropy.sum() / label_mask.sum().clamp(min=1)
    else:
        # If no mask, use all tokens
        min_loss = position_entropy.mean()
    
    return min_loss

def optimized_semantic_smoothing_loss(logits, L_index, L_values, label_mask=None):
    """
    Highly optimized semantic smoothing loss using a single scatter operation.
    No loops needed!
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Create target label distributions all at once
    target_labels = torch.zeros(batch_size, seq_len, vocab_size, device=logits.device)
    target_labels.scatter_(2, L_index, L_values)
    
    # Apply log_softmax to get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute loss (batch_size, seq_len)
    token_losses = -(target_labels * log_probs).sum(dim=2)
    
    # Apply mask if provided
    if label_mask is not None:
        token_losses = token_losses * label_mask.float()
        # Average loss over valid tokens
        total_loss = token_losses.sum() / label_mask.sum().clamp(min=1)
    else:
        # If no mask, use all tokens
        total_loss = token_losses.mean()
    
    return total_loss  

class OptimizedCrossAttention(nn.Module):
    """
    Cross-attention using PyTorch's optimized MultiheadAttention implementation.
    """
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1, init_method='xavier_uniform', gain=1.0, bias_init='zeros', **init_kwargs):
        super(OptimizedCrossAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # PyTorch's optimized multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important for our [batch, seq, features] format
        )
        
        # Layer normalization for pre-norm architecture (like GPT-2)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

        if hasattr(self.multihead_attn, 'in_proj_weight') and self.multihead_attn.in_proj_weight is not None:
            self._apply_init(self.multihead_attn.in_proj_weight, init_method, gain=gain, **init_kwargs)
        
        if hasattr(self.multihead_attn, 'out_proj') and hasattr(self.multihead_attn.out_proj, 'weight'):
            self._apply_init(self.multihead_attn.out_proj.weight, init_method, gain=gain, **init_kwargs)
        
        # Initialize bias terms
        if hasattr(self.multihead_attn, 'in_proj_bias') and self.multihead_attn.in_proj_bias is not None:
            self._apply_init(self.multihead_attn.in_proj_bias, bias_init, **init_kwargs)
        
        if hasattr(self.multihead_attn, 'out_proj') and hasattr(self.multihead_attn.out_proj, 'bias') and self.multihead_attn.out_proj.bias is not None:
            self._apply_init(self.multihead_attn.out_proj.bias, bias_init, **init_kwargs)
        
        # Initialize layer norm parameters
        if hasattr(self.layer_norm, 'weight') and self.layer_norm.weight is not None:
            self._apply_init(self.layer_norm.weight, init_method, gain=gain, **init_kwargs)
        
        if hasattr(self.layer_norm, 'bias') and self.layer_norm.bias is not None:
            self._apply_init(self.layer_norm.bias, bias_init, **init_kwargs)
        
        if hasattr(self.output_layer_norm, 'weight') and self.output_layer_norm.weight is not None:
            self._apply_init(self.output_layer_norm.weight, init_method, gain=gain, **init_kwargs)
        
        if hasattr(self.output_layer_norm, 'bias') and self.output_layer_norm.bias is not None:
            self._apply_init(self.output_layer_norm.bias, bias_init, **init_kwargs)
    
    def _apply_init(self, tensor, init_type, **kwargs):
        """
        Apply the specified initialization to a tensor.
        
        Args:
            tensor: The tensor to initialize
            init_type: Initialization method name
            **kwargs: Additional parameters for initialization
        """
        # Check tensor dimensions
        if len(tensor.shape) < 2:
            # For 1D tensors (biases, layer norm weights), use simpler initialization
            if init_type in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']:
                # For these methods that require 2D+ tensors, fall back to a simpler method
                fallback_init = kwargs.get('bias_init', 'zeros')
                
                if fallback_init == 'zeros':
                    nn.init.zeros_(tensor)
                elif fallback_init == 'ones':
                    nn.init.ones_(tensor)
                elif fallback_init == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif fallback_init == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif fallback_init == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                else:
                    nn.init.zeros_(tensor)  # Default fallback
            else:
                # For other initializations, proceed with the specified method
                if init_type == 'normal':
                    mean = kwargs.get('mean', 0.0)
                    std = kwargs.get('std', 0.01)
                    nn.init.normal_(tensor, mean=mean, std=std)
                elif init_type == 'uniform':
                    a = kwargs.get('a', -0.1)
                    b = kwargs.get('b', 0.1)
                    nn.init.uniform_(tensor, a=a, b=b)
                elif init_type == 'constant':
                    val = kwargs.get('val', 0.0)
                    nn.init.constant_(tensor, val=val)
                elif init_type == 'zeros':
                    nn.init.zeros_(tensor)
                elif init_type == 'ones':
                    nn.init.ones_(tensor)
        else:
            # For 2D+ tensors, use the specified initialization method
            if init_type == 'xavier_uniform':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_uniform_(tensor, gain=gain)
            
            elif init_type == 'xavier_normal':
                gain = kwargs.get('gain', 1.0)
                nn.init.xavier_normal_(tensor, gain=gain)
            
            elif init_type == 'kaiming_uniform':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
            
            elif init_type == 'kaiming_normal':
                a = kwargs.get('a', 0)
                mode = kwargs.get('mode', 'fan_in')
                nonlinearity = kwargs.get('nonlinearity', 'leaky_relu')
                nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
            
            elif init_type == 'normal':
                mean = kwargs.get('mean', 0.0)
                std = kwargs.get('std', 0.01)
                nn.init.normal_(tensor, mean=mean, std=std)
            
            elif init_type == 'uniform':
                a = kwargs.get('a', -0.1)
                b = kwargs.get('b', 0.1)
                nn.init.uniform_(tensor, a=a, b=b)
            
            elif init_type == 'constant':
                val = kwargs.get('val', 0.0)
                nn.init.constant_(tensor, val=val)
            
            elif init_type == 'zeros':
                nn.init.zeros_(tensor)
            
            elif init_type == 'ones':
                nn.init.ones_(tensor)

    def forward(self, hidden_states, video_representations, video_mask=None, stride=1):
        """
        Compute cross-attention between GPT token representations and video frames.
        """
        # Apply layer normalization to hidden states (pre-norm approach)
        hidden_states = torch.clamp(hidden_states, min=-1e4, max=1e4)
        video_representations = torch.clamp(video_representations, min=-1e4, max=1e4)

        query = self.layer_norm(hidden_states)
        
        # Handle strided video mask
        if video_mask is not None and stride > 1:
            # Subsample the mask to match video_representations shape
            video_mask = video_mask[:, ::stride]
            
            # Ensure mask length matches
            frame_length = video_representations.shape[1]
            if video_mask.shape[1] > frame_length:
                video_mask = video_mask[:, :frame_length]
            elif video_mask.shape[1] < frame_length:
                pad_size = frame_length - video_mask.shape[1]
                pad = torch.zeros((video_mask.shape[0], pad_size), dtype=torch.bool, device=video_mask.device)
                video_mask = torch.cat([video_mask, pad], dim=1)
            
            # Convert to attention mask format expected by PyTorch
            # True = don't attend, False = attend
            attn_mask = ~video_mask
            if attn_mask is not None:
                completely_masked = attn_mask.all(dim=1, keepdim=True)
                if completely_masked.any():
                    # For completely masked rows, unmask at least one position
                    attn_mask = attn_mask.clone()
                    attn_mask[completely_masked.expand_as(attn_mask)] = False
        else:
            attn_mask = None
        

        try:
            cross_attention_output, _ = self.multihead_attn(
                query=query,                  # From GPT tokens
                key=video_representations,    # From video frames
                value=video_representations,  # From video frames
                key_padding_mask=attn_mask,   # Mask for padding frames
                need_weights=False            # Don't return attention weights to save computation
            )

            cross_attention_output = self.output_layer_norm(cross_attention_output)
            return cross_attention_output
        
        except Exception as e:
            print(f"ERROR in cross-attention: {str(e)}")
            # Fall back to identity mapping if exception occurs
            return query
    

class Adapter(nn.Module):
    """Efficient adapter module for creating trainable pathways."""
    def __init__(self, hidden_size, adapter_dim):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        
        # Conservative initialization
        nn.init.xavier_uniform_(self.down.weight, gain=0.8)
        nn.init.xavier_uniform_(self.up.weight, gain=0.8)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        # Store input for residual
        residual = x
        
        # Down-project
        x = self.down(x)
        x = self.act(x)
        
        # Up-project
        x = self.up(x)
        
        # Add residual
        x = residual + x
        
        # Layer norm for stability
        x = self.ln(x)
        
        return x
    
class ExpansionAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.up = nn.Linear(hidden_size, 4 * hidden_size)
        self.down = nn.Linear(4 * hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.down.weight, gain=0.8)
        nn.init.xavier_uniform_(self.up.weight, gain=0.8)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        residual = x
        # Expand first
        x = self.up(x)
        x = self.act(x)
        # Then contract
        x = self.down(x)
        x = self.norm(x + residual)
        return x
    
#, init_method='xavier_uniform', gain=1.0, bias_init='zeros'
class VideoGPT(nn.Module):
    """
    Integrates pre-trained GPT-2 with cross-attention for video-to-text translation.
    """
    def __init__(self, model_name="distilgpt2", num_cross_heads=12, freeze_gpt=True, trainable_lm_head=True, trainable_final_ln=True, adapter_dim=384, stride=2):
        super(VideoGPT, self).__init__()
        
        # Load pre-trained model
        self.gpt = AutoModelForCausalLM.from_pretrained(model_name)
        self.gpt.to("cuda")
        self.config = self.gpt.config
        self.stride = stride
        
        # Dimensions
        self.hidden_size = self.config.n_embd  # 768 for distilGPT-2
        

        self.cross_attentions = nn.ModuleList([
            OptimizedCrossAttention(
                hidden_size=self.hidden_size,
                num_heads=num_cross_heads,
                dropout=0.1, init_method='xavier_uniform', gain=0.8, bias_init='zeros'
            ) 
            for _ in range(len(self.gpt.transformer.h))
        ])
        
        self.post_ffn_adapters = nn.ModuleList([
            Adapter(self.hidden_size, adapter_dim) 
            for _ in range(len(self.gpt.transformer.h))
        ])

        self.expansion_adapters = nn.ModuleList([
            ExpansionAdapter(self.hidden_size) 
            for _ in range(len(self.gpt.transformer.h))
        ])

        


        self.register_buffer('scale_min', torch.tensor(1.0))  # Register as buffer to move to correct device
        self.register_buffer('scale_max', torch.tensor(2.0))
        self.cross_attn_value = nn.Parameter(torch.zeros(1))
        

        self.trainable_lm_head = trainable_lm_head
        if trainable_lm_head:
            # Create new LM head initialized with pretrained weights
            self.new_lm_head = nn.Linear(self.hidden_size, self.gpt.config.vocab_size)
            self.new_lm_head.weight.data.copy_(self.gpt.lm_head.weight.data)
            if hasattr(self.gpt.lm_head, 'bias') and self.gpt.lm_head.bias is not None:
                self.new_lm_head.bias.data.copy_(self.gpt.lm_head.bias.data)


        self.trainable_final_ln = trainable_final_ln
        if trainable_final_ln:
            self.new_final_ln = nn.LayerNorm(self.hidden_size, eps=1e-5)
            # Copy weights and bias from pretrained layer norm
            self.new_final_ln.weight.data.copy_(self.gpt.transformer.ln_f.weight.data)
            self.new_final_ln.bias.data.copy_(self.gpt.transformer.ln_f.bias.data)

        # Freeze GPT-2 weights if specified
        if freeze_gpt:
            self._freeze_gpt_parameters()
    
    def _freeze_gpt_parameters(self):
        """Freeze all parameters of the GPT model."""
        for param in self.gpt.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, video_representations, video_mask=None, 
               L_index=None, L_values=None, label_mask=None):
        """
        Forward pass with integrated cross-attention and semantic smoothing loss.
        
        Args:
            input_ids: Token IDs for GPT [batch_size, n_tokens]
            video_representations: Video frame features [batch_size, n_frames/stride, hidden_size]
            video_mask: Mask tensor [batch_size, n_frames] with True for valid frames
            attention_mask: Mask for input tokens [batch_size, n_tokens]
            L_index: Token indices [batch_size, max_n_tokens, 6]
            L_values: Token values [batch_size, max_n_tokens, 6]
            label_mask: Boolean mask [batch_size, max_n_tokens]
            
        Returns:
            outputs: Model outputs including loss and logits
        """
        
        batch_size, n_tokens = input_ids.shape
        
        # Get GPT embeddings (word + position)
        position_ids = torch.arange(0, n_tokens, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        gpt_embeds = self.gpt.transformer.wte(input_ids) + self.gpt.transformer.wpe(position_ids)
        
        # Store states at each step
        hidden_states = gpt_embeds



        if label_mask is not None:
            # Create attention mask that combines padding and causal constraints
            extended_attention_mask = label_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

            # Step 2: Create causal mask (lower triangular matrix)
            seq_length = label_mask.size(1)
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), 
                                               device=label_mask.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

            # Step 3: Combine padding mask with causal mask
            combined_mask = causal_mask * extended_attention_mask.float()

            # Step 4: Convert to additive mask where 0 means "attend" and 
            # a large negative number means "don't attend"
            attention_mask = combined_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0


        # Process through GPT layers with cross-attention
        for i, block in enumerate(self.gpt.transformer.h):
            #hidden_states = torch.clamp(hidden_states, min=-50, max=50)
           
            # 1. GPT self-attention
            attn_outputs = block.attn(
                hidden_states,
                attention_mask=attention_mask if label_mask is not None else None
            )
            # Add residual connection
            hidden_states = attn_outputs[0] + hidden_states
            
            # 2. Feed-forward network
            feed_forward_output = block.mlp(hidden_states)
            hidden_states = hidden_states + feed_forward_output


            hidden_states = self.post_ffn_adapters[i](hidden_states)
           

            
            # 2. Insert our cross-attention between self-attention and FFN
            cross_attention_output = self.cross_attentions[i](
                hidden_states, 
                video_representations, 
                video_mask=video_mask,
                stride=self.stride
            )
            # Add residual connection to cross-attention
            
            scale_value = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(self.cross_attn_value)
            # Ensure proper broadcasting
            cross_attn_scale = scale_value.view(1, 1, 1)  # Reshape for broadcasting
            hidden_states = hidden_states + cross_attn_scale * cross_attention_output

            hidden_states = self.expansion_adapters[i](hidden_states)

        # Final layer norm
        if self.trainable_final_ln:
            hidden_states = self.new_final_ln(hidden_states)
        else:
            hidden_states = self.gpt.transformer.ln_f(hidden_states)
        
        # Language modeling head
        if self.trainable_lm_head:
            lm_logits = self.new_lm_head(hidden_states)
        else:
            lm_logits = self.gpt.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None

        
        if L_index is not None and L_values is not None:
            # Use our custom semantic smoothing loss
            loss = optimized_semantic_smoothing_loss(
                logits=lm_logits,
                L_index=L_index,
                L_values=L_values,
                label_mask=label_mask
            ) 
    
        
        return {
            "loss": loss, 
            "logits": lm_logits, 
            "hidden_states": hidden_states
        }
    

def make_inputs_for_model(L_index, tokenizer):
    primary_targets = L_index[:, :, 0].clone()
    batch_size, seq_len = primary_targets.shape
    input_ids = torch.zeros_like(primary_targets)
    input_ids[:, 0] = tokenizer.bos_token_id  # Start with BOS token
    input_ids[:, 1:] = primary_targets[:, :-1]
    return input_ids





def get_predictions_from_logits(logits, tokenizer):
    """
    Convert model logits to human-readable text predictions.
    """
    # Get the most likely token at each position
    predicted_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, sequence_length]
    
    # Convert to numpy for easier handling
    token_ids_np = predicted_token_ids.cpu().numpy()
    
    # Container for results
    results = []
    
    # Process each sequence in the batch
    for i, ids in enumerate(token_ids_np):
        # Decode the token IDs to text
        text = tokenizer.decode(ids)
        results.append({
            "label": i,
            "text": text,
        })
    
    return results


def compare_predictions(input_ids, logits, tokenizer):
    """Show comparison between inputs, predictions"""
    predictions = get_predictions_from_logits(logits=logits, tokenizer=tokenizer)
    results = []
    
    for i, pred in enumerate(predictions):
        # Input sequence
        input_sequence = tokenizer.decode(input_ids[i])
        results.append({
            "index": i,
            "input": input_sequence,
            "generated": pred['text']
        })
    
    return results


def train_asl_model(
    # Data parameters
    train_loader,
    val_loader, 
    train_batches,
    val_batches,
    
    # Model components (already initialized and on correct device)
    models_dict,  # Dictionary containing all model components
    tokenizer,    # Tokenizer for GPT model
    
    # Optimizer parameters
    learning_rate=1e-4,
    weight_decay=1e-6,
    
    # Training parameters
    num_epochs=30,
    grad_clip_value=1.0,
    scheduler_type='cosine',  # 'cosine', 'linear', 'step', None
    scheduler_params=None,    # Dict of params specific to the scheduler
    
    # Early stopping parameters
    early_stopping=True,
    patience=5,
    min_delta=0.001,
    
    # Checkpointing
    save_dir='./checkpoints',
    save_best_only=True,
    save_freq_epochs=1,
    auto_rescue=True,
    # Validation and visualization
    num_examples_to_display=5,  # Number of examples to display during validation
    
    # Logging
    log_freq_batches=10,
    verbose=1,
    
    # Device
    device='cuda'
):
    """
    Comprehensive training loop for the ASL Translation model.
    """

    
    # Create directory for checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    # Timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    def save_rescue_checkpoint(signal_received=None, frame=None):
            """Save a rescue checkpoint and exit gracefully if needed"""
            try:
                rescue_path = os.path.join(run_dir, f"rescue_checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt")
                print(f"\n\n{'='*50}")
                if signal_received:
                    print(f"Signal {signal_received} received. Saving rescue checkpoint...")
                else:
                    print(f"Exception detected. Saving rescue checkpoint...")

                # Save current state
                save_checkpoint(rescue_path, models_dict, optimizer, scheduler, epoch, history, 
                               extra_info={'interrupted_at_batch': batch_idx, 'exception': traceback.format_exc()})

                print(f"Rescue checkpoint saved to: {rescue_path}")
                print(f"You can resume training from this checkpoint later.")
                print(f"{'='*50}\n")

                if signal_received:  # If this was triggered by a signal, exit
                    sys.exit(0)

            except Exception as e:
                print(f"Failed to save rescue checkpoint: {e}")

    if auto_rescue:
        signal.signal(signal.SIGINT, save_rescue_checkpoint)  # Ctrl+C
        signal.signal(signal.SIGTERM, save_rescue_checkpoint)  # Termination request
    
    # Extract all models from dictionary
    model_components = list(models_dict.values())
    
    # Create optimizer with all parameters from all models
    all_params = []
    for model in model_components:
        all_params.extend(model.parameters())
    
    optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
    
    # Create scheduler if requested
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler_params = scheduler_params or {'T_max': num_epochs}
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'step':
        scheduler_params = scheduler_params or {'step_size': 10, 'gamma': 0.1}
        scheduler = StepLR(optimizer, **scheduler_params)
    elif scheduler_type == 'linear':
        scheduler_params = scheduler_params or {'start_factor': 1.0, 'end_factor': 0.1, 'total_iters': num_epochs}
        scheduler = LinearLR(optimizer, **scheduler_params)
    elif scheduler_type == 'plateau':
        scheduler_params = scheduler_params or {'mode': 'min', 'factor': 0.1, 'patience': 5, 'verbose': verbose > 0}
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_type == 'warmup_plateau':
        scheduler_params = scheduler_params or {'warmup_epochs': 3,'warmup_start_factor': 0.01,'mode': 'min', 'factor': 0.5, 'patience': 5, 'verbose': verbose > 0}
        scheduler = WarmupToPlateauScheduler(optimizer, **scheduler_params)
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_min_loss': [],  # Theoretical minimum loss
        'val_normalized_loss': [],  # Loss normalized by theoretical minimum
        'learning_rates': [],
        'examples': []  # Store validation examples
    }
    
    # Define a function to run the forward pass
    def forward_pass(batch):
        def check_nan(tensor, name):
            """Helper function to check for NaN values in tensors"""
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                return True
            return False

        # Check batch inputs first
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                print(f"batch_key {key}, contains nan")
        
        # Get embeddings
        embeddings = models_dict['embedding_table'].forward()
        check_nan(embeddings, "embeddings")

        dom_landmark_embeddings = embeddings[:20]
        dom_wrist_embedding = embeddings[20]
        non_dom_landmark_embeddings = embeddings[21:41]
        non_dom_wrist_embedding = embeddings[41]

        # Spatial encoding
        dom_landmarks_where = models_dict['landmark_encoder'].forward(batch['dom_landmarks'])
        check_nan(dom_landmarks_where, "dom_landmarks_where")

        non_dom_landmarks_where = models_dict['landmark_encoder'].forward(batch['non_dom_landmarks'])
        check_nan(non_dom_landmarks_where, "non_dom_landmarks_where")

        dom_landmarks_conc = combine_spatial_and_semantic_features(
            spatial_features=dom_landmarks_where, 
            semantic_features=dom_landmark_embeddings
        )
        check_nan(dom_landmarks_conc, "dom_landmarks_conc")

        non_dom_landmarks_conc = combine_spatial_and_semantic_features(
            spatial_features=non_dom_landmarks_where, 
            semantic_features=non_dom_landmark_embeddings
        )
        check_nan(non_dom_landmarks_conc, "non_dom_landmarks_conc")

        wrists_where = models_dict['wrist_encoder'].forward(wrist_coordinates=batch['nose_to_wrist_dist'])
        check_nan(wrists_where, "wrists_where")

        wrists_conc = combine_wrist_embedding_and_spatial(
            wrist_embeddings=torch.cat([dom_wrist_embedding, non_dom_wrist_embedding], dim=-1).reshape((2,-1)), 
            wrist_spatial_features=wrists_where
        )
        check_nan(wrists_conc, "wrists_conc")

        # Velocity encoding (both windows)
        dom_small_vel_encoded = models_dict['velocity_feedforward'].forward(batch['dom_velocity_small'])
        check_nan(dom_small_vel_encoded, "dom_small_vel_encoded")

        dom_large_vel_encoded = models_dict['velocity_feedforward'].forward(batch['dom_velocity_large'])
        check_nan(dom_large_vel_encoded, "dom_large_vel_encoded")

        non_dom_small_vel_encoded = models_dict['velocity_feedforward'].forward(batch['non_dom_velocity_small'])
        check_nan(non_dom_small_vel_encoded, "non_dom_small_vel_encoded")

        non_dom_large_vel_encoded = models_dict['velocity_feedforward'].forward(batch['non_dom_velocity_large'])
        check_nan(non_dom_large_vel_encoded, "non_dom_large_vel_encoded")

        dom_landmarks_velocity_conc = combine_semantic_and_velocity_features(
            semantic_features=dom_landmark_embeddings, 
            velocity_small_features=dom_small_vel_encoded, 
            velocity_large_features=dom_large_vel_encoded
        )
        check_nan(dom_landmarks_velocity_conc, "dom_landmarks_velocity_conc")

        non_dom_landmarks_velocity_conc = combine_semantic_and_velocity_features(
            semantic_features=non_dom_landmark_embeddings, 
            velocity_small_features=non_dom_small_vel_encoded, 
            velocity_large_features=non_dom_large_vel_encoded
        )
        check_nan(non_dom_landmarks_velocity_conc, "non_dom_landmarks_velocity_conc")

        wrist_vel_small_encoded = models_dict['wrist_vel_feedforward'].forward(batch['nose_to_wrist_velocity_small'])
        check_nan(wrist_vel_small_encoded, "wrist_vel_small_encoded")

        wrist_vel_large_encoded = models_dict['wrist_vel_feedforward'].forward(batch['nose_to_wrist_velocity_large'])
        check_nan(wrist_vel_large_encoded, "wrist_vel_large_encoded")

        wrists_vel_conc = combine_wrist_embedding_and_velocity(
            wrist_embeddings=torch.cat([dom_wrist_embedding, non_dom_wrist_embedding], dim=-1).reshape((2,-1)), 
            wrist_velocity_small=wrist_vel_small_encoded, 
            wrist_velocity_large=wrist_vel_large_encoded
        )
        check_nan(wrists_vel_conc, "wrists_vel_conc")

        # Blendshapes encoding
        blendshapes_encoded = models_dict['blendshapes_feedforward'](batch['blendshape_scores'])
        check_nan(blendshapes_encoded, "blendshapes_encoded")

        # Spatial transformer
        dom_contextualized = models_dict['dom_transformer'](dom_landmarks_conc)
        check_nan(dom_contextualized, "dom_contextualized")

        non_dom_contextualized = models_dict['non_dom_transformer'](non_dom_landmarks_conc)
        check_nan(non_dom_contextualized, "non_dom_contextualized")

        dom_pooled = models_dict['dom_pooling'](dom_contextualized)
        check_nan(dom_pooled, "dom_pooled")

        non_dom_pooled = models_dict['non_dom_pooling'](non_dom_contextualized)
        check_nan(non_dom_pooled, "non_dom_pooled")

        dom_wrist_conc = wrists_conc[:,:,0]
        non_dom_wrist_conc = wrists_conc[:,:,1]

        dom_spatial_combined = concat_pooled_wrists(pooled=dom_pooled, wrist=dom_wrist_conc)
        check_nan(dom_spatial_combined, "dom_spatial_combined")

        non_dom_spatial_combined = concat_pooled_wrists(pooled=non_dom_pooled, wrist=non_dom_wrist_conc)
        check_nan(non_dom_spatial_combined, "non_dom_spatial_combined")

        # Velocity transformer
        dom_vel_contextualized = models_dict['dom_vel_transformer'](dom_landmarks_velocity_conc)
        check_nan(dom_vel_contextualized, "dom_vel_contextualized")

        non_dom_vel_contextualized = models_dict['non_dom_vel_transformer'](non_dom_landmarks_velocity_conc)
        check_nan(non_dom_vel_contextualized, "non_dom_vel_contextualized")

        dom_vel_pooled = models_dict['dom_vel_pooling'](dom_vel_contextualized)
        check_nan(dom_vel_pooled, "dom_vel_pooled")

        non_dom_vel_pooled = models_dict['non_dom_vel_pooling'](non_dom_vel_contextualized)
        check_nan(non_dom_vel_pooled, "non_dom_vel_pooled")

        dom_wrist_vel_conc = wrists_vel_conc[:,:,0]
        non_dom_wrist_vel_conc = wrists_vel_conc[:,:,1]

        dom_velocity_combined = concat_pooled_wrists(pooled=dom_vel_pooled, wrist=dom_wrist_vel_conc)
        check_nan(dom_velocity_combined, "dom_velocity_combined")

        non_dom_velocity_combined = concat_pooled_wrists(pooled=non_dom_vel_pooled, wrist=non_dom_wrist_vel_conc)
        check_nan(non_dom_velocity_combined, "non_dom_velocity_combined")

        # Combining spatial with velocity features
        dom_combined = concat_pooled_wrists(dom_spatial_combined, dom_velocity_combined)
        check_nan(dom_combined, "dom_combined")

        non_dom_combined = concat_pooled_wrists(non_dom_spatial_combined, non_dom_velocity_combined)
        check_nan(non_dom_combined, "non_dom_combined")

        hands_combined = torch.stack([dom_combined, non_dom_combined], dim=2)
        check_nan(hands_combined, "hands_combined")

        # Second stage transformers between the two hands
        confidence_scores = {
            'Cd_spatial': batch['confidence_scores'],
            'Ci_spatial': batch['interpolation_scores'],
            'Cd_velocity': batch['velocity_calculation_confidence'],
            'Ci_velocity': batch['velocity_confidence']
        }

        enhanced_hands = models_dict['cross_hand_transformer'](hands_combined, confidence_scores)
        check_nan(enhanced_hands, "enhanced_hands")

        # Attention pooling to keep a weighted avg of the two
        final_hands_representation = models_dict['final_pooling'](enhanced_hands)
        check_nan(final_hands_representation, "final_hands_representation")

        # Combine with blendshapes
        frame_representation = concat_pooled_wrists(final_hands_representation, blendshapes_encoded)
        check_nan(frame_representation, "frame_representation")

        # Downsample with 1d convolution
        downsampled_representation = models_dict['conv1d'](frame_representation)
        check_nan(downsampled_representation, "downsampled_representation")

        # Positional encodings + temporal transformer
        downsampled_with_positional_encoding = models_dict['positional_encoder'](
            downsampled_representation, scale=1.0
        )
        check_nan(downsampled_with_positional_encoding, "downsampled_with_positional_encoding")

        multi_scale_representation = models_dict['temporal_transformer'](
            downsampled_with_positional_encoding, 
            mask=batch['mask']
        )
        check_nan(multi_scale_representation, "multi_scale_representation")

        # Re-enforce positional encodings with smaller scale
        video_representation = models_dict['positional_encoder'](multi_scale_representation, scale=0.25)
        check_nan(video_representation, "video_representation")

        # Prepare input IDs for GPT model
        input_ids = make_inputs_for_model(L_index=batch['L_index'], tokenizer=tokenizer)

        # Final GPT model
        outputs = models_dict['model'](
            input_ids=input_ids,
            video_representations=video_representation,
            video_mask=batch["mask"],
            L_index=batch["L_index"],
            L_values=batch["L_values"],
            label_mask=batch["label_mask"]
        )
        check_nan(outputs, "outputs")
        
        return outputs, input_ids
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Set all models to training mode
            for model in model_components:
                model.train()

            # Initialize metrics
            train_loss = 0.0
            batch_count = 0

            # Progress tracking
            if verbose > 0:
                print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")

            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                # Run forward pass
                outputs, input_ids = forward_pass(batch)

                # Get the loss
                loss = outputs['loss']
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss value: {loss.item()}, skipping batch")
                    continue
                # Backward pass
                loss.backward()
                
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Replace NaN gradients with zeros
                            if torch.isnan(param.grad).any():
                                print(f"NaN gradients detected in {name}")
                                param.grad[torch.isnan(param.grad)] = 0.0

                            # Replace inf gradients
                            if torch.isinf(param.grad).any():
                                print(f"Inf gradients detected in {name}")
                                param.grad[torch.isinf(param.grad)] = 0.0

                            # Check for extreme gradients
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 10.0:  # Threshold for reporting
                                print(f"High gradient norm in {name}: {grad_norm}")

                # Gradient clipping
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip_value)

                # Optimizer step
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                batch_count += 1

                # Log progress
                if verbose > 1 and batch_idx % log_freq_batches == 0:
                    print(f"Batch {batch_idx+1}/{train_batches} - Loss: {loss.item():.4f}")

                # Check if we've processed enough batches
                if batch_idx + 1 >= train_batches:
                    break
                
            # Calculate epoch metrics
            train_loss /= max(1, batch_count)

            # Validation phase
            val_loss = 0.0
            val_min_loss = 0.0
            val_batch_count = 0
            validation_examples = []

            # Set all models to evaluation mode
            for model in model_components:
                model.eval()

            # Validation loop
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Run forward pass
                    outputs, input_ids = forward_pass(batch)

                    # Get loss
                    loss = outputs['loss']

                    # Calculate theoretical minimum loss
                    target_labels = torch.zeros(batch['L_values'].shape[0], batch['L_values'].shape[1], 
                                               tokenizer.vocab_size, device=device)
                    target_labels.scatter_(2, batch['L_index'], batch['L_values'])
                    min_loss = compute_minimum_loss(target_labels, batch['label_mask'])

                    # If this is the first few batches, get example translations
                    if batch_idx < num_examples_to_display:
                        example_comparisons = compare_predictions(input_ids, outputs['logits'], tokenizer)
                        for ex in example_comparisons[:min(len(example_comparisons), 5)]:  # Limit to 5 examples
                            validation_examples.append({
                                "epoch": epoch + 1,
                                "batch": batch_idx,
                                "input": ex["input"],
                                "generated": ex["generated"]
                            })

                    # Update metrics
                    if loss is not None:
                        val_loss += loss.item()
                    if min_loss is not None:
                        val_min_loss += min_loss.item()
                    val_batch_count += 1

                    # Check if we've processed enough batches
                    if batch_idx + 1 >= val_batches:
                        break
                    
            # Calculate validation metrics
            val_loss /= max(1, val_batch_count)
            val_min_loss /= max(1, val_batch_count)
            val_normalized_loss = val_loss / max(val_min_loss, 1e-8)  # Avoid division by zero

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_min_loss'].append(val_min_loss)
            history['val_normalized_loss'].append(val_normalized_loss)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Add examples to history
            if validation_examples:
                history['examples'].extend(validation_examples)

            # Step the scheduler if it exists
            if scheduler is not None:
                if scheduler_type == 'plateau' or scheduler_type == 'warmup_plateau':
                    scheduler.step(val_loss)  # Pass validation loss to plateau scheduler
                else:
                    scheduler.step()

            # Time taken for epoch
            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            if verbose > 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f} (Min: {val_min_loss:.4f}, Normalized: {val_normalized_loss:.4f})")
                print(f"Learning Rate: {current_lr:.6f}")
                cross_attn_scale = models_dict['model'].scale_min + (models_dict['model'].scale_max - models_dict['model'].scale_min) * torch.sigmoid(models_dict['model'].cross_attn_value)
                if isinstance(cross_attn_scale, torch.Tensor):
                    cross_attn_scale = cross_attn_scale.item()
                print(f"cross_attn_scale has value: {cross_attn_scale:.6f}")
                # Display example translations
                if validation_examples:
                    print("\nExample Translations:")
                    table_data = []
                    for i, ex in enumerate(validation_examples[-5:]):  # Show last 5 examples
                        table_data.append([f"Example {i+1}", ex["input"], ex["generated"]])

                    print(tabulate(table_data, headers=["", "Input", "Generated"], tablefmt="grid"))

            # Save checkpoint if needed
            if (epoch + 1) % save_freq_epochs == 0 or epoch == num_epochs - 1:
                if not save_best_only or val_loss < best_val_loss:
                    checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    save_checkpoint(checkpoint_path, models_dict, optimizer, scheduler, epoch, history)
                    if verbose > 0:
                        print(f"Checkpoint saved to {checkpoint_path}")

            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    best_model_path = os.path.join(run_dir, "best_model.pt")
                    save_checkpoint(best_model_path, models_dict, optimizer, scheduler, epoch, history)
                    if verbose > 0:
                        print(f"New best model saved with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"Early stopping patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                        break
    except Exception as e:
        if auto_rescue:
            print(f"\nTraining interrupted by exception: {e}")
            save_rescue_checkpoint()
        raise  # Re-raise the exception after saving

    finally:
        # Always save a final checkpoint regardless of how training ended
        if auto_rescue:
            final_path = os.path.join(run_dir, "final_state_checkpoint.pt")
            save_checkpoint(final_path, models_dict, optimizer, scheduler, epoch, history)
            if verbose > 0:
                print(f"\nFinal state saved to {final_path}")

    # Training complete
    if verbose > 0:
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save history to JSON
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        # Convert any non-serializable objects
        serializable_history = {k: v for k, v in history.items() if k != 'examples'}
        serializable_history['examples'] = history['examples']  # These should already be serializable
        
        # Convert NumPy arrays to lists
        for key in serializable_history:
            if isinstance(serializable_history[key], list) and serializable_history[key] and isinstance(serializable_history[key][0], np.ndarray):
                serializable_history[key] = [x.tolist() for x in serializable_history[key]]
        
        json.dump(serializable_history, f, indent=2)
    
    # Plot training history
    plot_history(history, os.path.join(run_dir, "training_history.png"))
    
    return history

def resume_training_from_checkpoint(
    checkpoint_path,
    train_loader,
    val_loader,
    train_batches,
    val_batches,
    models_dict,
    tokenizer,
    learning_rate=None,  # Will use from checkpoint if None
    num_epochs=30,       # Additional epochs to train
    **kwargs             # Other training parameters
):
    """Resume training from a checkpoint"""
    import torch
    import torch.optim as optim
    
    device = kwargs.get('device', 'cuda')
    
    # Create optimizer shell (will be overwritten)
    all_params = []
    for model in models_dict.values():
        all_params.extend(model.parameters())
    temp_optimizer = optim.Adam(all_params, lr=learning_rate or 1e-4)
    
    # Load checkpoint
    epoch, history, extra_info = load_checkpoint(
        checkpoint_path, models_dict, temp_optimizer, None, device=device
    )
    
    if extra_info and 'interrupted_at_batch' in extra_info:
        print(f"Resuming from checkpoint saved at epoch {epoch+1}, batch {extra_info['interrupted_at_batch']+1}")
        if 'exception' in extra_info:
            print(f"Previous training was interrupted by exception:\n{extra_info['exception']}")
    else:
        print(f"Resuming from checkpoint saved at epoch {epoch+1}")
    
    # Use the learning rate from checkpoint if not specified
    if learning_rate is None:
        learning_rate = temp_optimizer.param_groups[0]['lr']
        print(f"Using learning rate from checkpoint: {learning_rate}")
    
    # Train for additional epochs
    print(f"Training for {num_epochs} additional epochs")
    
    # Resume training with the loaded state
    return train_asl_model(
        train_loader=train_loader,
        val_loader=val_loader,
        train_batches=train_batches,
        val_batches=val_batches,
        models_dict=models_dict,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        **kwargs
    )

def save_checkpoint(path, models_dict, optimizer, scheduler, epoch, history, extra_info=None):
    """Save a checkpoint with all model states and training information."""
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'history': {k: v for k, v in history.items() if k != 'examples'},  # Don't save examples
        'timestamp': datetime.now().isoformat()
    }

    if extra_info is not None:
        state_dict['extra_info'] = extra_info


    # Add scheduler state if it exists
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()
    
    # Add model states
    for name, model in models_dict.items():
        state_dict[f'model_{name}'] = model.state_dict()
    
    torch.save(state_dict, path)


def load_checkpoint(path, models_dict, optimizer=None, scheduler=None):
    """Load a checkpoint into models and training state."""
    checkpoint = torch.load(path, map_location=device)
    
    # Load models
    for name, model in models_dict.items():
        model_key = f'model_{name}'
        if model_key in checkpoint:
            model.load_state_dict(checkpoint[model_key])
    
    # Load optimizer if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Return epoch and history
    return (
        checkpoint.get('epoch', -1), 
        checkpoint.get('history', {}),
        checkpoint.get('extra_info', None)
    )


def plot_history(history, save_path=None):
    """Plot training history metrics."""
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    if 'val_min_loss' in history and history['val_min_loss']:
        plt.plot(history['val_min_loss'], label='Theoretical Min Loss', linestyle='--')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Normalized loss plot
    plt.subplot(2, 2, 2)
    if 'val_normalized_loss' in history and history['val_normalized_loss']:
        plt.plot(history['val_normalized_loss'], label='Normalized Val Loss')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Theoretical Minimum')
    plt.title('Normalized Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Ratio')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')
    
    # Empty subplot or future metric
    plt.subplot(2, 2, 4)
    plt.title('Reserved for Future Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



class WarmupToPlateauScheduler(ReduceLROnPlateau):
    """
    Scheduler that combines warm-up with ReduceLROnPlateau behavior.
    Starts with a low learning rate that increases linearly to the base learning rate
    over a specified number of epochs, then behaves like ReduceLROnPlateau.
    """
    def __init__(self, optimizer, warmup_epochs=5, warmup_start_factor=0.01, 
                 mode='min', factor=0.5, patience=5, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=0, verbose=False):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of epochs for warm-up phase
            warmup_start_factor: Initial LR will be base_lr * warmup_start_factor
            mode: 'min' or 'max' for metric monitoring
            factor: Factor by which to reduce LR after plateau
            patience: Epochs to wait before reducing LR
            threshold: Threshold for measuring improvement
            threshold_mode: How threshold is applied
            cooldown: Epochs to wait before resuming normal operation
            min_lr: Minimum allowable learning rate
            verbose: Print messages on update
        """
        # Initialize the ReduceLROnPlateau parent
        super(WarmupToPlateauScheduler, self).__init__(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, threshold_mode=threshold_mode,
            cooldown=cooldown, min_lr=min_lr, verbose=verbose
        )
        
        # Store additional parameters for warm-up
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
        # Set initial learning rates to the warm-up starting values
        self._set_initial_lrs()
    
    def _set_initial_lrs(self):
        """Set the initial learning rates to the warm-up starting values."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * self.warmup_start_factor
    
    def step(self, metrics=None, epoch=None):
        """
        Step the scheduler based on the current epoch and metrics.
        
        Args:
            metrics: Validation metrics to monitor (used in plateau phase)
            epoch: Current epoch number (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        # Handle warm-up phase
        if self.current_epoch < self.warmup_epochs:
            # Calculate the fraction of warm-up completed
            warmup_progress = (self.current_epoch + 1) / self.warmup_epochs
            
            # Adjust learning rate based on warm-up progress
            for i, param_group in enumerate(self.optimizer.param_groups):
                # Start at warmup_start_factor * base_lr and increase to base_lr
                start_lr = self.base_lrs[i] * self.warmup_start_factor
                end_lr = self.base_lrs[i]
                
                # Linear warm-up formula
                param_group['lr'] = start_lr + warmup_progress * (end_lr - start_lr)
                
            self.current_epoch += 1
            return
        
        # After warm-up, use standard ReduceLROnPlateau behavior
        super(WarmupToPlateauScheduler, self).step(metrics)

def main():
    try:
        print("Initializing architecture")
        device = torch.device('cuda')
        embedding_dim = 30
        embedding_table = LandmarkEmbedding(embedding_dim=embedding_dim, num_landmarks_per_hand=21)
        embedding_table.to(device)


        landmark_encoder_hidden_dims = [30, 30, 30] 
        landmark_encoder_activation = 'gelu'
        landmark_encoder_init_method = 'xavier_uniform'
        landmark_encoder_init_gain=0.8
        landmark_encoder = LandmarkSpatialEncoder(embedding_dim, hidden_dims=landmark_encoder_hidden_dims, activation=landmark_encoder_activation, init_method=landmark_encoder_init_method, init_gain=landmark_encoder_init_gain)
        landmark_encoder.to(device)


        wrist_encoder_hidden_dims = [30, 60, 30] 
        wrist_encoder_activation = 'gelu'
        wrist_encoder_init_method = 'xavier_uniform'
        wrist_encoder_init_gain=0.8
        wrist_encoder = WristSpatialEncoder(embedding_dim, hidden_dims=wrist_encoder_hidden_dims,activation=wrist_encoder_activation,init_method=wrist_encoder_init_method, init_gain=wrist_encoder_init_gain)
        wrist_encoder.to(device)

        blendshapes_encoder_hidden_dims = [60, 60, 60, 60] 
        blendshapes_encoder_activation = 'gelu'
        blendshapes_encoder_init_method = 'xavier_uniform'
        blendshapes_encoder_init_gain=0.8
        blendshapes_feedforward = BlendshapeEncoder(embedding_dim, hidden_dims=blendshapes_encoder_hidden_dims, activation=blendshapes_encoder_activation,init_method=blendshapes_encoder_init_method, init_gain=blendshapes_encoder_init_gain)
        blendshapes_feedforward.to(device)


        velocity_encoder_hidden_dims = [30, 30, 30, 30] 
        velocity_encoder_activation = 'gelu'
        velocity_encoder_init_method = 'xavier_uniform'
        velocity_encoder_init_gain=0.8
        velocity_feedforward = VelocityEncoder(n_velocity_encoding=2*embedding_dim, hidden_dims=velocity_encoder_hidden_dims, activation=velocity_encoder_activation, init_method=velocity_encoder_init_method, init_gain=velocity_encoder_init_gain)
        velocity_feedforward.to(device)


        wrist_vel_encoder_hidden_dims = [30, 30, 30, 30] 
        wrist_vel_encoder_activation = 'gelu'
        wrist_vel_encoder_init_method = 'xavier_uniform'
        wrist_vel_encoder_init_gain=0.8
        wrist_vel_feedforward = WristVelocityEncoder(n_velocity_encoding=2*embedding_dim, hidden_dims=wrist_vel_encoder_hidden_dims, activation=wrist_vel_encoder_activation, init_method=wrist_vel_encoder_init_method, init_gain=wrist_vel_encoder_init_gain)
        wrist_vel_feedforward.to(device)


        first_stage_transformer_num_layers = 5
        first_stage_transformer_num_heads = 8
        first_stage_transformer_hidden_dim = 256 #Output dimensionality
        first_stage_transformer_ff_dim=4*first_stage_transformer_hidden_dim #Feedforward networks hidden dimension
        first_stage_transformer_activation = 'gelu'
        first_stage_transformer_init_method = 'xavier_uniform'
        first_stage_transformer_prenorm = True
        first_stage_transformer_init_gain=0.8
        dom_transformer = LandmarkTransformerEncoder(input_dim=3 * embedding_dim, num_layers=first_stage_transformer_num_layers, num_heads=first_stage_transformer_num_heads, hidden_dim=first_stage_transformer_hidden_dim, ff_dim=first_stage_transformer_ff_dim, activation=first_stage_transformer_activation, prenorm=first_stage_transformer_prenorm, init_method=first_stage_transformer_init_method, init_gain=first_stage_transformer_init_gain)
        non_dom_transformer = LandmarkTransformerEncoder(input_dim=3 * embedding_dim, num_layers=first_stage_transformer_num_layers, num_heads=first_stage_transformer_num_heads, hidden_dim=first_stage_transformer_hidden_dim, ff_dim=first_stage_transformer_ff_dim, activation=first_stage_transformer_activation, prenorm=first_stage_transformer_prenorm, init_method=first_stage_transformer_init_method, init_gain=first_stage_transformer_init_gain)
        dom_transformer.to(device)
        non_dom_transformer.to(device)


        first_stage_pooling_output_dim=256
        first_stage_pooling_init_method='xavier_uniform'
        first_stage_pooling_gain=0.8
        first_stage_pooling_bias_init='zeros'
        dom_pooling = LandmarkAttentionPooling(input_dim=dom_transformer.hidden_dim,output_dim=first_stage_pooling_output_dim, init_method=first_stage_pooling_init_method, gain=first_stage_pooling_gain, bias_init=first_stage_pooling_bias_init)
        non_dom_pooling = LandmarkAttentionPooling(input_dim=non_dom_transformer.hidden_dim,output_dim=first_stage_pooling_output_dim, init_method=first_stage_pooling_init_method, gain=first_stage_pooling_gain, bias_init=first_stage_pooling_bias_init)
        dom_pooling.to(device)
        non_dom_pooling.to(device)


        first_stage_velocity_transformer_num_layers = 5
        first_stage_velocity_transformer_num_heads = 8
        first_stage_velocity_transformer_hidden_dim = 256 #Output dimensionality
        first_stage_velocity_transformer_ff_dim=4*first_stage_velocity_transformer_hidden_dim #Feedforward networks hidden dimension
        first_stage_velocity_transformer_activation = 'gelu'
        first_stage_velocity_transformer_init_method = 'xavier_uniform'
        first_stage_velocity_transformer_prenorm = True
        first_stage_velocity_transformer_init_gain=0.8
        dom_vel_transformer = LandmarkTransformerEncoder(input_dim=velocity_feedforward.output_dim*2+embedding_dim, num_layers=first_stage_velocity_transformer_num_layers, num_heads=first_stage_velocity_transformer_num_heads, hidden_dim=first_stage_velocity_transformer_hidden_dim, ff_dim=first_stage_velocity_transformer_ff_dim, activation=first_stage_velocity_transformer_activation, prenorm=first_stage_velocity_transformer_prenorm, init_method=first_stage_velocity_transformer_init_method, init_gain=first_stage_velocity_transformer_init_gain)
        non_dom_vel_transformer = LandmarkTransformerEncoder(input_dim=velocity_feedforward.output_dim*2+embedding_dim, num_layers=first_stage_velocity_transformer_num_layers, num_heads=first_stage_velocity_transformer_num_heads, hidden_dim=first_stage_velocity_transformer_hidden_dim, ff_dim=first_stage_velocity_transformer_ff_dim, activation=first_stage_velocity_transformer_activation, prenorm=first_stage_velocity_transformer_prenorm, init_method=first_stage_velocity_transformer_init_method, init_gain=first_stage_velocity_transformer_init_gain)
        dom_vel_transformer.to(device)
        non_dom_vel_transformer.to(device)


        first_stage_velocity_pooling_output_dim=256
        first_stage_velocity_pooling_init_method='xavier_uniform'
        first_stage_velocity_pooling_gain=0.8
        first_stage_velocity_pooling_bias_init='zeros'
        dom_vel_pooling = LandmarkAttentionPooling(input_dim=dom_vel_transformer.hidden_dim,output_dim=first_stage_velocity_pooling_output_dim, init_method=first_stage_velocity_pooling_init_method, gain=first_stage_velocity_pooling_gain, bias_init=first_stage_velocity_pooling_bias_init)
        non_dom_vel_pooling = LandmarkAttentionPooling(input_dim=non_dom_vel_transformer.hidden_dim,output_dim=first_stage_velocity_pooling_output_dim, init_method=first_stage_velocity_pooling_init_method, gain=first_stage_velocity_pooling_gain, bias_init=first_stage_velocity_pooling_bias_init)
        dom_vel_pooling.to(device)
        non_dom_vel_pooling.to(device)

        input_dim_for_cross_hand_transformer = dom_pooling.output_dim + 4*embedding_dim+dom_vel_pooling.output_dim+2*wrist_vel_feedforward.output_dim

        second_stage_transformer_num_layers = 4
        second_stage_transformer_num_heads = 8
        second_stage_transformer_hidden_dim = input_dim_for_cross_hand_transformer #Output dimensionality
        second_stage_transformer_ff_dim=4*second_stage_transformer_hidden_dim #Feedforward networks hidden dimension
        second_stage_transformer_activation = 'gelu'
        second_stage_transformer_init_method = 'xavier_uniform'
        second_stage_transformer_prenorm = True
        second_stage_transformer_init_gain=0.8
        cross_hand_transformer = ConfidenceWeightedTransformerEncoder(
            input_dim=input_dim_for_cross_hand_transformer,
            num_layers=second_stage_transformer_num_layers,
            num_heads=second_stage_transformer_num_heads,
            hidden_dim=second_stage_transformer_hidden_dim,
            ff_dim=second_stage_transformer_ff_dim,
            prenorm=second_stage_transformer_prenorm,
            activation=second_stage_transformer_activation,
            init_method=second_stage_transformer_init_method,
            init_gain=second_stage_transformer_init_gain
        )
        cross_hand_transformer.to(device)


        final_pooling_output_dim=second_stage_transformer_hidden_dim
        final_pooling_init_method='xavier_uniform'
        final_pooling_gain=0.8
        final_pooling_bias_init='zeros'
        final_pooling = LandmarkAttentionPooling(
            input_dim=second_stage_transformer_hidden_dim,
            output_dim=final_pooling_output_dim, init_method=final_pooling_init_method, gain=final_pooling_gain, bias_init=final_pooling_bias_init)
        final_pooling.to(device)


        stride = 2
        number_of_filters = 768 #Output dim
        kernel_size = 5
        convolution_activation = 'gelu'
        convolution_norm_layer = True
        convolution_init_method='xavier_uniform'
        convolution_gain=0.8
        convolution_bias_init='zeros'
        conv1d = TemporalDownsampler(
            input_dim=input_dim_for_cross_hand_transformer + 2*embedding_dim,          # Feature dimension (d)
            output_channels=number_of_filters,    
            kernel_size=kernel_size,          
            stride=stride,               
            activation=convolution_activation,      
            norm_layer=convolution_norm_layer,
            init_method=convolution_init_method,
            gain=convolution_gain,
            bias_init=convolution_bias_init        
        )
        conv1d.to(device)


        max_possible_number_of_frames = 120
        positional_encoder = PositionalEncoding(
            d_model=number_of_filters,  # Feature dimension
            max_len=int(max_possible_number_of_frames/stride)  
        )
        positional_encoder.to(device)

        
        temporal_transformer_num_layers = 6
        temporal_transformer_hidden_dim = input_dim_for_cross_hand_transformer #Output dimensionality
        temporal_transformer_ff_dim=4*second_stage_transformer_hidden_dim #Feedforward networks hidden dimension
        temporal_transformer_activation ='gelu'
        temporal_transformer_init_method='xavier_uniform',
        temporal_transformer_gain=0.8,
        temporal_transformer_bias_init='zeros'
        temporal_transformer = MultiScaleTemporalTransformer(
            d_model=number_of_filters,
            num_layers=temporal_transformer_num_layers,
            short_range=5,
            medium_range=15,
            long_range=45,
            dim_feedforward=temporal_transformer_ff_dim,
            activation=temporal_transformer_activation,
            stride=stride,
            init_method=temporal_transformer_init_method,
            gain=temporal_transformer_gain,
            bias_init=temporal_transformer_bias_init
        )
        temporal_transformer.to(device)


        GPT_model_name = "distilgpt2" 
        model_cross_attention_heads = 12
        freeze_GPT_model_weights=True
        trainable_GPT_model_lm_head=True
        trainable_GPT_final_ln=True
        GPT_adapter_dim=368
        
        #cross_attention_init_method='xavier_uniform'
        #cross_attention_gain=0.8,
        #cross_attention_bias_init='zeros'
        print("Before innit")
        model = VideoGPT(
            model_name=GPT_model_name,
            num_cross_heads=model_cross_attention_heads,
            freeze_gpt=freeze_GPT_model_weights,
            trainable_lm_head=trainable_GPT_model_lm_head,
            trainable_final_ln=trainable_GPT_final_ln,
            adapter_dim=GPT_adapter_dim,
            stride=stride,
        )
        model = model.to(device)
        print("After innit")

        tokenizer = AutoTokenizer.from_pretrained(GPT_model_name)
        tokenizer.pad_token = tokenizer.eos_token


        models_dict = {
            'embedding_table': embedding_table,
            'landmark_encoder': landmark_encoder,
            'wrist_encoder': wrist_encoder,
            'blendshapes_feedforward': blendshapes_feedforward,
            'velocity_feedforward': velocity_feedforward,
            'wrist_vel_feedforward': wrist_vel_feedforward,
            'dom_transformer': dom_transformer,
            'non_dom_transformer': non_dom_transformer,
            'dom_pooling': dom_pooling,
            'non_dom_pooling': non_dom_pooling,
            'dom_vel_transformer': dom_vel_transformer,
            'non_dom_vel_transformer': non_dom_vel_transformer,
            'dom_vel_pooling': dom_vel_pooling,
            'non_dom_vel_pooling': non_dom_vel_pooling,
            'cross_hand_transformer': cross_hand_transformer,
            'final_pooling': final_pooling,
            'conv1d': conv1d,
            'positional_encoder': positional_encoder,
            'temporal_transformer': temporal_transformer,
            'model': model
        }
        print("Loading datasets")
        train_high_df = pd.read_csv("./high_train_only_path.csv")
        val_high_df = pd.read_csv("./high_val_only_path.csv")


        train_mid_df = pd.read_csv("./mid_train_only_path.csv")
        val_mid_df = pd.read_csv("./mid_val_only_path.csv")

        train_low_df = pd.read_csv("./low_train_only_path.csv")
        val_low_df = pd.read_csv("./low_val_only_path.csv")

        print("Creating data loaders")
        # Create data loaders
        train_loader, train_expected_batches = create_asl_dataloader(
            low_df=train_low_df, 
            mid_df=train_mid_df, 
            high_df=train_high_df,
            batch_size=8,
            num_workers=0,
            device='cuda'
        )

        val_loader, val_expected_batches = create_asl_dataloader(
            low_df=val_low_df, 
            mid_df=val_mid_df, 
            high_df=val_high_df,
            batch_size=8,
            num_workers=0,
            device='cuda'
        )

        # Start training
        print("Begin training")
        history = train_asl_model(
            train_loader=train_loader,
            val_loader=val_loader,
            train_batches=train_expected_batches,
            val_batches=val_expected_batches,
            models_dict=models_dict,
            tokenizer=tokenizer,
            learning_rate=1e-3,
            weight_decay=1e-3,
            num_epochs=30,
            grad_clip_value=1.0,
            scheduler_type='warmup_plateau',
            scheduler_params={'warmup_epochs': 3,'warmup_start_factor': 0.01,'mode': 'min', 'factor': 0.5, 'min_lr':1e-7 ,'patience': 5, 'verbose': False},
            early_stopping=True,
            patience=15,
            min_delta=0.001,
            verbose=2, # Detailed logging
            save_dir='./checkpoints',
            save_best_only=False,
            save_freq_epochs=2,
            auto_rescue=True,
            # Validation and visualization
            num_examples_to_display=10,  # Number of examples to display during validation

            # Logging
            log_freq_batches=50,

            # Device
            device='cuda'
        )
        print("Training finished")

  


        

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()