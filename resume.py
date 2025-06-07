import torch
import pandas as pd
from transformers import AutoTokenizer

from translation_model_functions import (
    LandmarkEmbedding,
    LandmarkSpatialEncoder,
    WristSpatialEncoder,
    BlendshapeEncoder,
    VelocityEncoder,
    WristVelocityEncoder,
    LandmarkTransformerEncoder,
    LandmarkAttentionPooling,
    ConfidenceWeightedTransformerEncoder,
    TemporalDownsampler,
    PositionalEncoding,
    MultiScaleTemporalTransformer,
    VideoGPT_new,
    create_asl_dataloader,
    resume_training_from_checkpoint
)

def main():
    
    checkpoint_path = "./checkpoints/run_20250604_161812/checkpoint_epoch_2.pt"
    
    print(f"Will resume training from: {checkpoint_path}")
    
    # Set device
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Recreate exactly the same model architecture as in train.py
    print("Recreating model architecture (weights will be loaded from checkpoint)...")

    try:
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
        model_attention_heads = 12
        GPT_number_of_blocks=5
        freeze_GPT_model_weights=True
        model = VideoGPT_new(
            model_name=GPT_model_name,
            num_cross_heads=model_attention_heads,
            number_of_blocks=GPT_number_of_blocks,
            freeze_gpt=freeze_GPT_model_weights,
            stride=stride
        )
        model = model.to(device)


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

        train_high_df = pd.read_csv("./high_train_only_path.csv")
        val_high_df = pd.read_csv("./high_val_only_path.csv")


        train_mid_df = pd.read_csv("./mid_train_only_path.csv")
        val_mid_df = pd.read_csv("./mid_val_only_path.csv")

        train_low_df = pd.read_csv("./low_train_only_path.csv")
        val_low_df = pd.read_csv("./low_val_only_path.csv")


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


        print(f"Resuming training from checkpoint: {checkpoint_path}")
        history = resume_training_from_checkpoint(
            checkpoint_path=checkpoint_path,
            train_loader=train_loader,
            val_loader=val_loader,
            train_batches=train_expected_batches,
            val_batches=val_expected_batches,
            models_dict=models_dict,
            tokenizer=tokenizer,
            learning_rate=0.0001,  
            load_scheduler_and_optimizer=False, #If true will try to load optimizer and scheduler, then no need to set learning_rate, weight_decay, scheduler_type, scheduler_params
            grad_clip_value=1.0,
            weight_decay=0,  
            num_epochs=6,       # Train for n more epochs
            scheduler_type='plateau',  
            scheduler_params={'mode': 'min', 'factor': 0.1, 'patience': 2, 'verbose': False},
            early_stopping=True,
            patience=3,
            min_delta=0.001,
            verbose=2, # Detailed logging
            save_dir='./checkpoints',
            save_best_only=False,
            save_freq_epochs=2,
            auto_rescue=True,
            # Validation and visualization
            num_examples_to_display=10,  # Number of examples to display during validation

            # Logging
            log_freq_batches=20,

            # Device
            device='cuda'
        )
        

        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()