# verify_model_size.py

import torch
from torchinfo import summary

# Import all the components from your model file
from model import (
    FeatureExtractor,
    AttentionBlock,
    AttentionDecoder,
    MultiContrastiveModel
)

# --- Configuration ---
# These values must match the architecture defined in model.py
PROJECTION_DIM = 64
NUM_CONTRASTS = 4 # A typical number for analysis
BATCH_SIZE = 1 # We analyze the size for a single item batch

# --- 1. Analyze the Feature Extractor ---
print("\n" + "="*80)
print("Analyzing: FeatureExtractor (SlimUNet Encoder)")
print("="*80)
# Initialize the component
feature_extractor = FeatureExtractor("unet_lesion_segmentation.pth")
# Create a dummy input tensor for a single image
dummy_input_image = torch.randn(BATCH_SIZE, 1, 64, 64)
# Print the summary
summary(feature_extractor, input_data=dummy_input_image, depth=3, col_names=["output_size", "num_params", "params_percent", "mult_adds"])


# --- 2. Analyze the Attention Block & Projections ---
print("\n" + "="*80)
print("Analyzing: AttentionBlock (with Projections)")
print("="*80)
# Initialize the component
attention_block = AttentionBlock(embed_dim=PROJECTION_DIM)
# Create a dummy input tensor representing pooled & projected features from N contrasts
dummy_attention_input = torch.randn(BATCH_SIZE, NUM_CONTRASTS, PROJECTION_DIM)
# Print the summary
summary(attention_block, input_data=dummy_attention_input, depth=3, col_names=["output_size", "num_params", "params_percent", "mult_adds"])


# --- 3. Analyze the Decoder ---
print("\n" + "="*80)
print("Analyzing: AttentionDecoder (SlimUNet Decoder)")
print("="*80)
# Initialize the component
decoder = AttentionDecoder()
# Create dummy input tensors representing the bottleneck and skip connections
dummy_bottleneck = torch.randn(BATCH_SIZE, 256, 8, 8)
dummy_skips = [
    torch.randn(BATCH_SIZE, 32, 64, 64),
    torch.randn(BATCH_SIZE, 64, 32, 32),
    torch.randn(BATCH_SIZE, 128, 16, 16)
]
# Print the summary
summary(decoder, input_data=[dummy_bottleneck, dummy_skips], depth=3, col_names=["output_size", "num_params", "params_percent", "mult_adds"])


# --- 4. Analyze the Full Assembled Model ---
print("\n" + "="*80)
print("Analyzing: Full MultiContrastiveModel")
print("="*80)
# Initialize the full model
full_model = MultiContrastiveModel(
    feature_extractor=feature_extractor,
    attention_block=attention_block,
    decoder=decoder,
    projection_dim=PROJECTION_DIM
)
# Create dummy inputs for the full model's forward pass
dummy_full_input = torch.randn(BATCH_SIZE, NUM_CONTRASTS, 1, 64, 64)
dummy_primary_idx = torch.zeros(BATCH_SIZE, dtype=torch.long)
# Print the summary
summary(full_model, input_data=[dummy_full_input, dummy_primary_idx], depth=4, col_names=["output_size", "num_params", "params_percent", "mult_adds"])

