# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The UNet building blocks can be simplified and reused ---
class DoubleConv(nn.Module):
    """(convolution => => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to the size of x2
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionFusion(nn.Module):
    """
    Learns to adaptively fuse a variable number of feature maps using an
    independent, channel-wise attention mechanism (Sigmoid).
    This encourages collaborative fusion.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        # Attention network to compute independent weights for each modality
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # Use Sigmoid for independent [0, 1] weights
        )

    def forward(self, feature_maps_list):
        num_contrasts = len(feature_maps_list)
        if num_contrasts == 1:
            return feature_maps_list[0]

        stacked_features = torch.stack(feature_maps_list, dim=1)
        B, N, C, H, W = stacked_features.shape

        # Reshape for parallel processing
        reshaped_features = stacked_features.view(B * N, C, H, W)
        
        # --- KEY CHANGE ---
        # Compute independent attention weights for each contrast's feature map
        attention_weights = self.attention_net(reshaped_features)
        
        # Reshape weights back to (B, N, C, 1, 1)
        attention_weights = attention_weights.view(B, N, C, 1, 1)
        
        # Apply weights and then sum the features. This is now a collaborative weighting.
        weighted_features = stacked_features * attention_weights
        fused_features = torch.sum(weighted_features, dim=1)
        
        return fused_features
    
    
class BoundaryAwareUNet(nn.Module):
    """
    A multi-task U-Net that simultaneously predicts a segmentation mask and its boundary.
    It uses a shared encoder and decoder, with two separate output heads.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # --- Encoder and Fusion modules remain the same ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.fuse_skip1 = AttentionFusion(64)
        self.fuse_skip2 = AttentionFusion(128)
        self.fuse_skip3 = AttentionFusion(256)
        self.fuse_skip4 = AttentionFusion(512)
        self.fuse_bottleneck = AttentionFusion(1024)

        # --- Decoder remains the same ---
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # --- NEW: Two separate output heads for multi-task learning ---
        self.outc_mask = OutConv(64, out_channels)
        self.outc_boundary = OutConv(64, out_channels)

    def forward(self, contrasts):
        B, N, C, H, W = contrasts.shape
        skips1, skips2, skips3, skips4, bottlenecks =[],[],[],[],[]

        for i in range(N):
            contrast_slice = contrasts[:, i, :, :, :]
            s1 = self.inc(contrast_slice)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            bottleneck = self.down4(s4)
            skips1.append(s1); skips2.append(s2); skips3.append(s3); skips4.append(s4); bottlenecks.append(bottleneck)

        fused_s1 = self.fuse_skip1(skips1)
        fused_s2 = self.fuse_skip2(skips2)
        fused_s3 = self.fuse_skip3(skips3)
        fused_s4 = self.fuse_skip4(skips4)
        fused_bottleneck = self.fuse_bottleneck(bottlenecks)

        x = self.up1(fused_bottleneck, fused_s4)
        x = self.up2(x, fused_s3)
        x = self.up3(x, fused_s2)
        # This is the shared feature map before the final output layers
        shared_decoder_output = self.up4(x, fused_s1)

        # --- NEW: Generate predictions from each head ---
        mask_logits = self.outc_mask(shared_decoder_output)
        boundary_logits = self.outc_boundary(shared_decoder_output)
        
        # Return a dictionary for clear, multi-task outputs
        return {"mask": mask_logits, "boundary": boundary_logits}
    
class AlignmentAndFusionUNet(nn.Module):
    """
    A wrapper model that first performs learnable, task-driven registration of
    moving contrasts to a reference contrast, and then feeds the aligned stack
    into a segmentation network.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # --- 1. Registration Head (The STN's Localization Network) ---
        # This small CNN predicts the 6 parameters of an affine transformation matrix.
        # It takes a pair of images (reference + one moving) as input.
        self.registration_head = nn.Sequential(
            DoubleConv(in_channels * 2, 32),
            nn.MaxPool2d(2),
            DoubleConv(32, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Output 6 parameters for the 2x3 affine matrix
        )
        # Initialize the final layer to output an identity transformation
        self.registration_head[-1].weight.data.zero_()
        self.registration_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # --- 2. The Core Segmentation Network ---
        # We use the previously defined BoundaryAwareUNet for the main task.
        self.segmentation_network = BoundaryAwareUNet(in_channels, out_channels)

    def spatial_transformer(self, moving_image, theta):
        """
        Applies the predicted affine transformation `theta` to the `moving_image`.
        This is the warping part of the STN.
        """
        # `theta` has shape, reshape to for the grid sampler
        theta = theta.view(-1, 2, 3)
        # Generate the sampling grid
        grid = F.affine_grid(theta, moving_image.size(), align_corners=False)
        # Sample the moving image using the grid to get the warped image
        warped_image = F.grid_sample(moving_image, grid, align_corners=False)
        return warped_image

    def forward(self, reference_contrast, moving_contrasts):
        """
        Args:
            reference_contrast (torch.Tensor): The fixed image, shape.
            moving_contrasts (list of torch.Tensor): A list of N moving images,
                                                     each of shape.
        """
        aligned_contrasts = [reference_contrast]

        # --- Part 1: Align each moving contrast to the reference ---
        for moving_contrast in moving_contrasts:
            # Concatenate reference and moving images to feed to the registration head
            registration_input = torch.cat([reference_contrast, moving_contrast], dim=1)
            
            # Predict the transformation parameters
            theta = self.registration_head(registration_input)
            
            # Warp the moving contrast using the predicted parameters
            warped_moving_contrast = self.spatial_transformer(moving_contrast, theta)
            
            aligned_contrasts.append(warped_moving_contrast)

        # --- Part 2: Fuse and Segment the aligned stack ---
        # Stack the aligned contrasts into a single tensor for the segmentation network
        final_input_stack = torch.stack(aligned_contrasts, dim=1) # Shape
        
        # Pass the perfectly aligned stack to the segmentation network
        return self.segmentation_network(final_input_stack)