"""
CNN Encoders for Visual Feature Extraction in Atari Environments.
"""

import torch
import torch.nn as nn
import numpy as np


class NatureCNN(nn.Module):
    """
    Architecture reduces 28,224 dims -> 512 dims (55x compression)
    """
    
    def __init__(self, in_channels=4, output_dim=512):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.expected_obs_size = 84 * 84 * in_channels  # 28224 for 4 frames
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Conv1: 84x84x4 -> 20x20x32
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=False),
            
            # Conv2: 20x20x32 -> 9x9x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
            
            # Conv3: 9x9x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
        )
        
        # Calculate flattened size: 7 * 7 * 64 = 3136
        self.flatten_size = 7 * 7 * 64
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, output_dim),
            nn.ReLU(inplace=False)
        )
        
        print(f"[NatureCNN] Created: {in_channels} channels -> {output_dim} dims")
        print(f"[NatureCNN] Compression: 28224 -> {output_dim} ({28224/output_dim:.1f}x)")
        
    def forward(self, x):
        # Handle flattened input
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            
            # CRITICAL: If input has extra dimensions (e.g., concatenated with option),
            # extract only the observation part (first self.expected_obs_size elements)
            if x.shape[1] > self.expected_obs_size:
                x = x[:, :self.expected_obs_size]
            
            # Reshape to (batch, channels, height, width)
            x = x.view(batch_size, self.in_channels, 84, 84)
        
        # Convolutional layers
        x = self.conv(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected
        features = self.fc(x)
        
        return features


class ImpalaCNN(nn.Module):
    """
    IMPALA-style CNN with residual blocks (Espeholt et al., 2018).
    
    Deeper architecture for more complex visual feature extraction.
    Better for environments requiring fine-grained visual understanding.
    """
    
    def __init__(self, in_channels=4, output_dim=512):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.expected_obs_size = 84 * 84 * in_channels  # 28224 for 4 frames
        
        def residual_block(in_channels, out_channels):
            """Create a residual block with two conv layers."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            residual_block(16, 16),
            residual_block(16, 16),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            residual_block(32, 32),
            residual_block(32, 32),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            residual_block(32, 32),
            residual_block(32, 32),
        )
        
        # After 3 pooling layers: 84 -> 42 -> 21 -> 11
        self.flatten_size = 32 * 11 * 11  # 3872
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, output_dim),
            nn.ReLU(inplace=False)
        )
        
        print(f"[ImpalaCNN] Created: {in_channels} channels -> {output_dim} dims")
        print(f"[ImpalaCNN] Deeper architecture with residual blocks")
        
    def forward(self, x):
        """Forward pass through IMPALA CNN."""
        # Handle flattened input
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            
            # CRITICAL: If input has extra dimensions (e.g., concatenated with option),
            # extract only the observation part (first self.expected_obs_size elements)
            if x.shape[1] > self.expected_obs_size:
                x = x[:, :self.expected_obs_size]
            
            # Reshape to (batch, channels, height, width)
            x = x.view(batch_size, self.in_channels, 84, 84)
        
        # Convolutional stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = x.reshape(x.size(0), -1)
        features = self.fc(x)
        
        return features


class FramePreprocessor:
    """
    Atari frame preprocessing utilities.
    
    Converts RGB frames to stacked grayscale frames suitable for CNN input.
    """
    
    def __init__(self, frame_stack=4, frame_size=84):
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = []
        
    def reset(self):
        self.frames = []
        
    def process_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        else:
            gray = frame
        
        from scipy.ndimage import zoom
        h, w = gray.shape
        zoom_h = self.frame_size / h
        zoom_w = self.frame_size / w
        resized = zoom(gray, (zoom_h, zoom_w), order=1)

        normalized = resized / 255.0
        
        return normalized.astype(np.float32)
    
    def add_frame(self, frame):
        processed = self.process_frame(frame)
        self.frames.append(processed)
        
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)
        while len(self.frames) < self.frame_stack:
            self.frames.insert(0, processed)
        
        return np.array(self.frames, dtype=np.float32)
    
    def get_stacked_frames(self):
        if len(self.frames) == 0:
            return np.zeros((self.frame_stack, self.frame_size, self.frame_size), 
                          dtype=np.float32)
        return np.array(self.frames, dtype=np.float32)


def test_cnn_encoder():
    """Test CNN encoders with dummy input."""
    print("\n" + "="*60)
    print("Testing CNN Encoders")
    print("="*60)
    
    # Create dummy input (batch=2, channels=4, height=84, width=84)
    batch_size = 2
    x = torch.randn(batch_size, 4, 84, 84)
    print(f"\nInput shape: {x.shape}")
    print(f"Input size: {x.numel()} elements")
    
    # Test Nature CNN
    print("\n--- Nature CNN ---")
    nature = NatureCNN(in_channels=4, output_dim=512)
    out_nature = nature(x)
    print(f"Output shape: {out_nature.shape}")
    print(f"Output size: {out_nature.numel()} elements")
    
    # Test IMPALA CNN
    print("\n--- IMPALA CNN ---")
    impala = ImpalaCNN(in_channels=4, output_dim=512)
    out_impala = impala(x)
    print(f"Output shape: {out_impala.shape}")
    print(f"Output size: {out_impala.numel()} elements")
    
    # Test with flattened input
    print("\n--- Flattened Input Test ---")
    x_flat = torch.randn(batch_size, 4 * 84 * 84)
    print(f"Flattened input shape: {x_flat.shape}")
    out_flat = nature(x_flat)
    print(f"Output shape: {out_flat.shape}")
    
    print("\n✅ All tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_cnn_encoder()
