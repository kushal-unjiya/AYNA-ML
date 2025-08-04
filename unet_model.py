import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection"""
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
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final_relu(self.double_conv(x) + self.residual(x))

class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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
    """Upscaling with attention then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        if use_attention:
            self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            x2 = self.attention(g=x1, x=x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_colors, color_embedding_dim=64, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_colors = num_colors
        self.color_embedding_dim = color_embedding_dim

        # Enhanced color embedding with multiple projections
        self.color_embedding = nn.Embedding(num_colors, color_embedding_dim)
        
        # Color projections for different levels
        self.color_proj_64 = nn.Linear(color_embedding_dim, 64)
        self.color_proj_128 = nn.Linear(color_embedding_dim, 128)

        factor = 2 if bilinear else 1
        
        # Encoder with color conditioning at multiple levels
        self.inc = DoubleConv(n_channels + 1, 64)  # +1 for color channel
        self.down1 = Down(64 + 16, 128)  # +16 for reduced color features
        self.down2 = Down(128 + 32, 256)  # +32 for reduced color features
        self.down3 = Down(256 + 64, 512)  # +64 for reduced color features
        self.down4 = Down(512 + 128, 1024 // factor)  # +128 for reduced color features
        
        # Reduce color projections to avoid channel explosion
        self.color_proj_16 = nn.Linear(color_embedding_dim, 16)
        self.color_proj_32 = nn.Linear(color_embedding_dim, 32)

        # Decoder with attention
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=True)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention=True)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention=True)
        self.up4 = Up(128, 64, bilinear, use_attention=True)
        
        # Output with residual connection
        self.outc = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Tanh()  # Use Tanh for better color range
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, color_idx):
        B, _, H, W = x.shape
        
        # Get color embedding
        color_emb = self.color_embedding(color_idx)  # (B, embedding_dim)
        
        # Create color channel - normalize color index to [0, 1]
        color_channel = torch.full((B, 1, H, W), fill_value=0.0, device=x.device)
        for i in range(B):
            color_channel[i, 0, :, :] = color_idx[i].float() / (self.num_colors - 1)
        
        # Encoder path with multi-level color conditioning  
        x_input = torch.cat([x, color_channel], dim=1)  # Should be (B, 4, H, W)
        x1 = self.inc(x_input)
        
        # Add color features at each level (reduced dimensions)
        color_16 = self.color_proj_16(color_emb).unsqueeze(-1).unsqueeze(-1).expand(B, 16, x1.size(2), x1.size(3))
        x1_colored = torch.cat([x1, color_16], dim=1)
        x2 = self.down1(x1_colored)
        
        color_32 = self.color_proj_32(color_emb).unsqueeze(-1).unsqueeze(-1).expand(B, 32, x2.size(2), x2.size(3))
        x2_colored = torch.cat([x2, color_32], dim=1)
        x3 = self.down2(x2_colored)
        
        color_64 = self.color_proj_64(color_emb).unsqueeze(-1).unsqueeze(-1).expand(B, 64, x3.size(2), x3.size(3))
        x3_colored = torch.cat([x3, color_64], dim=1)
        x4 = self.down3(x3_colored)
        
        color_128 = self.color_proj_128(color_emb).unsqueeze(-1).unsqueeze(-1).expand(B, 128, x4.size(2), x4.size(3))
        x4_colored = torch.cat([x4, color_128], dim=1)
        x5 = self.down4(x4_colored)

        # Decoder path with attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return (logits + 1) / 2  # Convert tanh output [-1,1] to [0,1]