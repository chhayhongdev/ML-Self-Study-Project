import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

print("Chapter 8: Modern CNN Architectures")
print("=" * 50)

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 8.1 EfficientNet Implementation
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale
        return x * y

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""

    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3):
        super(MBConvBlock, self).__init__()

        self.stride = stride
        self.expand_ratio = expand_ratio
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        else:
            self.expand_conv = None

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels)

        # Pointwise convolution
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand_conv is not None:
            x = F.relu6(self.expand_bn(self.expand_conv(x)))

        # Depthwise
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))

        # SE block
        x = self.se(x)

        # Projection
        x = self.project_bn(self.project_conv(x))

        # Skip connection
        if self.skip_connection:
            x += identity

        return x

class EfficientNet(nn.Module):
    """EfficientNet-B0 Implementation"""

    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        # MBConv blocks configuration for B0
        self.blocks = nn.Sequential(
            # Stage 1: 1 block, 32 -> 16, stride 1
            MBConvBlock(32, 16, expand_ratio=1, stride=1),

            # Stage 2: 2 blocks, 16 -> 24, stride 2
            MBConvBlock(16, 24, expand_ratio=6, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, stride=1),

            # Stage 3: 2 blocks, 24 -> 40, stride 2
            MBConvBlock(24, 40, expand_ratio=6, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, stride=1),

            # Stage 4: 3 blocks, 40 -> 80, stride 2
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 80, expand_ratio=6, stride=1),

            # Stage 5: 3 blocks, 80 -> 112, stride 1
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, stride=1),

            # Stage 6: 4 blocks, 112 -> 192, stride 2
            MBConvBlock(112, 192, expand_ratio=6, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 192, expand_ratio=6, stride=1),

            # Stage 7: 1 block, 192 -> 320, stride 1
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# 8.2 MobileNetV3 Implementation
class HardSwish(nn.Module):
    """Hard Swish activation: x * ReLU6(x+3)/6"""

    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class MobileNetV3Block(nn.Module):
    """MobileNetV3 Block with SE and Hard Swish"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MobileNetV3Block, self).__init__()

        self.stride = stride
        expanded_channels = in_channels * expand_ratio

        # Expansion
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                HardSwish()
            )
        else:
            self.expand = None

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            HardSwish()
        )

        # SE Block
        if se_ratio > 0:
            se_channels = max(1, int(expanded_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                HardSwish(),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection
        self.skip_connection = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x

        if self.expand is not None:
            x = self.expand(x)

        x = self.depthwise(x)

        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight

        x = self.project(x)

        if self.skip_connection:
            x += identity

        return x

class MobileNetV3(nn.Module):
    """MobileNetV3-Large"""

    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )

        # MobileNetV3 blocks
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, 3, 1, 1, se_ratio=0.25),
            MobileNetV3Block(16, 24, 3, 2, 4, se_ratio=0),
            MobileNetV3Block(24, 24, 3, 1, 3, se_ratio=0),
            MobileNetV3Block(24, 40, 5, 2, 3, se_ratio=0.25),
            MobileNetV3Block(40, 40, 5, 1, 3, se_ratio=0.25),
            MobileNetV3Block(40, 40, 5, 1, 3, se_ratio=0.25),
            MobileNetV3Block(40, 80, 3, 2, 6, se_ratio=0),
            MobileNetV3Block(80, 80, 3, 1, 2, se_ratio=0),  # Changed from 2.5 to 2
            MobileNetV3Block(80, 80, 3, 1, 2, se_ratio=0),  # Changed from 2.5 to 2
            MobileNetV3Block(80, 80, 3, 1, 2, se_ratio=0),  # Changed from 2.5 to 2
            MobileNetV3Block(80, 112, 3, 1, 6, se_ratio=0.25),
            MobileNetV3Block(112, 112, 3, 1, 6, se_ratio=0.25),
            MobileNetV3Block(112, 160, 5, 2, 6, se_ratio=0.25),
            MobileNetV3Block(160, 160, 5, 1, 6, se_ratio=0.25),
            MobileNetV3Block(160, 160, 5, 1, 6, se_ratio=0.25),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# 8.3 Vision Transformer Implementation
class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Position embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, _, _, _ = x.shape

        # Create patches: (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)

        # Flatten patches: (B, embed_dim, num_patches)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.position_embedding

        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        return x

class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification: use class token
        x = self.norm(x)
        x = self.head(x[:, 0])  # Class token is at position 0

        return x

# 8.4 Training and Comparison
def create_data_loaders():
    """Create CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.Resize(224),  # For ViT compatibility
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, test_loader
    """Train a model and return final accuracy"""
    print(f"\\nTraining {model_name}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, test_loader = create_data_loaders()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}: {accuracy:.2f}%")

    training_time = time.time() - start_time
    print(".2f")

    return accuracy, training_time

def compare_architectures():
    """Compare modern architectures"""
    print("Comparing Modern CNN Architectures on CIFAR-10")
    print("=" * 60)

    architectures = {
        "EfficientNet": EfficientNet(num_classes=10),
        "MobileNetV3": MobileNetV3(num_classes=10),
        "ViT": VisionTransformer(
            image_size=224, patch_size=16, num_classes=10,
            embed_dim=384, depth=6, num_heads=6  # Smaller ViT for CIFAR-10
        )
    }

    results = {}

    for name, model in architectures.items():
        try:
            accuracy, training_time = train_model(model, name, epochs=2)
            results[name] = {
                'accuracy': accuracy,
                'time': training_time,
                'params': sum(p.numel() for p in model.parameters())
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # Print comparison
    print("\\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    print("<15")
    print("-" * 60)

    for name, metrics in results.items():
        print("<15")

    print("\\nNote: ViT typically requires large datasets and pre-training for best performance.")

if __name__ == '__main__':
    # Test individual architectures
    print("Testing EfficientNet...")
    efficientnet = EfficientNet(num_classes=10)
    print(f"EfficientNet parameters: {sum(p.numel() for p in efficientnet.parameters()):,}")

    print("\\nTesting MobileNetV3...")
    mobilenet = MobileNetV3(num_classes=10)
    print(f"MobileNetV3 parameters: {sum(p.numel() for p in mobilenet.parameters()):,}")

    print("\\nTesting Vision Transformer...")
    vit = VisionTransformer(image_size=224, patch_size=16, num_classes=10,
                           embed_dim=384, depth=6, num_heads=6)
    print(f"ViT parameters: {sum(p.numel() for p in vit.parameters()):,}")

    # Uncomment to run full comparison (takes time)
    # compare_architectures()