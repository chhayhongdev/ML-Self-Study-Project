#!/usr/bin/env python3
"""
Chapter 12: Research Frontiers in Computer Vision - Code Examples

This file demonstrates implementations of cutting-edge CV research:
- Multimodal Learning (CLIP, Image Captioning)
- Self-Supervised Learning (SimCLR, MAE)
- Neural Architecture Search (DARTS)
- Federated Learning (FedAvg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Chapter 12: Research Frontiers in Computer Vision")
print("=" * 50)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 12.1 Multimodal Learning
# ============================================================================

class CLIPLikeModel(nn.Module):
    """Simplified CLIP-like model for vision-language understanding"""

    def __init__(self, vision_dim=512, text_dim=512, embed_dim=256):
        super(CLIPLikeModel, self).__init__()

        # Vision encoder (simplified)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, vision_dim),
            nn.ReLU(),
            nn.Linear(vision_dim, embed_dim)
        )

        # Text encoder (simplified)
        self.text_encoder = nn.Sequential(
            nn.Embedding(50000, text_dim),  # Vocabulary size
            nn.LSTM(text_dim, text_dim, batch_first=True),
            nn.Linear(text_dim, embed_dim)
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, input_ids):
        # Encode images
        image_features = self.vision_encoder(images)
        image_features = F.normalize(image_features, dim=-1)

        # Encode text
        embedded = self.text_encoder[0](input_ids)  # Embedding
        lstm_out, _ = self.text_encoder[1](embedded)  # LSTM
        text_embeddings = self.text_encoder[2](lstm_out[:, -1, :])  # Linear on last timestep
        text_features = F.normalize(text_embeddings, dim=-1)

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

def clip_contrastive_loss(logits_per_image, logits_per_text):
    """CLIP contrastive loss"""
    batch_size = logits_per_image.size(0)

    # Labels are diagonal (each image matches its corresponding text)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # Image-to-text loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)

    # Text-to-image loss
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2

def demo_multimodal_learning():
    """Demonstrate multimodal learning with CLIP-like model"""
    print("\n12.1 Multimodal Learning Demo")
    print("-" * 30)

    # Create sample inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 20)).to(device)  # 20 tokens per text

    # Test CLIP-like model
    print("Testing CLIP-like model...")
    model = CLIPLikeModel().to(device)
    model.eval()

    with torch.no_grad():
        logits_per_image, logits_per_text = model(images, input_ids)

    print(f"Logits per image shape: {logits_per_image.shape}")
    print(f"Logits per text shape: {logits_per_text.shape}")

    # Test contrastive loss
    loss = clip_contrastive_loss(logits_per_image, logits_per_text)
    print(f"CLIP contrastive loss: {loss:.4f}")

# ============================================================================
# 12.2 Self-Supervised Learning
# ============================================================================

class SimCLR(nn.Module):
    """SimCLR: Simple Framework for Contrastive Learning of Representations"""

    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()

        # Base encoder (e.g., ResNet)
        self.encoder = base_encoder

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        # Encode
        h = self.encoder(x)  # [B, 2048]

        # Project
        z = self.projector(h)  # [B, projection_dim]

        return h, z

def simclr_loss(z_i, z_j, temperature=0.5):
    """
    SimCLR contrastive loss
    z_i, z_j: representations of two augmentations of the same image
    """
    batch_size = z_i.size(0)

    # Concatenate representations
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Mask out self-similarities
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))

    # Labels: positive pairs are at positions (i, i+B) and (i+B, i)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Cross entropy loss
    loss = F.cross_entropy(sim, labels)

    return loss

def demo_self_supervised_learning():
    """Demonstrate self-supervised learning with SimCLR"""
    print("\n12.2 Self-Supervised Learning Demo")
    print("-" * 35)

    # Create base encoder
    base_encoder = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 2048),
        nn.ReLU()
    )

    # Test SimCLR
    print("Testing SimCLR...")
    simclr = SimCLR(base_encoder).to(device)
    simclr.eval()

    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        h, z = simclr(x)

    print(f"SimCLR representations shape: h={h.shape}, z={z.shape}")

    # Test contrastive loss with two views
    z_i = torch.randn(batch_size, 128).to(device)
    z_j = torch.randn(batch_size, 128).to(device)

    loss = simclr_loss(z_i, z_j)
    print(f"SimCLR contrastive loss: {loss:.4f}")

# ============================================================================
# 12.3 Neural Architecture Search
# ============================================================================

class MixedOperation(nn.Module):
    """Mixed operation for differentiable NAS"""

    def __init__(self, operations):
        super(MixedOperation, self).__init__()
        self.operations = nn.ModuleList(operations)
        self.alphas = nn.Parameter(torch.randn(len(operations)))

    def forward(self, x):
        # Softmax over operation weights
        weights = F.softmax(self.alphas, dim=0)

        # Weighted sum of operations
        output = 0
        for weight, op in zip(weights, self.operations):
            output = output + weight * op(x)

        return output

class DARTSCell(nn.Module):
    """DARTS cell with mixed operations"""

    def __init__(self, in_channels, out_channels):
        super(DARTSCell, self).__init__()

        # Define possible operations
        operations = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 5, padding=2), nn.BatchNorm2d(out_channels), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, out_channels, 1)),
            nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, out_channels, 1)),
            nn.Identity()  # Skip connection
        ]

        self.mixed_op = MixedOperation(operations)

    def forward(self, x):
        return self.mixed_op(x)

class DARTSNetwork(nn.Module):
    """DARTS network"""

    def __init__(self, num_classes=10):
        super(DARTSNetwork, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Normal cells
        self.cells = nn.ModuleList([
            DARTSCell(64, 64) for _ in range(8)
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        for cell in self.cells:
            x = cell(x)

        x = self.classifier(x)
        return x

def demo_neural_architecture_search():
    """Demonstrate neural architecture search with DARTS"""
    print("\n12.3 Neural Architecture Search Demo")
    print("-" * 38)

    # Test DARTS network
    print("Testing DARTS network...")
    darts = DARTSNetwork(num_classes=10).to(device)
    darts.eval()

    x = torch.randn(2, 3, 32, 32).to(device)  # CIFAR-10 size

    with torch.no_grad():
        output = darts(x)

    print(f"DARTS output shape: {output.shape}")

    # Test mixed operation
    print("Testing mixed operation...")
    operations = [
        nn.Conv2d(64, 64, 3, padding=1),
        nn.Conv2d(64, 64, 5, padding=2),
        nn.Identity()
    ]
    mixed_op = MixedOperation(operations).to(device)
    mixed_op.eval()

    x = torch.randn(2, 64, 32, 32).to(device)

    with torch.no_grad():
        output = mixed_op(x)

    print(f"Mixed operation output shape: {output.shape}")
    print(f"Mixed operation alphas: {mixed_op.alphas}")

# ============================================================================
# 12.4 Federated Learning
# ============================================================================

class FederatedClient:
    """Federated learning client"""

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.local_data = None

    def set_local_data(self, data_loader):
        self.local_data = data_loader

    def local_train(self, num_epochs=1):
        """Train on local data"""
        self.model.train()

        for _ in range(num_epochs):
            for inputs, targets in self.local_data:
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

        return self.get_model_parameters()

    def get_model_parameters(self):
        """Get current model parameters"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, parameters):
        """Set model parameters"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()

class FederatedServer:
    """Federated learning server"""

    def __init__(self, global_model):
        self.global_model = global_model
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)

    def federated_averaging(self, client_parameters_list):
        """FedAvg algorithm"""
        # Average parameters from all clients
        averaged_params = {}

        for name, _ in self.global_model.named_parameters():
            param_sum = torch.stack([client_params[name] for client_params in client_parameters_list])
            averaged_params[name] = torch.mean(param_sum, dim=0)

        # Update global model
        for name, param in self.global_model.named_parameters():
            param.data = averaged_params[name].clone()

        return averaged_params

    def global_round(self, num_local_epochs=1):
        """One round of federated learning"""

        # Send global model to clients
        global_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_model_parameters(global_params)

        # Local training on each client
        client_parameters_list = []
        for client in self.clients:
            client_params = client.local_train(num_local_epochs)
            client_parameters_list.append(client_params)

        # Aggregate parameters
        self.federated_averaging(client_parameters_list)

        return self.global_model

def demo_federated_learning():
    """Demonstrate federated learning with FedAvg"""
    print("\n12.4 Federated Learning Demo")
    print("-" * 30)

    # Create simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Create federated server
    global_model = model.to(device)
    server = FederatedServer(global_model)

    # Create clients
    num_clients = 3
    clients = []

    for _ in range(num_clients):
        client_model = model.to(device)
        optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        client = FederatedClient(client_model, optimizer, criterion)

        # Create synthetic local data for each client
        # In practice, this would be real data partitioned across clients
        local_data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(100, 1, 28, 28),  # MNIST-like
                torch.randint(0, 10, (100,))
            ),
            batch_size=10,
            num_workers=0
        )
        client.set_local_data(local_data)
        clients.append(client)

    # Add clients to server
    for client in clients:
        server.add_client(client)

    print(f"Created federated learning setup with {num_clients} clients")

    print("Running federated learning round...")

    updated_model = server.global_round(num_local_epochs=1)

    print("Federated learning round completed.")

    # Test global model
    test_input = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        output = updated_model(test_input)

    print(f"Global model output shape: {output.shape}")

# ============================================================================
# Main Demo Function
# ============================================================================

def benchmark_research_models():
    """Benchmark research frontier models"""
    print("\nResearch Models Benchmarking")
    print("=" * 30)

    models = {
        'CLIP-like': CLIPLikeModel(),
        'SimCLR': SimCLR(nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 2048), nn.ReLU())),
        'DARTS': DARTSNetwork(num_classes=10),
        'Federated Model': nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    }

    input_sizes = {
        'CLIP-like': [(2, 3, 224, 224), (2, 20)],  # images, text_ids
        'SimCLR': (2, 3, 224, 224),
        'DARTS': (2, 3, 32, 32),
        'Federated Model': (2, 1, 28, 28)
    }

    results = []

    for name, model in models.items():
        model = model.to(device)
        model.eval()

        input_size = input_sizes[name]

        # Warm up
        with torch.no_grad():
            if name == 'CLIP-like':
                _ = model(torch.randn(*input_size[0]).to(device), torch.randint(0, 1000, input_size[1]).to(device))
            else:
                _ = model(torch.randn(*input_size).to(device))

        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                if name == 'CLIP-like':
                    _ = model(torch.randn(*input_size[0]).to(device), torch.randint(0, 1000, input_size[1]).to(device))
                else:
                    _ = model(torch.randn(*input_size).to(device))

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # ms

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        results.append({
            'model': name,
            'params': params,
            'inference_time': avg_time
        })

        print("20")

    return results

def main():
    """Run all research frontiers demonstrations"""
    print("Chapter 12: Research Frontiers in Computer Vision - Complete Demo")
    print("=" * 70)

    # Run individual demos
    demo_multimodal_learning()
    demo_self_supervised_learning()
    demo_neural_architecture_search()
    demo_federated_learning()

    # Benchmark models
    benchmark_research_models()

    # Summary
    print("\nSummary")
    print("=" * 10)
    print("Successfully demonstrated research frontiers:")
    print("• Multimodal Learning: CLIP-like vision-language model")
    print("• Self-Supervised Learning: SimCLR contrastive learning")
    print("• Neural Architecture Search: DARTS with mixed operations")
    print("• Federated Learning: FedAvg algorithm implementation")
    print("\nKey insights:")
    print("• Multimodal models enable richer understanding by combining modalities")
    print("• Self-supervised learning reduces labeled data requirements")
    print("• NAS automates architecture design for optimal performance")
    print("• Federated learning preserves privacy in distributed settings")
    print("• These frontiers represent the cutting edge of CV research")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()