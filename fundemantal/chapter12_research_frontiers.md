# Chapter 12: Research Frontiers in Computer Vision

## 12.1 Multimodal Learning

Multimodal learning combines information from multiple modalities (vision, text, audio, etc.) to improve understanding and performance.

### 12.1.1 Vision-Language Models

**CLIP (Contrastive Language-Image Pretraining)** learns joint representations of images and text.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

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
        text_embeddings = self.text_encoder(input_ids)
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
```

### 12.1.2 Image Captioning with Transformers

```python
class ImageCaptioningModel(nn.Module):
    """Image captioning using transformer decoder"""

    def __init__(self, vocab_size=10000, embed_dim=256, num_heads=8, num_layers=6):
        super(ImageCaptioningModel, self).__init__()

        # Image encoder
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048, embed_dim)

        # Text decoder (transformer)
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(50, embed_dim)  # Max sequence length

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Special tokens
        self.sos_token = 1  # Start of sequence
        self.eos_token = 2  # End of sequence

    def forward(self, images, captions=None, max_length=50):
        batch_size = images.size(0)

        # Encode images
        image_features = self.encoder(images)  # [B, embed_dim]
        image_features = image_features.unsqueeze(0)  # [1, B, embed_dim]

        if self.training and captions is not None:
            # Teacher forcing during training
            return self._forward_train(image_features, captions)
        else:
            # Autoregressive generation during inference
            return self._forward_generate(image_features, batch_size, max_length)

    def _forward_train(self, image_features, captions):
        # Create target sequence (shift right)
        tgt = captions[:, :-1]  # Remove last token
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))

        # Embed target sequence
        tgt_embed = self.decoder_embed(tgt)  # [B, seq_len, embed_dim]
        tgt_embed = tgt_embed + self.pos_embed(torch.arange(tgt.size(1), device=tgt.device))
        tgt_embed = tgt_embed.transpose(0, 1)  # [seq_len, B, embed_dim]

        # Decode
        output = self.decoder(tgt_embed, image_features, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # [B, seq_len, embed_dim]

        # Project to vocabulary
        logits = self.output_proj(output)  # [B, seq_len, vocab_size]

        return logits

    def _forward_generate(self, image_features, batch_size, max_length):
        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=image_features.device)

        for _ in range(max_length - 1):
            tgt_embed = self.decoder_embed(generated)
            tgt_embed = tgt_embed + self.pos_embed(torch.arange(generated.size(1), device=generated.device))
            tgt_embed = tgt_embed.transpose(0, 1)

            tgt_mask = self._generate_square_subsequent_mask(generated.size(1))

            output = self.decoder(tgt_embed, image_features, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)

            logits = self.output_proj(output[:, -1:])  # Only last token
            next_token = logits.argmax(dim=-1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated
            if (next_token == self.eos_token).all():
                break

        return generated

    def _generate_square_subsequent_mask(self, sz):
        """Generate mask for transformer decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
```

## 12.2 Self-Supervised Learning

Self-supervised learning learns representations without labeled data.

### 12.2.1 Contrastive Learning: SimCLR

```python
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

class SimCLRDataAugmentation:
    """Data augmentation pipeline for SimCLR"""

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)  # Two augmentations
```

### 12.2.2 Masked Image Modeling: MAE

```python
class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for Vision (MAE)"""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12):
        super(MaskedAutoencoder, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(embed_dim))

        # Encoder (ViT-like)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=3072,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=3072,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * 3)
        )

    def forward(self, x, mask_ratio=0.75):
        # Patchify
        patches = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add position embeddings
        patches = patches + self.pos_embed

        # Create mask
        num_masked = int(mask_ratio * self.num_patches)
        mask_indices = torch.randperm(self.num_patches)[:num_masked]
        keep_indices = torch.randperm(self.num_patches)[num_masked:]

        # Create masked sequence
        masked_patches = patches.clone()
        masked_patches[:, mask_indices] = self.mask_token

        # Encode
        encoded = self.encoder(masked_patches.transpose(0, 1))  # [num_patches, B, embed_dim]
        encoded = encoded.transpose(0, 1)  # [B, num_patches, embed_dim]

        # Prepare decoder input
        decoder_input = torch.zeros_like(patches)
        decoder_input[:, keep_indices] = encoded[:, keep_indices]
        decoder_input[:, mask_indices] = self.mask_token

        # Decode
        decoder_output = self.decoder(
            decoder_input.transpose(0, 1),
            encoded.transpose(0, 1)
        )
        decoder_output = decoder_output.transpose(0, 1)

        # Reconstruct
        reconstructed = self.reconstruction_head(decoder_output)

        # Only compute loss on masked patches
        loss = F.mse_loss(reconstructed[:, mask_indices], patches[:, mask_indices])

        return reconstructed, loss
```

## 12.3 Neural Architecture Search

Neural Architecture Search (NAS) automatically finds optimal network architectures.

### 12.3.1 Differentiable NAS

```python
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
```

## 12.4 Federated Learning

Federated learning trains models across decentralized devices while keeping data private.

### 12.4.1 Federated Averaging (FedAvg)

```python
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

        for epoch in range(num_epochs):
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
```

## 12.5 Summary

This chapter explored cutting-edge research frontiers in computer vision:

1. **Multimodal Learning**: CLIP, image captioning with transformers
2. **Self-Supervised Learning**: SimCLR, MAE for representation learning
3. **Neural Architecture Search**: DARTS for automated architecture design
4. **Federated Learning**: Privacy-preserving distributed training

Key takeaways:
- Multimodal models combine vision with other modalities for richer understanding
- Self-supervised learning reduces dependence on labeled data
- NAS automates architecture design for optimal performance
- Federated learning enables privacy-preserving collaborative learning

The field of computer vision continues to evolve rapidly, with these research directions shaping the future of AI-powered visual understanding.