# Chapter 11: Advanced Computer Vision Topics

## 11.1 Object Detection

Object detection is the task of identifying and localizing objects within images. Unlike image classification which assigns a single label to an entire image, object detection draws bounding boxes around objects and classifies each one.

### 11.1.1 Two-Stage Detectors: R-CNN Family

**R-CNN (Region-based CNN)** was the first successful application of CNNs to object detection.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
import numpy as np

class RCNN(nn.Module):
    """Simplified R-CNN implementation"""

    def __init__(self, num_classes=21):  # 20 classes + background
        super(RCNN, self).__init__()

        # Backbone CNN (pre-trained ResNet)
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove FC and avg pool

        # Region of Interest (RoI) pooling
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes * 4)  # 4 coordinates per class
        )

    def forward(self, x, rois):
        """
        x: input image [B, C, H, W]
        rois: region proposals [N, 5] (batch_idx, x1, y1, x2, y2)
        """
        # Extract features
        features = self.backbone(x)  # [B, 2048, H/32, W/32]

        # Apply RoI pooling to each region
        roi_features = []
        for roi in rois:
            batch_idx = int(roi[0])
            roi_box = roi[1:5]  # [x1, y1, x2, y2]

            # Extract RoI features
            roi_feat = self.extract_roi_features(features[batch_idx:batch_idx+1], roi_box)
            roi_features.append(roi_feat)

        if roi_features:
            roi_features = torch.cat(roi_features, dim=0)

            # Classification
            class_logits = self.classifier(roi_features)

            # Bounding box regression
            bbox_deltas = self.bbox_regressor(roi_features)

            return class_logits, bbox_deltas

        return None, None

    def extract_roi_features(self, features, roi_box):
        """Extract features for a single RoI"""
        # Convert roi_box to feature map coordinates
        x1, y1, x2, y2 = roi_box

        # Simple spatial pooling (in practice, use RoIAlign)
        roi_feat = F.adaptive_max_pool2d(features, (7, 7))

        return roi_feat
```

### 11.1.2 Single-Stage Detectors: YOLO and SSD

**YOLO (You Only Look Once)** treats object detection as a regression problem.

```python
class YOLOv3Tiny(nn.Module):
    """Simplified YOLOv3-Tiny implementation"""

    def __init__(self, num_classes=80):
        super(YOLOv3Tiny, self).__init__()

        # Darknet backbone
        self.backbone = nn.Sequential(
            # Layer 1-4: Basic conv blocks
            self._make_conv_block(3, 16, 3),
            nn.MaxPool2d(2, 2),

            self._make_conv_block(16, 32, 3),
            nn.MaxPool2d(2, 2),

            self._make_conv_block(32, 64, 3),
            nn.MaxPool2d(2, 2),

            self._make_conv_block(64, 128, 3),
            nn.MaxPool2d(2, 2),

            self._make_conv_block(128, 256, 3),
        )

        # Detection head
        self.detector = nn.Sequential(
            nn.MaxPool2d(2, 2),
            self._make_conv_block(256, 512, 3),
            nn.Dropout(0.5),
            nn.Conv2d(512, 3 * (5 + num_classes), 1)  # 3 anchors, 5 + num_classes per anchor
        )

    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detector(features)

        return detections

def yolo_loss(predictions, targets, num_classes=80, num_anchors=3):
    """YOLO loss function"""

    batch_size = predictions.size(0)
    grid_size = predictions.size(2)

    # Reshape predictions: [B, 3*(5+num_classes), H, W] -> [B, H, W, 3, 5+num_classes]
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
    predictions = predictions.permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 5+num_classes]

    # Extract components
    pred_bbox = predictions[..., :4]  # [tx, ty, tw, th]
    pred_obj = predictions[..., 4:5]  # objectness score
    pred_cls = predictions[..., 5:]   # class probabilities

    # Apply sigmoid to objectness and class predictions
    pred_obj = torch.sigmoid(pred_obj)
    pred_cls = torch.sigmoid(pred_cls)

    # Convert tx, ty, tw, th to bx, by, bw, bh
    anchors = torch.tensor([[10, 14], [23, 27], [37, 58]])  # Example anchors

    # Loss computation would go here...
    # This is a simplified version - full YOLO loss is more complex

    return torch.tensor(0.0)  # Placeholder
```

## 11.2 Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in an image.

### 11.2.1 U-Net Architecture

```python
class UNet(nn.Module):
    """U-Net for semantic segmentation"""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)
        )

        # Decoder (Expanding Path)
        self.dec4 = self._make_decoder_block(1024, 256)
        self.dec3 = self._make_decoder_block(512, 128)
        self.dec2 = self._make_decoder_block(256, 64)
        self.dec1 = self._make_decoder_block(128, 64)

        # Final convolution
        self.final = nn.Conv2d(64, out_channels, 1)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        # Final output
        output = self.final(dec1)

        return output
```

### 11.2.2 DeepLab with Atrous Convolution

```python
class DeepLabv3Plus(nn.Module):
    """DeepLabv3+ with Atrous Spatial Pyramid Pooling (ASPP)"""

    def __init__(self, num_classes=21, backbone='resnet50'):
        super(DeepLabv3Plus, self).__init__()

        # Backbone encoder
        if backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # ASPP module
        self.aspp = ASPP(2048, [6, 12, 18])  # Atrous rates

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        features = self.backbone(x)

        # ASPP
        aspp_features = self.aspp(features)

        # Decoder
        output = self.decoder(aspp_features)

        # Upsample to original size
        output = self.upsample(output)

        return output

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Atrous convolution branches
        self.atrous_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for rate in atrous_rates
        ])

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 * (len(atrous_rates) + 2), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # Global average pooling branch
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Atrous branches
        atrous_features = [branch(x) for branch in self.atrous_branches]

        # 1x1 branch
        conv1x1_features = self.conv1x1(x)

        # Concatenate all features
        all_features = [global_features] + atrous_features + [conv1x1_features]
        concatenated = torch.cat(all_features, dim=1)

        # Final convolution
        output = self.final_conv(concatenated)

        return output
```

## 11.3 Instance Segmentation

Instance segmentation identifies individual object instances and segments each one.

### 11.3.1 Mask R-CNN

```python
class MaskRCNN(nn.Module):
    """Simplified Mask R-CNN implementation"""

    def __init__(self, num_classes=81):  # COCO has 80 classes + background
        super(MaskRCNN, self).__init__()

        # Backbone
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Region Proposal Network (RPN)
        self.rpn = RPN()

        # RoI Heads
        self.roi_heads = RoIHeads(num_classes)

        # Mask Head
        self.mask_head = MaskHead(num_classes)

    def forward(self, images, targets=None):
        """
        images: list of tensors [C, H, W]
        targets: list of dicts with 'boxes', 'labels', 'masks'
        """

        # Extract features
        features = self.backbone(images)

        # Generate proposals
        proposals = self.rpn(features)

        # RoI processing
        roi_features = self.roi_heads.extract_roi_features(features, proposals)

        # Classification and bbox regression
        class_logits, bbox_deltas = self.roi_heads.classify_and_regress(roi_features)

        # Mask prediction
        masks = self.mask_head(roi_features)

        return {
            'boxes': proposals,
            'labels': class_logits.argmax(dim=1),
            'scores': class_logits.softmax(dim=1).max(dim=1)[0],
            'masks': masks
        }

class RPN(nn.Module):
    """Region Proposal Network"""

    def __init__(self):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(2048, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, 9 * 2, 1)  # 9 anchors, 2 classes (obj/background)
        self.bbox_deltas = nn.Conv2d(512, 9 * 4, 1)  # 9 anchors, 4 coordinates

    def forward(self, features):
        # RPN forward pass
        rpn_features = F.relu(self.conv(features))

        # Classification logits
        cls_logits = self.cls_logits(rpn_features)

        # Bounding box deltas
        bbox_deltas = self.bbox_deltas(rpn_features)

        # Generate proposals (simplified)
        proposals = self.generate_proposals(cls_logits, bbox_deltas)

        return proposals

    def generate_proposals(self, cls_logits, bbox_deltas):
        """Generate region proposals from RPN outputs"""
        # This is a simplified version - real RPN uses anchor generation
        # and non-maximum suppression
        return torch.randn(100, 4)  # Placeholder: 100 random proposals

class RoIHeads(nn.Module):
    """RoI classification and regression heads"""

    def __init__(self, num_classes):
        super(RoIHeads, self).__init__()

        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 4)
        )

    def extract_roi_features(self, features, proposals):
        """Extract RoI features"""
        # Simplified RoI pooling
        roi_features = self.roi_pool(features)
        return roi_features

    def classify_and_regress(self, roi_features):
        """Classify and regress bounding boxes"""
        class_logits = self.classifier(roi_features)
        bbox_deltas = self.bbox_regressor(roi_features)

        return class_logits, bbox_deltas

class MaskHead(nn.Module):
    """Mask prediction head"""

    def __init__(self, num_classes):
        super(MaskHead, self).__init__()

        self.mask_conv = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2dTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, roi_features):
        """Predict masks"""
        masks = self.mask_conv(roi_features)
        masks = torch.sigmoid(masks)  # Convert to probabilities

        return masks
```

## 11.4 Vision Transformers for Dense Prediction

### 11.4.1 DETR (DEtection TRansformer)

```python
class DETR(nn.Module):
    """DETR: End-to-End Object Detection with Transformers"""

    def __init__(self, num_classes=91, num_queries=100):
        super(DETR, self).__init__()

        # CNN backbone
        self.backbone = nn.Sequential(
            *list(torchvision.models.resnet50(pretrained=True).children())[:-2]
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )

        # Position embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, 256, 25, 25))  # For 800x800 input

        # Object queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, 256))

        # Detection heads
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, 2048, H/32, W/32]

        # Reduce channels
        features = nn.Conv2d(2048, 256, 1)(features)  # [B, 256, H/32, W/32]

        # Add position embeddings
        features = features + self.position_embedding

        # Flatten spatial dimensions
        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        # Transformer
        memory = self.transformer.encoder(features)  # Encoder output

        # Decoder with object queries
        queries = self.object_queries.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        decoder_output = self.transformer.decoder(queries, memory)

        # Detection heads
        class_logits = self.class_embed(decoder_output)  # [num_queries, B, num_classes+1]
        bbox_coords = self.bbox_embed(decoder_output)    # [num_queries, B, 4]

        return {
            'pred_logits': class_logits.transpose(0, 1),  # [B, num_queries, num_classes+1]
            'pred_boxes': bbox_coords.transpose(0, 1)     # [B, num_queries, 4]
        }
```

### 11.4.2 Segment Anything Model (SAM)

```python
class SAMImageEncoder(nn.Module):
    """SAM Image Encoder - simplified version"""

    def __init__(self, img_size=1024, patch_size=16, embed_dim=768):
        super(SAMImageEncoder, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1
            ) for _ in range(12)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

class SAMPromptEncoder(nn.Module):
    """SAM Prompt Encoder"""

    def __init__(self, embed_dim=256):
        super(SAMPromptEncoder, self).__init__()

        self.embed_dim = embed_dim

        # Point embeddings
        self.point_embeddings = nn.Embedding(3, embed_dim)  # 0: negative, 1: positive, 2: box corner

        # Box embeddings
        self.box_embeddings = nn.Embedding(3, embed_dim)  # 0: corner, 1: side, 2: center

    def forward(self, points=None, boxes=None):
        """Encode prompts"""

        embeddings = []

        if points is not None:
            point_embeddings = self.point_embeddings(points[..., 2:3].long())  # Use label as embedding index
            embeddings.append(point_embeddings)

        if boxes is not None:
            # Encode box corners
            box_embeddings = self.box_embeddings(torch.zeros_like(boxes[..., :2]).long())
            embeddings.append(box_embeddings)

        if embeddings:
            return torch.cat(embeddings, dim=1)

        return None

class SAMMaskDecoder(nn.Module):
    """SAM Mask Decoder"""

    def __init__(self, embed_dim=256):
        super(SAMMaskDecoder, self).__init__()

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=2
        )

        self.mask_tokens = nn.Embedding(4, embed_dim)  # 4 mask tokens
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 16, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 16, 1, 1)  # Output mask
        )

    def forward(self, image_embeddings, prompt_embeddings):
        """Decode masks from image and prompt embeddings"""

        # Get mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(1).repeat(1, image_embeddings.size(0), 1)

        # Transformer decoder
        decoded = self.transformer(mask_tokens, image_embeddings)

        # Reshape for upscaling
        B, num_masks, C = decoded.shape
        decoded = decoded.view(B * num_masks, C, 1, 1)

        # Upscale to mask size
        masks = self.output_upscaling(decoded)

        return masks.view(B, num_masks, masks.size(2), masks.size(3))

class SAM(nn.Module):
    """Segment Anything Model"""

    def __init__(self):
        super(SAM, self).__init__()

        self.image_encoder = SAMImageEncoder()
        self.prompt_encoder = SAMPromptEncoder()
        self.mask_decoder = SAMMaskDecoder()

    def forward(self, image, points=None, boxes=None):
        # Encode image
        image_embeddings = self.image_encoder(image)

        # Encode prompts
        prompt_embeddings = self.prompt_encoder(points, boxes)

        # Decode masks
        masks = self.mask_decoder(image_embeddings, prompt_embeddings)

        return masks
```

## 11.5 Summary

This chapter covered advanced computer vision topics beyond basic image classification:

1. **Object Detection**: R-CNN, YOLO, SSD architectures
2. **Semantic Segmentation**: U-Net, DeepLab with ASPP
3. **Instance Segmentation**: Mask R-CNN
4. **Vision Transformers**: DETR, SAM for dense prediction tasks

Key takeaways:
- Object detection requires both classification and localization
- Segmentation tasks need architectures that preserve spatial information
- Transformers are increasingly important for vision tasks
- Modern CV systems combine multiple techniques for comprehensive scene understanding

The next chapter will explore research frontiers and future directions in computer vision.