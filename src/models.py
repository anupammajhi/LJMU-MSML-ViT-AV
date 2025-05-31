import timm
import math
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from collections import OrderedDict
import torch.nn as nn
import torch
from torchvision.models import mobilenet_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from vars import CLASS2IDX,IMG_SIZE,NUM_CLASSES

CLASS2IDX = CLASS2IDX
IMG_SIZE = IMG_SIZE
NUM_CLASSES = NUM_CLASSES

class ViTBackbone(nn.Module):
    def __init__(self, model_name, img_size, pretrained=True):
      super().__init__()

      self.img_size = img_size
      self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=False)
      if 'efficientvit' in model_name:
            self.num_channels = 256
            print(f"INFO: Manually set output channels of {model_name} to {self.num_channels}")

      elif 'mobilevit' in model_name:
            self.num_channels = 640
            print(f"INFO: Manually set output channels of {model_name} to {self.num_channels}")
      else:
            try:
              self.num_channels = self.backbone.head.in_features
              print(f"INFO: Determined output channels from backbone.head.in_features: {self.num_channels}")
            except AttributeError:
                try:
                    if hasattr(self.backbone, 'conv_head'):
                        self.num_channels = self.backbone.conv_head.out_channels
                        print(f"INFO: Determined output channels from backbone.conv_head.out_channels: {self.num_channels}")
                    elif hasattr(self.backbone, 'head'): 
                        if isinstance(self.backbone.head, nn.Conv2d):
                            self.num_channels = self.backbone.head.in_channels
                            print(f"INFO: Determined output channels from backbone.head.in_channels: {self.num_channels}")
                        elif isinstance(self.backbone.head, nn.Sequential) and isinstance(self.backbone.head[-1], nn.Conv2d):
                            self.num_channels = self.backbone.head[-1].in_channels
                            print(f"INFO: Determined output channels from backbone.head[-1].in_channels: {self.num_channels}")
                        else:
                            print(f"Warning: Could not automatically determine output channels from head for {model_name}. Assuming embed_dim or 768.")
                            self.num_channels = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else 768
                            if hasattr(self.backbone, 'embed_dim'):
                                print(f"INFO: Assuming embed_dim for {model_name}: {self.num_channels}")
                            else:
                                print(f"INFO: Assuming default 768 channels for {model_name}.")
                    else:
                        print(f"Warning: Could not automatically determine output channels for {model_name}. Assuming 768 channels.")
                        self.num_channels = 768
                except AttributeError:
                        print(f"Warning: Could not automatically determine output channels for {model_name}. Assuming 768 channels.")
                        self.num_channels = 768


      self.out_channels = self.num_channels

      if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'patch_size'):
          self.patch_size = self.backbone.patch_embed.patch_size[0]
      elif 'mobilevit' in model_name:
          self.patch_size = 2
      elif 'efficientvit' in model_name:
          self.patch_size = 16
      else:
          self.patch_size = 16

    def forward(self, x):
        features = self.backbone.forward_features(x)
        B = x.shape[0]
        img_h, img_w = x.shape[2:]

        if isinstance(features, torch.Tensor):
            if len(features.shape) == 3:
                has_cls_token = hasattr(self.backbone, 'cls_token')
                if has_cls_token:
                    features = features[:, 1:, :]

                num_patches = features.shape[1]
                expected_h = img_h // self.patch_size
                expected_w = img_w // self.patch_size

                if num_patches == expected_h * expected_w:
                    h, w = expected_h, expected_w
                elif int(round(math.sqrt(num_patches)))**2 == num_patches:
                    h = w = int(round(math.sqrt(num_patches)))
                else:
                    raise ValueError(f"Unexpected number of patches ({num_patches}) for reshaping. Expected {expected_h*expected_w} based on image size {img_h}x{img_w} and patch size {self.patch_size}. Feature shape: {features.shape}")

                features = features.transpose(1, 2).reshape(B, self.num_channels, h, w)

            elif len(features.shape) == 4:
                if features.shape[1] != self.num_channels:
                    print(f"Warning: Feature tensor channel count ({features.shape[1]}) does not match expected channels ({self.num_channels})")
                pass
            else:
                raise ValueError(f"Unexpected feature tensor shape: {features.shape}")

        elif isinstance(features, (list, tuple)):
            last_feature_map = features[-1]
            if not isinstance(last_feature_map, torch.Tensor) or len(last_feature_map.shape) != 4:
                 raise TypeError(f"Expected the last element in the feature list/tuple to be a 4D tensor (B, C, H, W), but got shape: {last_feature_map.shape}")
            features = last_feature_map
            if features.shape[1] != self.num_channels:
                 print(f"Warning: Last feature map channel count ({features.shape[1]}) does not match expected channels ({self.num_channels})")

        else:
            raise TypeError(f"Unexpected feature type from backbone: {type(features)}")

        if not isinstance(features, torch.Tensor) or len(features.shape) != 4:
             raise RuntimeError(f"Backbone forward method returned features in an unexpected final shape: {features.shape}")


        return features

def make_baseline_vit():
    backbone = ViTBackbone("vit_tiny_patch16_384", img_size = IMG_SIZE) #vit_tiny_patch16_384, vit_small_patch16_384
    backbone_out_channels = backbone.out_channels

    anchor_gen = AnchorGenerator( sizes=((13, 21, 34, 55, 89, 144, 233, 377, 500),),
                                aspect_ratios=((0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0),))

    num_anchors = len(anchor_gen.sizes[0]) * len(anchor_gen.aspect_ratios[0])

    rpn_head = RPNHead(
        in_channels=backbone_out_channels,
        num_anchors=num_anchors
    )

    model = FasterRCNN(backbone,
                      num_classes=NUM_CLASSES+1,  # +background
                      rpn_anchor_generator=anchor_gen,
                      rpn_head=rpn_head, 
                      box_nms_thresh = 0.3,
                      box_score_thresh = 0.5,
                      min_size=IMG_SIZE, max_size=IMG_SIZE)
    return model


def make_mobilevit():
    backbone = ViTBackbone("mobilevit_s", img_size=IMG_SIZE)
    backbone_out_channels = backbone.out_channels

    anchor_gen = AnchorGenerator(sizes=((13, 21, 34, 55, 89, 144, 233, 377),),
                                aspect_ratios=((0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0),))

    num_anchors = len(anchor_gen.sizes[0]) * len(anchor_gen.aspect_ratios[0])

    rpn_head = RPNHead(
        in_channels=backbone_out_channels,
        num_anchors=num_anchors
    )

    model = FasterRCNN(backbone,
                      num_classes=NUM_CLASSES+1,  # +background
                      rpn_anchor_generator=anchor_gen,
                      rpn_head=rpn_head,
                      box_nms_thresh = 0.3,
                      box_score_thresh = 0.7,
                      min_size=IMG_SIZE, max_size=IMG_SIZE)
    return model


def make_efficientvit():
    backbone = ViTBackbone("efficientvit_b1.r224_in1k", img_size=IMG_SIZE)
    backbone_out_channels = backbone.out_channels 

    anchor_gen = AnchorGenerator(sizes=((13, 21, 34, 55, 89, 144, 233, 377),),
                                aspect_ratios=((0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0),))

    num_anchors = len(anchor_gen.sizes[0]) * len(anchor_gen.aspect_ratios[0])

    rpn_head = RPNHead(
        in_channels=backbone_out_channels, 
        num_anchors=num_anchors
    )

    model = FasterRCNN(backbone,
                      num_classes=NUM_CLASSES+1,  # +background
                      rpn_anchor_generator=anchor_gen,
                      rpn_head=rpn_head, 
                      box_nms_thresh = 0.3,
                      box_score_thresh = 0.7,
                      min_size=IMG_SIZE, max_size=IMG_SIZE)
    return model

class DummyBackbone(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
    def forward(self, x):
        # This dummy forward doesn't process images.
        # Placeholder for FasterRCNN initialization
        # Actual feature extraction happens in HybridCNNViT's forward
        return None

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x) 
        x = identity + x 

        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = identity + x 
        return x

class HybridCNNViT(nn.Module):
    def __init__(self):
        super().__init__()

        cnn = mobilenet_v2(weights='DEFAULT')
        self.cnn = nn.Sequential(*list(cnn.features)[:15])

        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            dummy_output = self.cnn(dummy_input)
            self.cnn_out_channels = dummy_output.shape[1]
            H_feat_initial, W_feat_initial = dummy_output.shape[-2:]
            self.cnn_stride = IMG_SIZE // H_feat_initial 
            print(f"[INFO] Determined CNN output channels: {self.cnn_out_channels}")
            print(f"[INFO] Determined CNN stride: {self.cnn_stride}")

        self.vit_embed_dim = 192 
        self.conv_to_vit_embed = nn.Conv2d(self.cnn_out_channels, self.vit_embed_dim, 1)

        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            cnn_features = self.cnn(dummy_input)
            H_feat, W_feat = cnn_features.shape[-2:]
            patch_size_for_padding = 8
            pad_h = (patch_size_for_padding - (H_feat % patch_size_for_padding)) % patch_size_for_padding
            pad_w = (patch_size_for_padding - (W_feat % patch_size_for_padding)) % patch_size_for_padding
            H_padded = H_feat + pad_h
            W_padded = W_feat + pad_w
            self.num_spatial_tokens = H_padded * W_padded # Each token is a spatial location

        print(f"[INFO] Expected feature map size after CNN: {H_feat}x{W_feat}")
        print(f"[INFO] Padded feature map size: {H_padded}x{W_padded}")
        print(f"[INFO] Number of spatial tokens: {self.num_spatial_tokens}")

        self.transformer_blocks = nn.Sequential(
            TransformerBlock(
                embed_dim=self.vit_embed_dim, 
                num_heads=4, 
                mlp_ratio=2., 
                drop=0.1,
                attn_drop=0.1
            ),
             TransformerBlock( 
                embed_dim=self.vit_embed_dim,
                num_heads=4,
                mlp_ratio=2.,
                drop=0.1,
                attn_drop=0.1
            )
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial_tokens + 1, self.vit_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

        self.hybrid_out_channels = self.vit_embed_dim 

        self.out_channels = self.hybrid_out_channels 

        # Create dummy backbone instance for FasterRCNN initialization
        dummy_backbone = DummyBackbone(out_channels=self.out_channels)

        lightweight_anchor_gen = AnchorGenerator(
                                sizes=((32, 64, 128, 256, 512),), 
                                aspect_ratios=((0.5, 1.0, 2.0),)) 

        num_anchors = len(lightweight_anchor_gen.sizes[0]) * len(lightweight_anchor_gen.aspect_ratios[0])

        lightweight_rpn_head = RPNHead(
            in_channels=self.hybrid_out_channels, 
            num_anchors=num_anchors
        )

        self.det = FasterRCNN(
                              backbone=dummy_backbone, 
                              num_classes=NUM_CLASSES+1, 
                              rpn_anchor_generator=lightweight_anchor_gen,
                              rpn_head=lightweight_rpn_head,
                              box_nms_thresh = 0.3,
                              box_score_thresh = 0.5,
                              min_size=IMG_SIZE, max_size=IMG_SIZE
        )

        class FlattenBoxHead(nn.Module):
            def forward(self, x):
                return x.flatten(start_dim=1)

        self.det.roi_heads.box_head = FlattenBoxHead()

        roi_pool_output_size = self.det.roi_heads.box_roi_pool.output_size[0] 
        expected_in_features_for_predictor = self.hybrid_out_channels * roi_pool_output_size**2 

        self.det.roi_heads.box_predictor = FastRCNNPredictor(
            expected_in_features_for_predictor,
            num_classes=NUM_CLASSES + 1 # Including background class
        )


    def forward(self, images, targets=None):
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        image_list = ImageList(images, original_image_sizes)

        if isinstance(image_list.tensors, list):
            if len(image_list.tensors) > 0 and isinstance(image_list.tensors[0], torch.Tensor):
                cnn_input_tensor = torch.stack(image_list.tensors, dim=0)
            else:
                raise ValueError("image_list.tensors is a list, but does not contain torch.Tensors.")
        elif isinstance(image_list.tensors, torch.Tensor):
            cnn_input_tensor = image_list.tensors
        else:
             raise TypeError(f"Unexpected type for image_list.tensors: {type(image_list.tensors)}")

        image_list.tensors = cnn_input_tensor
        image_list.image_sizes = [(img.shape[-2], img.shape[-1]) for img in images] 

        cnn_features = self.cnn(image_list.tensors)

        H_feat, W_feat = cnn_features.shape[-2:]
        patch_size_for_padding = 8 
        pad_h = (patch_size_for_padding - (H_feat % patch_size_for_padding)) % patch_size_for_padding
        pad_w = (patch_size_for_padding - (W_feat % patch_size_for_padding)) % patch_size_for_padding
        cnn_features_padded = nn.functional.pad(cnn_features, (0, pad_w, 0, pad_h))
        H_padded, W_padded = cnn_features_padded.shape[-2:]

        vit_input_padded = self.conv_to_vit_embed(cnn_features_padded) 

        B, C, H, W = vit_input_padded.shape
        vit_tokens = vit_input_padded.view(B, C, H * W).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        current_num_spatial_tokens = H_padded * W_padded
        if self.pos_embed.shape[1] != current_num_spatial_tokens + 1:
             self.pos_embed = nn.Parameter(torch.zeros(1, current_num_spatial_tokens + 1, self.vit_embed_dim).to(vit_tokens.device))
             nn.init.trunc_normal_(self.pos_embed, std=.02)


        vit_input_with_cls = torch.cat((cls_tokens, vit_tokens), dim=1) 
        vit_input_with_pos = vit_input_with_cls + self.pos_embed.to(vit_input_with_cls.device) 

        vit_output = self.transformer_blocks(vit_input_with_pos) 

        spatial_tokens = vit_output[:, 1:, :] 

        hybrid_features = spatial_tokens.transpose(1, 2).view(
            B,
            self.hybrid_out_channels,
            H_padded,
            W_padded
        )

        features_dict_for_det = OrderedDict([("0", hybrid_features)])

        proposals, rpn_losses = self.det.rpn(image_list, features_dict_for_det, targets)

        detections, detector_losses = self.det.roi_heads(
            features_dict_for_det, proposals, image_list.image_sizes, targets
        )

        if targets is None:
            return detections
        else:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses
