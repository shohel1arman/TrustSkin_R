"""Model factory built on timm.

All four TrustSkin backbones are created through timm so that later phases
(MC Dropout, attention rollout, Grad-CAM target layers) work uniformly.
`drop_rate` inserts dropout before the classifier head; this is what MC Dropout
in Phase 3 re-activates at inference time.
"""

import timm
import torch.nn as nn

# Config name -> timm model name
MODEL_REGISTRY = {
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "vit_b16": "vit_base_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    # small model used only for pipeline smoke tests
    "resnet18": "resnet18",
}


def create_model(name: str, num_classes: int = 7, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    model = timm.create_model(
        MODEL_REGISTRY[name],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Set dropout layers to train mode while keeping everything else in eval.

    Used in Phase 3 (uncertainty). Included here so the interface is fixed early.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model
