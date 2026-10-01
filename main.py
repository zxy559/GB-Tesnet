# Main settings used in the paper: ResNet-18, 10 prototypes per class, Top-k=3, `beta=10`, `d = -cos`.
# Training: AdamW, lr=1e-3, 100 epochs, grad clip 1.0. Point-based ablation: `use_ball=False`.

from model import construct_GBTesNet
from loss import gbtesnet_loss

model = construct_GBTesNet(num_classes=2)  # 3 for COVIDx
logits, min_ball = model(images)
loss, logs = gbtesnet_loss(logits, labels, min_ball, model)
