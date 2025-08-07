# src/inference/explainer.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap(img_tensor, cam, alpha=0.5):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    return Image.fromarray(overlay)

def generate_gradcam(model, input_tensor, target_class=None, save_path=None):
    cam = GradCAM(model, target_layer=model.features[-1])
    heatmap = cam.generate(input_tensor, class_idx=target_class)
    overlay = overlay_heatmap(input_tensor, heatmap)

    if save_path:
        overlay.save(save_path)
        print(f"✅ Grad-CAM saved at {save_path}")
    return overlay
