!git clone https://github.com/YQ-XiaMLTech/ALS-classification.git
%cd ALS-classification
import sys
sys.path.append("model")

from DenseNet import SE_DenseNet
import torch
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from model.SE_ResNet18 import SE_ResNet18
from process_dataset import pre_data
from torchvision.transforms.functional import to_pil_image
import os

# Cargar el modelo
model = SE_DenseNet(num_classes=2, dropout_rate=0.5)
!wget https://github.com/YQ-XiaMLTech/ALS-classification/raw/main/saves/model_fullDenseSE.pth -P /content/
model = torch.load('/content/model_fullDenseSE.pth', map_location='cpu', weights_only=False)

def apply_gradcam(input_tensor, model, target_layer, target_category):
    activations = None
    gradients = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor.unsqueeze(0))
    model.zero_grad()

    if target_category is None:
        target_category = output.argmax(dim=1)

    score = output[:, target_category].squeeze()
    score.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    hook_forward.remove()
    hook_backward.remove()

    return heatmap

def main():
    # Ruta donde se encuentran las imágenes (puedes poner la ruta de las imágenes locales o si usas un repositorio)
    img_folder_path = "dataset/AptamerROIs020623"  # Ajusta la ruta si es necesario

    # Obtener lista de todas las imágenes en la carpeta
    img_files = [f for f in os.listdir(img_folder_path) if f.endswith('.tif')]

    dataset_path = "dataset/AptamerROIs020623"  # Ajusta si es necesario
    target_category = 2

    mean, std = pre_data.compute_mean_std(dataset_path)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    for img_filename in img_files:
        img_path = os.path.join(img_folder_path, img_filename)

        img_original = Image.open(img_path)
        img = img_original.convert('RGB')
        img_tensor = transform(img)

        target_layer = model.features[-1]  # Última capa convolucional
        heatmap = apply_gradcam(img_tensor, model, target_layer, target_category)

        heatmap_np = heatmap.cpu().detach().numpy()
        heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np))
        heatmap_pil = to_pil_image(heatmap_np, mode='F').resize(img_original.size, PIL.Image.BICUBIC)

        overlay_np = np.array(heatmap_pil)

        # Crear dos máscaras con umbrales diferentes
        mask_gradcam_70 = overlay_np > (0.7 * np.max(overlay_np))
        mask_gradcam_90 = overlay_np > (0.9 * np.max(overlay_np))

        # Crear colormap
        overlay_colormap = cm.jet(overlay_np / np.max(overlay_np))
        overlay_colormap_rgb = (overlay_colormap[..., :3] * 255).astype(np.uint8)

        # Visualizar resultados
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(img_original)
        plt.title(f"Original: {img_filename}")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(img_original)
        plt.imshow(overlay_colormap_rgb, alpha=0.5)
        plt.title(f"Grad-CAM: {img_filename}")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(img_original)
        plt.imshow(mask_gradcam_70, cmap='gray', alpha=0.5)
        plt.title(f"Threshold 70%: {img_filename}")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(img_original)
        plt.imshow(mask_gradcam_90, cmap='gray', alpha=0.5)
        plt.title(f"Threshold 90%: {img_filename}")
        plt.axis('off')

        plt.show()

if __name__ == '__main__':
    main()