import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import config
from models.image_model import ImageModel
from data.dataset import MultimodalDataset

def main():
    save_dir = os.path.join(config.RESULT_DIR, "analysis/gradcam")
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = ImageModel().to(config.DEVICE)
    state = torch.load(os.path.join(config.RESULT_DIR, "image_model/best_model.pth"), map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Select sample IDs (first 4 of each subtype)
    labels_df = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    c1_ids = labels_df[labels_df.label==0]['sample_id'].tolist()[:4]
    c2_ids = labels_df[labels_df.label==1]['sample_id'].tolist()[:4]
    sample_ids = c1_ids + c2_ids

    dataset = MultimodalDataset(sample_ids, mode='image', is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Hook for Grad-CAM
    features = []
    grads = []
    def forward_hook(module, inp, out):
        features.append(out)
        out.retain_grad()
    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    target_layer = model.resnet[-2][-1]  # last conv layer
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    results = []
    for i, (patches, label) in enumerate(loader):
        sid = sample_ids[i]
        patches = patches.to(config.DEVICE)
        logits = model(patches)
        probs = F.softmax(logits, dim=1)
        cls_idx = probs.argmax(1).item()
        prob_c2 = probs[0,1].item()
        # Backward
        model.zero_grad()
        logits[0, cls_idx].backward()

        # Compute CAM
        fmap = features[0].mean(dim=1)  # B, C, H, W -> B, H, W
        gmap = grads[0].mean(dim=1)
        weights = gmap.mean(dim=(1,2), keepdim=True)
        cam = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        cam = cam[0,0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Visualization
        img = patches[0,0].permute(1,2,0).cpu().numpy()
        img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img + 0.4 * heatmap

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.imshow(img); plt.axis('off'); plt.title('Original')
        plt.subplot(1,2,2); plt.imshow(overlay); plt.axis('off'); plt.title('Grad-CAM')
        plt.suptitle(f"{sid} | True: {label.item()} | Pred C2 prob: {prob_c2:.3f}")
        plt.savefig(os.path.join(save_dir, f"{sid}_gradcam.png"), dpi=150, bbox_inches='tight')
        plt.close()

        results.append({'sample_id': sid, 'true_label': label.item(), 'pred_prob_C2': prob_c2})

    pd.DataFrame(results).to_csv(os.path.join(save_dir, "gradcam_results.csv"), index=False)
    fwd_handle.remove()
    bwd_handle.remove()
    print("Grad-CAM analysis completed.")

if __name__ == "__main__":
    main()