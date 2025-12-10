"""
t-SNE visualization of CNN final features on SVHN.

Usage example:

  python tsne_cnn_visual.py \
      --ckpt_path ./checkpoints/svhn_cnn_run0.pt \
      --output tsne_svhn_cnn.png \
      --max_samples 3000
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------
# Model definition (must match training script)
# ---------------------------

class SmallSVHNCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),   # index 2
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Data loader (test set only)
# ---------------------------

def get_test_loader(data_dir: str, batch_size: int, num_workers: int = 4):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    test_dataset = datasets.SVHN(
        root=data_dir,
        split='test',
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader


# ---------------------------
# Feature extraction
# ---------------------------

def extract_final_features(model, loader, device, max_samples: int = 2000):
    """
    Extract final layer features (256-dim) and labels from the test set.
    Returns:
      features: (N, 256) numpy array
      labels:   (N,) numpy array of int labels
    """
    model.eval()
    features_list = []
    labels_list = []

    # We hook the ReLU after the first Linear in classifier: classifier[2]
    # classifier = [0:Flatten, 1:Linear, 2:ReLU, 3:Dropout, 4:Linear]
    final_activations = []

    def hook_fn(module, input, output):
        # output: (B, 256)
        final_activations.append(output.detach().cpu())

    handle = model.classifier[2].register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.cpu().numpy()

            # clear for this batch
            final_activations.clear()

            _ = model(x)  # forward pass triggers hook

            # After forward, final_activations should have one tensor
            if len(final_activations) != 1:
                raise RuntimeError("Unexpected number of activations captured by hook.")

            feats_batch = final_activations[0].numpy()  # (B, 256)

            features_list.append(feats_batch)
            labels_list.append(y)

            total = sum(arr.shape[0] for arr in features_list)
            if total >= max_samples:
                break

    handle.remove()

    features = np.concatenate(features_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]

    return features, labels


# ---------------------------
# t-SNE + plotting
# ---------------------------

def run_tsne(features, labels, perplexity: float = 30.0, learning_rate: float = 200.0, random_state: int = 42):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    )
    emb_2d = tsne.fit_transform(features)
    return emb_2d


def plot_tsne(emb_2d, labels, output_path: str, title: str = "t-SNE of CNN Final Features"):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap='tab10',
        s=5,
        alpha=0.8,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Digit label")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved t-SNE plot to {output_path}")


# ---------------------------
# Main + argument parsing
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization of SVHN CNN final features.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to trained CNN checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="SVHN data directory (same as training)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for feature extraction")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Maximum number of test samples to use for t-SNE")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    parser.add_argument("--learning_rate", type=float, default=200.0,
                        help="t-SNE learning rate")
    parser.add_argument("--output", type=str, default="tsne_svhn_cnn.png",
                        help="Output PNG path for t-SNE plot")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SmallSVHNCNN().to(device)
    print(f"Loading checkpoint from {args.ckpt_path}")
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)

    # Data
    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Extract features
    print(f"Extracting up to {args.max_samples} final features...")
    feats, labels = extract_final_features(
        model,
        test_loader,
        device=device,
        max_samples=args.max_samples,
    )
    print(f"Collected features shape: {feats.shape}, labels shape: {labels.shape}")

    # Run t-SNE
    print("Running t-SNE... (this will take a bit)")
    emb_2d = run_tsne(
        feats,
        labels,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        random_state=42,
    )

    # Plot
    plot_tsne(
        emb_2d,
        labels,
        output_path=args.output,
        title="t-SNE of SVHN CNN Final Features",
    )

if __name__ == "__main__":
    main()
