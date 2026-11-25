import glob as gl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import os

from src.core.CifarDataset import CifarDataset
from src.core.FeatureExtraction.HandcraftedFeatureExtractor import (
    HandcraftedFeatureExtractor
)
from src.core.models.Settings import settings

database_root_dir = os.path.abspath(settings.database_dir)
databse_files = gl.glob(database_root_dir + "/" + "data_batch_[1-9]")
test_file = gl.glob(database_root_dir + "/" + "test_batch")
loaded_db = [CifarDataset(file) for file in databse_files]

train_x = np.concatenate([db.get_X() for db in loaded_db], axis=0)
train_y = np.concatenate([db.get_Y() for db in loaded_db], axis=0)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CELL = 8
N_BINS = 9
BLOCK = 2
SIGNED = False
BIN_CENTERED = True
FLIP_Y = False
EDGE_ROTATION = np.pi / 2


def to_gray(img):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img @ np.array([0.299, 0.587, 0.114])
    return img


def descriptor_to_cell_grid(feat, img_hw):
    """Flat descriptor -> (cells_y, cells_x, bins), validating the length."""
    h, w = img_hw
    cells_y, cells_x = h // CELL, w // CELL

    if BLOCK is None:
        expected = cells_y * cells_x * N_BINS
        _check(feat, expected, f"{cells_y}x{cells_x} cells x {N_BINS} bins")
        return feat.reshape(cells_y, cells_x, N_BINS)

    by, bx = cells_y - BLOCK + 1, cells_x - BLOCK + 1
    expected = by * bx * BLOCK * BLOCK * N_BINS
    _check(feat, expected,
           f"{by}x{bx} blocks x {BLOCK}x{BLOCK} cells x {N_BINS} bins")
    f = feat.reshape(by, bx, BLOCK, BLOCK, N_BINS)

    # Overlapping blocks give each cell several differently-normalised
    # histograms; average them so there is one histogram per cell.
    grid = np.zeros((cells_y, cells_x, N_BINS))
    count = np.zeros((cells_y, cells_x, 1))
    for i in range(by):
        for j in range(bx):
            grid[i:i + BLOCK, j:j + BLOCK] += f[i, j]
            count[i:i + BLOCK, j:j + BLOCK] += 1
    return grid / count


def _check(feat, expected, desc):
    if feat.size != expected:
        raise ValueError(
            f"Descriptor has {feat.size} values but CONFIG implies {expected} "
            f"({desc}). Fix CELL / N_BINS / BLOCK to match your HOG."
        )


def draw_hog(ax, img_gray, cell_hists, scale=0.95):
    ny, nx, n_bins = cell_hists.shape
    span = 2 * np.pi if SIGNED else np.pi
    offset = 0.5 if BIN_CENTERED else 0.0
    theta = (np.arange(n_bins) + offset) * span / n_bins
    edge = theta + EDGE_ROTATION
    dx, dy = np.cos(edge), np.sin(edge)
    if FLIP_Y:
        dy = -dy

    vmax = cell_hists.max() + 1e-12
    half = CELL / 2 * scale
    segs = []
    for i in range(ny):
        for j in range(nx):
            cx = j * CELL + (CELL - 1) / 2
            cy = i * CELL + (CELL - 1) / 2
            for b in range(n_bins):
                r = half * cell_hists[i, j, b] / vmax
                if r <= 0:
                    continue
                segs.append([(cx - dx[b] * r, cy - dy[b] * r),
                             (cx + dx[b] * r, cy + dy[b] * r)])

    ax.imshow(img_gray, cmap="gray", alpha=0.5)
    ax.add_collection(LineCollection(segs, colors="yellow", linewidths=1.2))
    # grid lines show the cell boundaries
    for k in range(1, max(ny, nx)):
        ax.axhline(k * CELL - 0.5, color="white", lw=0.3, alpha=0.4)
        ax.axvline(k * CELL - 0.5, color="white", lw=0.3, alpha=0.4)
    ax.set_xlim(-0.5, nx * CELL - 0.5)
    ax.set_ylim(ny * CELL - 0.5, -0.5)
    ax.set_title("Own HOG (lines = edge direction, length = bin value)")
    ax.axis("off")


def plot_hog_features(x, y, per_class=5):
    feature_extractor = HandcraftedFeatureExtractor(10)
    buckets = [[] for _ in range(10)]
    for img, label in zip(x, y):
        buckets[int(label)].append(img)

    for i in range(10):
        images = np.stack(buckets[i][-per_class:])
        hog_features = feature_extractor._hog_features(images)

        for idx in range(len(images)):
            gray = to_gray(images[idx])
            grid = descriptor_to_cell_grid(hog_features[idx], gray.shape)

            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            axes[0].imshow(images[idx].astype(np.uint8))
            axes[0].set_title(f"Input image (class {i})")
            draw_hog(axes[1], gray, grid)
            plt.tight_layout()
            plt.show()


def sanity_check_synthetic():
    """Vertical stripes -> only vertical lines; flat image -> empty plot."""
    extractor = HandcraftedFeatureExtractor(10)
    stripes = np.zeros((32, 32), dtype=np.float64)
    stripes[:, ::4] = 255
    flat = np.full((32, 32), 128.0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, im, name in zip(axes, (stripes, flat), ("stripes", "flat")):
        rgb = np.stack([im] * 3, axis=-1)[None]       # (1, 32, 32, 3)
        feat = extractor._hog_features(rgb)[0]
        draw_hog(ax, im, descriptor_to_cell_grid(feat, im.shape))
        ax.set_title(name)
    plt.tight_layout()
    plt.show()


def plot_green_values(x, y):
    buckets: list[list[np.typing.NDArray]] = [[] for _ in range(10)]
    for img, label in zip(x, y):
        buckets[int(label)].append(img)

    fig, ax = plt.subplots(figsize=(10, 5))

    for class_idx in range(10):
        class_imgs = np.stack(buckets[class_idx], axis=0)
        green_per_image = class_imgs[:, :, :, 1].mean(axis=(1, 2))
        y_vals = np.sort(green_per_image)
        x_vals = np.arange(len(y_vals))

        ax.plot(
            x_vals,
            y_vals,
            label=CIFAR10_CLASSES[class_idx],
            linewidth=1.2,
            alpha=0.85
        )

    ax.set_title("Mean Green Channel Value per Image Across CIFAR-10 Classes")
    ax.set_xlabel("Sorted Image Index")
    ax.set_ylabel("Mean Green Value")
    ax.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()

    return None


def plot_blue_values(x, y):
    buckets: list[list[np.typing.NDArray]] = [[] for _ in range(10)]
    for img, label in zip(x, y):
        buckets[int(label)].append(img)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(10):
        class_imgs = np.stack(buckets[i], axis=0)
        blue_per_image = np.mean(class_imgs[:, :, :, 2], axis=(1, 2))
        y_vals = np.sort(blue_per_image)
        x_vals = np.arange(len(y_vals))

        ax.plot(
            x_vals,
            y_vals,
            label=CIFAR10_CLASSES[i],
            linewidth=1.2,
            alpha=0.85
        )

    ax.set_title("Mean Blue Channel Value per Image Across CIFAR-10 Classes")
    ax.set_xlabel("Sorted Image Index")
    ax.set_ylabel("Mean Blue Value")
    ax.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_red_values(x, y):
    buckets: list[list[np.typing.NDArray]] = [[] for _ in range(10)]
    for img, label in zip(x, y):
        buckets[int(label)].append(img)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(10):
        class_imgs = np.stack(buckets[i], axis=0)
        red_per_image = np.mean(class_imgs[:, :, :, 0], axis=(1, 2))
        y_vals = np.sort(red_per_image)
        x_vals = np.arange(len(y_vals))

        ax.plot(
            x_vals,
            y_vals,
            label=CIFAR10_CLASSES[i],
            linewidth=1.2,
            alpha=0.85
        )

    ax.set_title("Mean Red Channel Value per Image Across CIFAR-10 Classes")
    ax.set_xlabel("Sorted Image Index")
    ax.set_ylabel("Mean Red Value")
    ax.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()
    return None


if __name__ == "__main__":
    plot_green_values(train_x, train_y)
    plot_blue_values(train_x, train_y)
    plot_red_values(train_x, train_y)
    plot_hog_features(train_x, train_y)
