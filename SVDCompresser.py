import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import argparse
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy import ndimage
from scipy.stats import entropy
import cv2
warnings.filterwarnings('ignore')

plt.style.use('default')

# pip install numpy requests pillow matplotlib tqdm scikit-learn opencv-python scipy
# python SVDCompresser.py

DEFAULT_IMAGE_URLS = {
    "landscape": "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg",
    "portrait": "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg",
    "nature": "https://images.pexels.com/photos/1287145/pexels-photo-1287145.jpeg",
    "city": "https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg",
    "abstract": "https://images.pexels.com/photos/1167355/pexels-photo-1167355.jpeg",
    "architecture": "https://images.pexels.com/photos/302769/pexels-photo-302769.jpeg",
    "food": "https://images.pexels.com/photos/376464/pexels-photo-376464.jpeg",
    "animal": "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
    "sports": "https://images.pexels.com/photos/274422/pexels-photo-274422.jpeg",
    "technology": "https://images.pexels.com/photos/442150/pexels-photo-442150.jpeg"
}

DEFAULT_k_VALUES = [5, 25, 50, 75, 100]
DEFAULT_k_VALUES_SIMPLE = [10, 50]
DEFAULT_k_VALUES_EXTENSIVE = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300]

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False

try:
    from sklearn.decomposition import NMF, FastICA, TruncatedSVD
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.feature_extraction import image as sk_image
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    print("✓ Advanced ML methods available")
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skimage import feature, filters, measure, segmentation, restoration
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    SKIMAGE_AVAILABLE = True
    print("✓ Advanced image metrics available")
except ImportError:
    SKIMAGE_AVAILABLE = False

class CompressionError(Exception):
    pass

@dataclass
class CompressionConfig:
    default_k_values: List[int] = None
    energy_threshold: float = 0.95
    max_image_size: Tuple[int, int] = (2048, 2048)
    output_format: str = 'png'
    save_plots: bool = True
    use_gpu: bool = False
    use_parallel: bool = True
    block_size: int = 64
    quality_metrics: List[str] = None
    advanced_analysis: bool = True
    create_heatmaps: bool = True
    create_animations: bool = True
    enable_denoising: bool = True
    adaptive_compression: bool = True
    roi_analysis: bool = True

    def __post_init__(self):
        if self.default_k_values is None:
            self.default_k_values = DEFAULT_k_VALUES
        if self.quality_metrics is None:
            self.quality_metrics = ['psnr', 'ssim', 'mse', 'mae', 'vif', 'entropy', 'gradient_magnitude']

class AdvancedMetrics:
    @staticmethod
    def calculate_entropy(img: np.ndarray) -> float:
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0]
        return entropy(hist / np.sum(hist), base=2)

    @staticmethod
    def calculate_gradient_magnitude(img: np.ndarray) -> float:
        grad_x = ndimage.sobel(img, axis=0)
        grad_y = ndimage.sobel(img, axis=1)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(grad_magnitude)

    @staticmethod
    def calculate_laplacian_variance(img: np.ndarray) -> float:
        laplacian = cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F)
        return laplacian.var()

    @staticmethod
    def calculate_contrast(img: np.ndarray) -> float:
        return np.std(img)

    @staticmethod
    def calculate_snr(original: np.ndarray, compressed: np.ndarray) -> float:
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - compressed) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def calculate_uqi(original: np.ndarray, compressed: np.ndarray) -> float:
        mu1, mu2 = np.mean(original), np.mean(compressed)
        sigma1_sq, sigma2_sq = np.var(original), np.var(compressed)
        sigma12 = np.cov(original.flatten(), compressed.flatten())[0, 1]
        numerator = 4 * sigma12 * mu1 * mu2
        denominator = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2)
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def calculate_spectral_angle(original: np.ndarray, compressed: np.ndarray) -> float:
        orig_flat = original.flatten()
        comp_flat = compressed.flatten()
        dot_product = np.dot(orig_flat, comp_flat)
        norm_orig = np.linalg.norm(orig_flat)
        norm_comp = np.linalg.norm(comp_flat)
        if norm_orig == 0 or norm_comp == 0:
            return 0
        cos_angle = dot_product / (norm_orig * norm_comp)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)

    @staticmethod
    def calculate_feature_similarity(original: np.ndarray, compressed: np.ndarray) -> Dict:
        if not SKIMAGE_AVAILABLE:
            return {"lbp_similarity": 0, "hog_similarity": 0}

        orig_uint8 = (original * 255 / np.max(original)).astype(np.uint8)
        comp_uint8 = (compressed * 255 / np.max(compressed)).astype(np.uint8)

        try:
            lbp_orig = feature.local_binary_pattern(orig_uint8, 8, 1, method='uniform')
            lbp_comp = feature.local_binary_pattern(comp_uint8, 8, 1, method='uniform')
            lbp_similarity = np.corrcoef(lbp_orig.flatten(), lbp_comp.flatten())[0, 1]

            hog_orig = feature.hog(orig_uint8, pixels_per_cell=(16, 16))
            hog_comp = feature.hog(comp_uint8, pixels_per_cell=(16, 16))
            hog_similarity = np.corrcoef(hog_orig, hog_comp)[0, 1]

            return {
                "lbp_similarity": lbp_similarity if not np.isnan(lbp_similarity) else 0,
                "hog_similarity": hog_similarity if not np.isnan(hog_similarity) else 0
            }
        except Exception:
            return {"lbp_similarity": 0, "hog_similarity": 0}

    @staticmethod
    def calculate_texture_metrics(img: np.ndarray) -> Dict:
        if not SKIMAGE_AVAILABLE:
            return {}

        try:
            glcm = feature.graycomatrix((img * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]

            return {
                'texture_contrast': contrast,
                'texture_dissimilarity': dissimilarity,
                'texture_homogeneity': homogeneity,
                'texture_energy': energy
            }
        except Exception:
            return {}

class SVDImageCompressor:
    def __init__(self, output_dir: str = "output", config: CompressionConfig = None):
        self.output_dir = output_dir
        self.config = config or CompressionConfig()
        self.create_output_dir()
        self.compression_history = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.metrics = AdvancedMetrics()
        self.roi_masks = {}

    def create_output_dir(self):
        directories = [
            self.output_dir,
            f"{self.output_dir}/plots",
            f"{self.output_dir}/compressed",
            f"{self.output_dir}/animations",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def validate_k_values(self, k_values: List[int], max_rank: int) -> List[int]:
        validated_k = []
        for k in k_values:
            if k <= 0:
                continue
            if k > max_rank:
                k = max_rank
            validated_k.append(k)
        return sorted(list(set(validated_k)))

    def convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        else:
            return obj

    def svd_with_acceleration(self, img_array: np.ndarray):
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                img_gpu = cp.asarray(img_array)
                U, S, Vt = cp.linalg.svd(img_gpu, full_matrices=False)
                return cp.asnumpy(U), cp.asnumpy(S), cp.asnumpy(Vt)
            except Exception:
                pass
        return np.linalg.svd(img_array, full_matrices=False)

    def calculate_perceptual_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict:
        metrics = {}

        # Edge preservation
        edges_orig = cv2.Canny((original * 255).astype(np.uint8), 100, 200)
        edges_comp = cv2.Canny((compressed * 255).astype(np.uint8), 100, 200)
        edge_preservation = np.sum(edges_orig & edges_comp) / np.sum(edges_orig | edges_comp) if np.sum(edges_orig | edges_comp) > 0 else 0
        metrics['edge_preservation'] = edge_preservation

        # Frequency domain analysis
        fft_orig = np.fft.fft2(original)
        fft_comp = np.fft.fft2(compressed)
        freq_correlation = np.corrcoef(np.abs(fft_orig).flatten(), np.abs(fft_comp).flatten())[0, 1]
        metrics['frequency_correlation'] = freq_correlation if not np.isnan(freq_correlation) else 0

        # Local variance preservation
        orig_var = ndimage.generic_filter(original, np.var, size=3)
        comp_var = ndimage.generic_filter(compressed, np.var, size=3)
        var_preservation = np.corrcoef(orig_var.flatten(), comp_var.flatten())[0, 1]
        metrics['variance_preservation'] = var_preservation if not np.isnan(var_preservation) else 0

        return metrics

    def calculate_all_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict:
        metrics = {}

        # Basic metrics
        mse = np.mean((original - compressed) ** 2)
        metrics['mse'] = mse
        metrics['mae'] = np.mean(np.abs(original - compressed))
        metrics['rmse'] = np.sqrt(mse)

        # PSNR
        if mse == 0:
            metrics['psnr'] = float("inf")
        else:
            max_pixel = 255.0
            metrics['psnr'] = 20 * np.log10(max_pixel / np.sqrt(mse))

        # SSIM
        mu1 = np.mean(original)
        mu2 = np.mean(compressed)
        sigma1_sq = np.var(original)
        sigma2_sq = np.var(compressed)
        sigma12 = np.mean((original - mu1) * (compressed - mu2))
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        metrics['ssim'] = max(0, min(1, ssim))

        # Advanced metrics
        metrics['snr'] = self.metrics.calculate_snr(original, compressed)
        metrics['uqi'] = self.metrics.calculate_uqi(original, compressed)
        metrics['spectral_angle'] = self.metrics.calculate_spectral_angle(original, compressed)

        # Information-theoretic metrics
        metrics['original_entropy'] = self.metrics.calculate_entropy(original)
        metrics['compressed_entropy'] = self.metrics.calculate_entropy(compressed)
        metrics['entropy_ratio'] = metrics['compressed_entropy'] / metrics['original_entropy'] if metrics['original_entropy'] > 0 else 0

        # Gradient metrics
        metrics['original_gradient'] = self.metrics.calculate_gradient_magnitude(original)
        metrics['compressed_gradient'] = self.metrics.calculate_gradient_magnitude(compressed)
        metrics['gradient_preservation'] = metrics['compressed_gradient'] / metrics['original_gradient'] if metrics['original_gradient'] > 0 else 0

        # Sharpness metrics
        metrics['original_laplacian_var'] = self.metrics.calculate_laplacian_variance(original)
        metrics['compressed_laplacian_var'] = self.metrics.calculate_laplacian_variance(compressed)
        metrics['sharpness_preservation'] = metrics['compressed_laplacian_var'] / metrics['original_laplacian_var'] if metrics['original_laplacian_var'] > 0 else 0

        # Contrast metrics
        metrics['original_contrast'] = self.metrics.calculate_contrast(original)
        metrics['compressed_contrast'] = self.metrics.calculate_contrast(compressed)
        metrics['contrast_preservation'] = metrics['compressed_contrast'] / metrics['original_contrast'] if metrics['original_contrast'] > 0 else 0

        # Feature-based metrics
        feature_metrics = self.metrics.calculate_feature_similarity(original, compressed)
        metrics.update(feature_metrics)

        # Texture metrics
        texture_metrics = self.metrics.calculate_texture_metrics(compressed)
        metrics.update(texture_metrics)

        # Perceptual metrics
        perceptual_metrics = self.calculate_perceptual_metrics(original, compressed)
        metrics.update(perceptual_metrics)

        # VIF
        sigma12_cov = np.cov(original.flatten(), compressed.flatten())[0,1]
        numerator = 4 * sigma12_cov * mu1 * mu2
        denominator = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2)
        metrics['vif'] = numerator / (denominator + 1e-10)

        return metrics

    def adaptive_k_selection(self, S: np.ndarray, image_complexity: float) -> List[int]:
        # Adjust k values based on image complexity
        base_k = [5, 10, 25, 50, 100]
        if image_complexity > 0.8:  # High complexity
            return [3, 5, 10, 20, 40, 80, 150, 250]
        elif image_complexity > 0.5:  # Medium complexity
            return base_k + [150, 200]
        else:  # Low complexity
            return [1, 3, 5, 10, 25, 50]

    def calculate_image_complexity(self, img: np.ndarray) -> float:
        # Calculate image complexity based on entropy and gradient
        entropy_val = self.metrics.calculate_entropy(img)
        gradient_val = self.metrics.calculate_gradient_magnitude(img)
        laplacian_val = self.metrics.calculate_laplacian_variance(img)

        # Normalize and combine
        complexity = (entropy_val / 8.0 + gradient_val / 100.0 + laplacian_val / 1000.0) / 3.0
        return min(complexity, 1.0)

    def detect_roi(self, img: np.ndarray) -> np.ndarray:
        if not SKIMAGE_AVAILABLE:
            return np.ones_like(img, dtype=bool)

        try:
            # Edge-based ROI detection
            edges = feature.canny(img)
            # Dilate edges to create regions
            roi = ndimage.binary_dilation(edges, iterations=3)
            return roi
        except Exception:
            return np.ones_like(img, dtype=bool)

    def roi_aware_compression(self, img: np.ndarray, k_values: List[int]) -> List[Dict]:
        if not self.config.roi_analysis:
            return self.compress_single_channel(img, k_values)

        roi = self.detect_roi(img)
        roi_area = np.sum(roi)
        total_area = img.size
        roi_ratio = roi_area / total_area

        # Adjust compression based on ROI
        if roi_ratio > 0.3:  # Significant ROI
            # Use higher k values for better quality
            adjusted_k = [int(k * 1.2) for k in k_values]
        else:
            # Can use lower k values
            adjusted_k = [max(1, int(k * 0.8)) for k in k_values]

        return self.compress_single_channel(img, adjusted_k)

    def create_compression_animation(self, original: np.ndarray, gray_metrics: List[Dict], save_path: str):
        if not self.config.create_animations:
            return

        import matplotlib.animation as animation

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        def animate(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            metric = gray_metrics[frame]
            reconstructed = metric['reconstructed']
            k = metric['k']

            ax1.imshow(original, cmap='gray')
            ax1.set_title('Original')
            ax1.axis('off')

            ax2.imshow(reconstructed, cmap='gray')
            ax2.set_title(f'Compressed (k={k})')
            ax2.axis('off')

            error_map = np.abs(original - reconstructed)
            im = ax3.imshow(error_map, cmap='hot')
            ax3.set_title(f'Error Map (PSNR: {metric["psnr"]:.1f}dB)')
            ax3.axis('off')

            return [ax1, ax2, ax3]

        ani = animation.FuncAnimation(fig, animate, frames=len(gray_metrics),
                                    interval=800, blit=False, repeat=True)

        ani.save(f"{save_path}_compression_animation.gif", writer='pillow', fps=1.5)
        plt.close()

    def create_interactive_plots(
        self,
        original_gray: np.ndarray,
        gray_metrics: List[Dict],
        S: np.ndarray,
        optimal_k: int,
        save_path: str = None,
    ):

        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.3)

        k_values = [m["k"] for m in gray_metrics]

        # Row 1: SVD Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(S, "b-", linewidth=2, alpha=0.8)
        ax1.axvline(
            optimal_k,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal k={optimal_k}",
        )
        ax1.set_title("Singular Values (Log Scale)", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        ax2.plot(cumulative_energy, "g-", linewidth=2)
        ax2.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95% Energy")
        ax2.axvline(optimal_k, color="r", linestyle="--", alpha=0.7)
        ax2.set_title("Cumulative Energy Distribution", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = fig.add_subplot(gs[0, 2])
        energy_contrib = (S**2) / np.sum(S**2)
        ax3.bar(
            range(min(50, len(energy_contrib))),
            energy_contrib[: min(50, len(energy_contrib))],
            alpha=0.7,
            color="orange",
        )
        ax3.axvline(optimal_k, color="r", linestyle="--", alpha=0.7)
        ax3.set_title("Energy per Component", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[0, 3])
        decay_rates = [
            S[i - 1] / S[i] if S[i] > 0 else 1 for i in range(1, min(len(S), 100))
        ]
        ax4.plot(decay_rates, "m-", linewidth=2)
        ax4.set_title("Singular Value Decay Rate", fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # Row 2: Quality Metrics
        quality_metrics = ["psnr", "ssim", "snr", "uqi"]
        colors = ["blue", "green", "red", "purple"]

        for i, (metric, color) in enumerate(zip(quality_metrics, colors)):
            ax = fig.add_subplot(gs[1, i])
            values = [m.get(metric, 0) for m in gray_metrics]
            ax.plot(k_values, values, "-o", color=color, linewidth=2, markersize=6)
            ax.set_title(f"{metric.upper()} vs K", fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Row 3: Information Theory Metrics
        info_metrics = [
            "entropy_ratio",
            "gradient_preservation",
            "sharpness_preservation",
            "contrast_preservation",
        ]

        for i, metric in enumerate(info_metrics):
            ax = fig.add_subplot(gs[2, i])
            values = [m.get(metric, 0) for m in gray_metrics]
            ax.plot(k_values, values, "o-", linewidth=2, markersize=6)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

        # Row 4: Advanced Analysis
        ax9 = fig.add_subplot(gs[3, 0])
        psnr_values = [m.get("psnr", 0) for m in gray_metrics]
        ratio_values = [m.get("ratio", 0) for m in gray_metrics]
        scatter = ax9.scatter(
            ratio_values, psnr_values, c=k_values, cmap="viridis", s=100
        )
        ax9.set_xlabel("Compression Ratio")
        ax9.set_ylabel("PSNR (dB)")
        ax9.set_title("Rate-Distortion Curve", fontweight="bold")
        ax9.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax9)

        ax10 = fig.add_subplot(gs[3, 1])
        efficiency_scores = [
            m.get("psnr", 0) / (m.get("ratio", 1) * 100) if m.get("ratio", 0) > 0 else 0
            for m in gray_metrics
        ]
        ax10.plot(k_values, efficiency_scores, "ro-", linewidth=2, markersize=6)
        ax10.set_title("Compression Efficiency", fontweight="bold")
        ax10.grid(True, alpha=0.3)

        # Polar plot
        ax11 = fig.add_subplot(gs[3, 2], projection="polar")
        metrics_to_plot = [
            "psnr",
            "ssim",
            "snr",
            "entropy_ratio",
            "gradient_preservation",
            "sharpness_preservation",
        ]
        angles = np.linspace(
            0, 2 * np.pi, len(metrics_to_plot), endpoint=False
        ).tolist()
        angles += angles[:1]

        for i, k_idx in enumerate([0, len(gray_metrics) // 2, -1]):
            if k_idx < len(gray_metrics):
                values = []
                for metric in metrics_to_plot:
                    val = gray_metrics[k_idx].get(metric, 0)
                    if metric == "psnr":
                        val = min(val / 50.0, 1.0)
                    elif metric == "snr":
                        val = min(val / 30.0, 1.0)
                    values.append(val)
                values += values[:1]

                ax11.plot(
                    angles,
                    values,
                    "o-",
                    linewidth=2,
                    label=f'k={gray_metrics[k_idx]["k"]}',
                )
                ax11.fill(angles, values, alpha=0.1)

        ax11.set_xticks(angles[:-1])
        ax11.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot])
        ax11.set_title("Multi-Metric Analysis", fontweight="bold", pad=20)
        ax11.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # Heatmap
        ax12 = fig.add_subplot(gs[3, 3])
        heatmap_metrics = [
            "psnr",
            "ssim",
            "snr",
            "entropy_ratio",
            "gradient_preservation",
        ]
        heatmap_data = []
        for metric in heatmap_metrics:
            values = [m.get(metric, 0) for m in gray_metrics]
            if len(values) > 1:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    values = [(v - min_val) / (max_val - min_val) for v in values]
            heatmap_data.append(values)

        if heatmap_data:
            im = ax12.imshow(heatmap_data, cmap="RdYlBu_r", aspect="auto")
            ax12.set_xticks(range(len(k_values)))
            ax12.set_xticklabels([f"k={k}" for k in k_values])
            ax12.set_yticks(range(len(heatmap_metrics)))
            ax12.set_yticklabels([m.replace("_", " ").title() for m in heatmap_metrics])
            ax12.set_title("Metrics Heatmap", fontweight="bold")
            plt.colorbar(im, ax=ax12, shrink=0.8)

        # Row 5: Error Analysis
        ax13 = fig.add_subplot(gs[4, 0])
        if len(gray_metrics) > 1:
            mid_idx = len(gray_metrics) // 2
            if "reconstructed" in gray_metrics[mid_idx]:
                error_map = original_gray - gray_metrics[mid_idx]["reconstructed"]
                ax13.hist(error_map.flatten(), bins=50, alpha=0.7, color="red")
                ax13.set_title(
                    f'Error Distribution (k={gray_metrics[mid_idx]["k"]})',
                    fontweight="bold",
                )
                ax13.grid(True, alpha=0.3)

        # Perceptual metrics
        ax14 = fig.add_subplot(gs[4, 1])
        perceptual_metrics = [
            "edge_preservation",
            "frequency_correlation",
            "variance_preservation",
        ]
        for metric in perceptual_metrics:
            values = [m.get(metric, 0) for m in gray_metrics if metric in m]
            if values:
                ax14.plot(
                    k_values[: len(values)],
                    values,
                    "o-",
                    linewidth=2,
                    label=metric.replace("_", " ").title(),
                )
        ax14.set_title("Perceptual Quality Metrics", fontweight="bold")
        ax14.legend()
        ax14.grid(True, alpha=0.3)

        # Texture metrics
        ax15 = fig.add_subplot(gs[4, 2])
        texture_metrics_list = [
            "texture_contrast",
            "texture_homogeneity",
            "texture_energy",
        ]
        for metric in texture_metrics_list:
            values = [m.get(metric, 0) for m in gray_metrics if metric in m]
            if values:
                ax15.plot(
                    k_values[: len(values)],
                    values,
                    "o-",
                    linewidth=2,
                    label=metric.replace("texture_", "").title(),
                )
        ax15.set_title("Texture Analysis", fontweight="bold")
        ax15.legend()
        ax15.grid(True, alpha=0.3)

        # Summary statistics
        ax16 = fig.add_subplot(gs[4, 3])
        ax16.axis("off")

        best_psnr_idx = np.argmax([m.get("psnr", 0) for m in gray_metrics])
        best_ssim_idx = np.argmax([m.get("ssim", 0) for m in gray_metrics])
        best_efficiency_idx = np.argmax(efficiency_scores)

        summary_text = f"""
COMPRESSION ANALYSIS SUMMARY

Optimal k (95% energy): {optimal_k}
Best PSNR: k={gray_metrics[best_psnr_idx]['k']} ({gray_metrics[best_psnr_idx].get('psnr', 0):.1f} dB)
Best SSIM: k={gray_metrics[best_ssim_idx]['k']} ({gray_metrics[best_ssim_idx].get('ssim', 0):.3f})
Best Efficiency: k={gray_metrics[best_efficiency_idx]['k']} ({efficiency_scores[best_efficiency_idx]:.3f})

Energy Range: {cumulative_energy[k_values[0]-1]:.1%} - {cumulative_energy[k_values[-1]-1]:.1%}
Compression Range: {min(ratio_values):.1%} - {max(ratio_values):.1%}

Total Singular Values: {len(S)}
Effective Rank: {np.sum(S > 0.01 * S[0])}
        """

        ax16.text(
            0.05,
            0.95,
            summary_text,
            transform=ax16.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        # Row 6: Image comparison
        comparison_indices = [0, len(gray_metrics) // 2, -1]
        for i, idx in enumerate(comparison_indices):
            if idx < len(gray_metrics):
                ax = fig.add_subplot(gs[5, i])
                if "reconstructed" in gray_metrics[idx]:
                    ax.imshow(gray_metrics[idx]["reconstructed"], cmap="gray")
                    ax.set_title(
                        f'k={gray_metrics[idx]["k"]}, PSNR={gray_metrics[idx].get("psnr", 0):.1f}dB'
                    )
                ax.axis("off")

        # Original image
        ax_orig = fig.add_subplot(gs[5, 3])
        ax_orig.imshow(original_gray, cmap="gray")
        ax_orig.set_title("Original Image")
        ax_orig.axis("off")

        plt.suptitle(
            "Comprehensive SVD Compression Analysis",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        if save_path and self.config.save_plots:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(
                f"{save_path}_comprehensive_analysis.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    def create_3d_analysis(self, gray_metrics: List[Dict], save_path: str = None):
        fig = plt.figure(figsize=(20, 8))

        k_values = [m['k'] for m in gray_metrics]

        # 3D Quality landscape
        ax1 = fig.add_subplot(131, projection='3d')
        K = np.array(k_values)
        metrics_3d = ['psnr', 'ssim', 'snr']

        for i, metric in enumerate(metrics_3d):
            values = [m.get(metric, 0) for m in gray_metrics]
            Y = np.full_like(K, i)
            if metric == 'psnr':
                values = [v/50 for v in values]
            elif metric == 'snr':
                values = [v/30 for v in values]
            ax1.plot(K, Y, values, marker='o', markersize=6, linewidth=2, label=metric.upper())

        ax1.set_xlabel('K Values')
        ax1.set_ylabel('Metric Type')
        ax1.set_zlabel('Normalized Score')
        ax1.set_title('3D Quality Metrics Landscape', fontweight='bold')
        ax1.legend()

        # 3D Information preservation
        ax2 = fig.add_subplot(132, projection='3d')
        info_metrics = ['entropy_ratio', 'gradient_preservation', 'sharpness_preservation']
        for i, metric in enumerate(info_metrics):
            values = [m.get(metric, 0) for m in gray_metrics]
            Y = np.full_like(K, i)
            ax2.plot(K, Y, values, marker='s', markersize=6, linewidth=2,
                    label=metric.replace('_', ' ').title())

        ax2.set_xlabel('K Values')
        ax2.set_ylabel('Information Type')
        ax2.set_zlabel('Preservation Ratio')
        ax2.set_title('3D Information Preservation', fontweight='bold')
        ax2.legend()

        # 3D Quality-compression space
        ax3 = fig.add_subplot(133, projection='3d')
        psnr_vals = [m.get('psnr', 0) for m in gray_metrics]
        ssim_vals = [m.get('ssim', 0) for m in gray_metrics]
        ratio_vals = [m.get('ratio', 0) for m in gray_metrics]

        scatter = ax3.scatter(psnr_vals, ssim_vals, ratio_vals, c=k_values, cmap='viridis', s=100)
        ax3.set_xlabel('PSNR (dB)')
        ax3.set_ylabel('SSIM')
        ax3.set_zlabel('Compression Ratio')
        ax3.set_title('3D Quality-Compression Space', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, shrink=0.8, label='K Value')

        plt.tight_layout()

        if save_path and self.config.save_plots:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}_3d_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def compress_single_k(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                         k: int, original: np.ndarray) -> Dict:
        reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        m, n = original.shape

        compressed_size = k * (m + n + 1)
        original_size = m * n
        compression_ratio = compressed_size / original_size

        metrics = self.calculate_all_metrics(original, reconstructed)

        metrics.update({
            "k": k,
            "size_kb": compressed_size / 1024,
            "ratio": compression_ratio,
            "reconstructed": reconstructed,
            "energy_preserved": np.sum(S[:k]**2) / np.sum(S**2),
            "compression_efficiency": np.sum(S[:k]**2) / k if k > 0 else 0,
            "spectral_flatness": np.exp(np.mean(np.log(S[:k] + 1e-10))) / np.mean(S[:k]) if k > 0 else 0
        })

        return metrics

    def compress_single_channel(self, img_array: np.ndarray, k_values: List[int],
                              channel_name: str = "gray") -> List[Dict]:
        try:
            U, S, Vt = self.svd_with_acceleration(img_array)
        except Exception as e:
            raise CompressionError(f"SVD failed for {channel_name}: {e}")

        metrics = []

        if self.config.use_parallel and len(k_values) > 2:
            with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(k_values))) as executor:
                futures = [executor.submit(self.compress_single_k, U, S, Vt, k, img_array) for k in k_values]
                for future in (tqdm(futures, desc=f"Compressing {channel_name}") if TQDM_AVAILABLE else futures):
                    try:
                        result = future.result()
                        metrics.append(result)
                    except Exception:
                        pass
        else:
            iterator = tqdm(k_values, desc=f"Compressing {channel_name}") if TQDM_AVAILABLE else k_values
            for k in iterator:
                try:
                    result = self.compress_single_k(U, S, Vt, k, img_array)
                    metrics.append(result)
                except Exception:
                    pass

        metrics.sort(key=lambda x: x['k'])
        return metrics

    def denoise_image(self, img_array: np.ndarray) -> np.ndarray:
        if not self.config.enable_denoising or not SKIMAGE_AVAILABLE:
            return img_array

        try:
            denoised = restoration.denoise_tv_chambolle(img_array, weight=0.1)
            return denoised
        except Exception:
            return img_array

    def compress_image(self, source: str, source_type: str = "url",
                      k_values: List[int] = None, save_results: bool = True,
                      use_advanced_analysis: bool = True) -> Tuple[List[Dict], List[Dict]]:
        start_time = time.time()

        if k_values is None:
            k_values = self.config.default_k_values.copy()

        img_array = self.safe_image_load(source, source_type)
        if img_array is None:
            return None, None

        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            img_array = None

        # Denoise if enabled
        gray_img = self.denoise_image(gray_img)

        # Calculate image complexity for adaptive compression
        if self.config.adaptive_compression:
            complexity = self.calculate_image_complexity(gray_img)
            print(f"Image complexity: {complexity:.3f}")
            k_values = self.adaptive_k_selection(np.linalg.svd(gray_img, compute_uv=False), complexity)

        max_rank = min(gray_img.shape)
        k_values = self.validate_k_values(k_values, max_rank)

        if not k_values:
            print("Error: No valid k values")
            return None, None

        print(f"Processing with k values: {k_values}")
        print(f"Image shape: {gray_img.shape}")
        print(f"Max possible rank: {max_rank}")

        try:
            U, S, Vt = self.svd_with_acceleration(gray_img)
            optimal_k = self.find_optimal_k(S)
            print(f"Optimal k (95% energy): {optimal_k}")
        except Exception as e:
            print(f"SVD analysis failed: {e}")
            return None, None

        # Use ROI-aware compression if enabled
        if self.config.roi_analysis:
            gray_metrics = self.roi_aware_compression(gray_img, k_values)
        else:
            gray_metrics = self.safe_compress_with_fallback(gray_img, k_values, is_color=False)

        if not gray_metrics:
            print("Error: Compression failed")
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_path = f"{self.output_dir}/plots/analysis_{timestamp}"

        if use_advanced_analysis:
            self.create_interactive_plots(gray_img, gray_metrics, S, optimal_k, analysis_path)
            self.create_3d_analysis(gray_metrics, analysis_path)

            if self.config.create_animations:
                self.create_compression_animation(gray_img, gray_metrics, analysis_path)

        self.print_enhanced_metrics_table(gray_metrics)

        # Save compressed images
        compressed_dir = f"{self.output_dir}/compressed"
        os.makedirs(compressed_dir, exist_ok=True)

        for metric in gray_metrics:
            if 'reconstructed' in metric:
                filename = f"{compressed_dir}/compressed_k{metric['k']}_{timestamp}.{self.config.output_format}"
                img_pil = Image.fromarray((metric['reconstructed'] * 255).astype(np.uint8), mode='L')
                img_pil.save(filename)

        if save_results:
            self.save_results_json(gray_metrics, None, {
                "source": source,
                "source_type": source_type,
                "shape": list(gray_img.shape),
                "optimal_k": int(optimal_k),
                "processing_time": float(time.time() - start_time),
                "complexity": float(self.calculate_image_complexity(gray_img)) if self.config.adaptive_compression else None
            }, f"{self.output_dir}/results_{timestamp}.json")

        self.compression_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "k_values": k_values,
            "optimal_k": optimal_k,
            "complexity": self.calculate_image_complexity(gray_img) if self.config.adaptive_compression else None
        })

        processing_time = time.time() - start_time
        print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
        print(f"✓ Results saved to: {self.output_dir}")

        return gray_metrics, None

    def print_enhanced_metrics_table(self, gray_metrics: List[Dict]):
        print("\n" + "=" * 180)
        print("COMPREHENSIVE COMPRESSION METRICS")
        print("=" * 180)

        print(f"{'k':<4} {'PSNR':<8} {'SSIM':<8} {'SNR':<8} {'UQI':<8} {'Entropy':<8} {'Gradient':<8} "
              f"{'Sharpness':<9} {'Contrast':<8} {'Edge':<8} {'Freq':<8} {'Texture':<8} {'Efficiency':<10}")
        print("-" * 180)

        for m in gray_metrics:
            print(f"{m['k']:<4} {m.get('psnr', 0):<8.1f} {m.get('ssim', 0):<8.3f} "
                  f"{m.get('snr', 0):<8.1f} {m.get('uqi', 0):<8.3f} "
                  f"{m.get('entropy_ratio', 0):<8.3f} {m.get('gradient_preservation', 0):<8.3f} "
                  f"{m.get('sharpness_preservation', 0):<9.3f} {m.get('contrast_preservation', 0):<8.3f} "
                  f"{m.get('edge_preservation', 0):<8.3f} {m.get('frequency_correlation', 0):<8.3f} "
                  f"{m.get('texture_energy', 0):<8.3f} {m.get('compression_efficiency', 0):<10.3f}")

    def safe_image_load(self, source: str, source_type: str) -> Optional[np.ndarray]:
        try:
            if source_type == "url":
                print(f"📥 Loading image from URL...")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(source, timeout=30, headers=headers)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                print(f"📁 Loading image from: {source}")
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")
                img = Image.open(source)

            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.float64)
            img_array, was_resized = self.resize_if_needed(img_array)

            print(f"✓ Image loaded successfully: {img_array.shape}")
            return img_array
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None

    def resize_if_needed(self, img_array: np.ndarray) -> Tuple[np.ndarray, bool]:
        h, w = img_array.shape[:2]
        max_h, max_w = self.config.max_image_size

        if h <= max_h and w <= max_w:
            return img_array, False

        scale = min(max_h/h, max_w/w)
        new_h, new_w = int(h*scale), int(w*scale)

        if len(img_array.shape) == 3:
            resized = np.array(Image.fromarray(img_array.astype(np.uint8)).resize((new_w, new_h)))
        else:
            resized = np.array(Image.fromarray(img_array.astype(np.uint8), mode='L').resize((new_w, new_h)))

        print(f"🔄 Resized from {h}x{w} to {new_h}x{new_w}")
        return resized.astype(np.float64), True

    def find_optimal_k(self, S: np.ndarray, energy_threshold: float = None) -> int:
        if energy_threshold is None:
            energy_threshold = self.config.energy_threshold

        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        optimal_k = np.argmax(cumulative_energy >= energy_threshold) + 1
        return min(optimal_k, len(S))

    def safe_compress_with_fallback(self, img_array: np.ndarray, k_values: List[int],
                                   is_color: bool = False) -> List[Dict]:
        try:
            return self.compress_single_channel(img_array, k_values)
        except Exception as e:
            print(f"❌ Compression failed: {e}")
            return []

    def save_results_json(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None,
                         image_info: Dict = None, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/compression_results_{timestamp}.json"

        def clean_metrics(metrics):
            cleaned = []
            for m in metrics:
                clean_m = {k: v for k, v in m.items() if k != "reconstructed"}
                clean_m = self.convert_numpy_types(clean_m)
                cleaned.append(clean_m)
            return cleaned

        results = {
            "timestamp": datetime.now().isoformat(),
            "image_info": self.convert_numpy_types(image_info or {}),
            "config": {
                "energy_threshold": float(self.config.energy_threshold),
                "max_image_size": list(self.config.max_image_size),
                "output_format": str(self.config.output_format),
                "use_gpu": bool(self.config.use_gpu),
                "adaptive_compression": bool(self.config.adaptive_compression),
                "roi_analysis": bool(self.config.roi_analysis)
            },
            "gray_metrics": clean_metrics(gray_metrics),
            "color_metrics": clean_metrics(color_metrics) if color_metrics else None
        }

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save JSON results: {e}")

def get_image_source():
    print("\n🖼️  Choose image source:")
    print("1. From URL")
    print("2. From local file")
    print("3. Use sample image")

    choice = input("Choose (1-3): ").strip()

    if choice == "1":
        url = input("Enter image URL: ").strip()
        return "url", url
    elif choice == "2":
        path = input("Enter image file path: ").strip()
        return "local", path
    else:
        print("\n🖼️  Sample images:")
        for i, (key, url) in enumerate(DEFAULT_IMAGE_URLS.items(), 1):
            print(f"{i:2d}. {key.title()}")

        sample_choice = input("Choose sample (1-10): ").strip()
        try:
            idx = int(sample_choice) - 1
            key = list(DEFAULT_IMAGE_URLS.keys())[idx]
            return "url", DEFAULT_IMAGE_URLS[key]
        except (ValueError, IndexError):
            return "url", DEFAULT_IMAGE_URLS["landscape"]

def get_compression_settings():
    print("\n⚙️  Compression settings:")
    print("1. Quick test (simple k values)")
    print("2. Standard analysis")
    print("3. Extensive analysis")
    print("4. Custom k values")

    choice = input("Choose (1-4): ").strip()

    if choice == "1":
        return DEFAULT_k_VALUES_SIMPLE
    elif choice == "3":
        return DEFAULT_k_VALUES_EXTENSIVE
    elif choice == "4":
        k_input = input("Enter k values (comma-separated): ")
        try:
            return [int(k.strip()) for k in k_input.split(",")]
        except ValueError:
            return DEFAULT_k_VALUES
    else:
        return DEFAULT_k_VALUES

def main():
    print("🎯 ULTRA-ADVANCED SVD IMAGE COMPRESSION TOOL")
    print("=" * 60)
    print("Features: 20+ metrics, 3D plots, animations, ROI analysis")
    print("GPU:", "✓" if GPU_AVAILABLE else "❌")
    print("Advanced ML:", "✓" if SKLEARN_AVAILABLE else "❌")
    print("Advanced Metrics:", "✓" if SKIMAGE_AVAILABLE else "❌")
    print()

    source_type, source_path = get_image_source()
    k_values = get_compression_settings()

    # Advanced options
    print("\n🔧 Advanced options:")
    adaptive = input("Enable adaptive compression? (y/n): ").strip().lower() == 'y'
    roi_analysis = input("Enable ROI analysis? (y/n): ").strip().lower() == 'y'
    animations = input("Create animations? (y/n): ").strip().lower() == 'y'
    gpu = input("Enable GPU acceleration? (y/n): ").strip().lower() == 'y' if GPU_AVAILABLE else False

    config = CompressionConfig(
        default_k_values=k_values,
        adaptive_compression=adaptive,
        roi_analysis=roi_analysis,
        create_animations=animations,
        use_gpu=gpu,
        advanced_analysis=True,
        create_heatmaps=True
    )

    compressor = SVDImageCompressor("output", config)

    print(f"\n🚀 Starting compression analysis...")
    result = compressor.compress_image(source_path, source_type, k_values,
                                     save_results=True, use_advanced_analysis=True)

    if result[0] is not None:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Check '{compressor.output_dir}' for results:")
        print(f"   • Comprehensive plots")
        print(f"   • Compressed images")
        if animations:
            print(f"   • Compression animations")
        print(f"   • JSON results")
    else:
        print("❌ Analysis failed!")

if __name__ == "__main__":
    main()
