import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
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
warnings.filterwarnings('ignore')

# pip install numpy requests pillow matplotlib tqdm scikit-learn
# python SVDCompresser.py

# Enhanced image URLs for testing
DEFAULT_IMAGE_URLS = {
    "landscape": "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg",
    "portrait": "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg",
    "nature": "https://images.pexels.com/photos/1287145/pexels-photo-1287145.jpeg",
    "city": "https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg"
}

DEFAULT_k_VALUES = [5, 25, 50, 75, 100]
DEFAULT_k_VALUES_SIMPLE = [10, 50]
DEFAULT_k_VALUES_EXTENSIVE = [1, 5, 10, 25, 50, 75, 100, 150, 200]

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Tip: Install tqdm for progress bars: pip install tqdm")

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False

try:
    from sklearn.decomposition import NMF, FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Tip: Install scikit-learn for additional matrix factorization methods")

class CompressionError(Exception):
    """Custom exception for compression errors"""
    pass

@dataclass
class CompressionConfig:
    """Configuration class for SVD compression parameters"""
    default_k_values: List[int] = None
    energy_threshold: float = 0.95
    max_image_size: Tuple[int, int] = (2048, 2048)
    output_format: str = 'png'
    save_plots: bool = True
    use_gpu: bool = False
    use_parallel: bool = True
    block_size: int = 64
    quality_metrics: List[str] = None

    def __post_init__(self):
        if self.default_k_values is None:
            self.default_k_values = DEFAULT_k_VALUES
        if self.quality_metrics is None:
            self.quality_metrics = ['psnr', 'ssim', 'mse']

class SVDImageCompressor:
    """Advanced SVD-based image compression tool with comprehensive analysis"""

    def __init__(self, output_dir: str = "output", config: CompressionConfig = None):
        self.output_dir = output_dir
        self.config = config or CompressionConfig()
        self.create_output_dir()
        self.compression_history = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def validate_k_values(self, k_values: List[int], max_rank: int) -> List[int]:
        """Validate and adjust k values based on image rank"""
        validated_k = []
        for k in k_values:
            if k <= 0:
                print(f"Warning: k={k} is invalid, skipping")
                continue
            if k > max_rank:
                print(f"Warning: k={k} exceeds max rank {max_rank}, using {max_rank}")
                k = max_rank
            validated_k.append(k)
        return sorted(list(set(validated_k)))  # Remove duplicates and sort

    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
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
        """Perform SVD with GPU acceleration if available"""
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                img_gpu = cp.asarray(img_array)
                U, S, Vt = cp.linalg.svd(img_gpu, full_matrices=False)
                return cp.asnumpy(U), cp.asnumpy(S), cp.asnumpy(Vt)
            except Exception as e:
                print(f"GPU SVD failed, falling back to CPU: {e}")

        return np.linalg.svd(img_array, full_matrices=False)

    def calculate_psnr(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float("inf")
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_ssim_simple(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Structural Similarity Index (simplified version)"""
        mu1 = np.mean(original)
        mu2 = np.mean(compressed)
        sigma1_sq = np.var(original)
        sigma2_sq = np.var(compressed)
        sigma12 = np.mean((original - mu1) * (compressed - mu2))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return max(0, min(1, ssim))  # Clamp to [0, 1]

    def calculate_vif(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """Visual Information Fidelity metric (simplified)"""
        mu1, mu2 = np.mean(original), np.mean(compressed)
        sigma1_sq, sigma2_sq = np.var(original), np.var(compressed)
        sigma12 = np.cov(original.flatten(), compressed.flatten())[0,1]

        numerator = 4 * sigma12 * mu1 * mu2
        denominator = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2)

        return numerator / (denominator + 1e-10)

    def find_knee_point(self, x_values: List, y_values: List) -> int:
        """Find knee point in curve using the kneedle algorithm (simplified)"""
        if len(x_values) < 3:
            return x_values[0]

        # Normalize values
        x_norm = np.array(x_values) / max(x_values)
        y_norm = np.array(y_values) / max(y_values)

        # Calculate differences
        differences = []
        for i in range(1, len(x_norm) - 1):
            diff = abs((y_norm[i] - y_norm[i-1]) - (y_norm[i+1] - y_norm[i]))
            differences.append(diff)

        if differences:
            knee_idx = np.argmax(differences) + 1
            return x_values[knee_idx]
        return x_values[0]

    def resize_if_needed(self, img_array: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Resize image if it's too large"""
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

        print(f"Resized image from {h}x{w} to {new_h}x{new_w}")
        return resized.astype(np.float64), True

    def safe_image_load(self, source: str, source_type: str) -> Optional[np.ndarray]:
        """Safely load image with proper error handling"""
        try:
            if source_type == "url":
                print(f"Loading image from URL...")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(source, timeout=30, headers=headers)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                print(f"Loading image from: {source}")
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")

                # Check file extension
                _, ext = os.path.splitext(source.lower())
                if ext not in self.supported_formats:
                    print(f"Warning: Unsupported format {ext}, trying anyway...")

                img = Image.open(source)

            # Convert to RGB if necessary
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.float64)
            img_array, was_resized = self.resize_if_needed(img_array)

            print(f"Image loaded successfully: {img_array.shape}")
            return img_array

        except requests.exceptions.RequestException as e:
            print(f"Error loading from URL: {e}")
            return None
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def save_compressed_images(self, recon_array: np.ndarray, k: int, image_type: str,
                             timestamp: str = None) -> str:
        """Save compressed images with timestamp"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure values are in valid range
        recon_array = np.clip(recon_array, 0, 255)

        if image_type == "gray":
            img = Image.fromarray(recon_array.astype(np.uint8), mode="L")
        else:
            img = Image.fromarray(recon_array.astype(np.uint8), mode="RGB")

        filename = f"{self.output_dir}/compressed_{image_type}_k{k}_{timestamp}.{self.config.output_format}"
        img.save(filename)
        return filename

    def find_optimal_k(self, S: np.ndarray, energy_threshold: float = None) -> int:
        """Find optimal k value based on energy threshold"""
        if energy_threshold is None:
            energy_threshold = self.config.energy_threshold

        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        optimal_k = np.argmax(cumulative_energy >= energy_threshold) + 1
        return min(optimal_k, len(S))

    def find_adaptive_k(self, S: np.ndarray, target_quality: float = 0.9) -> int:
        """Find k that maintains target quality with minimum size"""
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        quality_k = np.argmax(cumulative_energy >= target_quality) + 1

        # Consider rate-distortion optimization
        rd_scores = []
        for k in range(1, min(len(S), quality_k * 2)):
            energy = cumulative_energy[k-1]
            size_penalty = k / len(S)
            rd_score = energy - 0.1 * size_penalty  # Tunable weight
            rd_scores.append(rd_score)

        if rd_scores:
            optimal_k = np.argmax(rd_scores) + 1
            return min(optimal_k, quality_k)
        return quality_k

    def analyze_compression_efficiency(self, metrics: List[Dict]) -> Dict:
        """Analyze compression efficiency across k values"""
        k_values = [m['k'] for m in metrics]
        psnr_values = [m['psnr'] for m in metrics]
        ratio_values = [m['ratio'] for m in metrics]

        # Find knee point in PSNR curve
        knee_point = self.find_knee_point(k_values, psnr_values)

        # Calculate efficiency score
        efficiency_scores = []
        for m in metrics:
            # Balance between quality and compression
            score = m['psnr'] / (m['ratio'] * 100) if m['ratio'] > 0 else 0
            efficiency_scores.append(score)

        best_efficiency_idx = np.argmax(efficiency_scores) if efficiency_scores else 0

        return {
            'knee_point_k': knee_point,
            'best_efficiency_k': k_values[best_efficiency_idx],
            'efficiency_scores': efficiency_scores,
            'avg_efficiency': np.mean(efficiency_scores) if efficiency_scores else 0
        }

    def create_singular_values_plot(self, S: np.ndarray, optimal_k: int, save_path: str = None):
        """Create comprehensive singular values analysis plots"""
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 4, 1)
        plt.plot(S, 'b-', linewidth=2)
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        plt.title("Singular Values Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Singular Value")
        plt.xlabel("Index")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 4, 2)
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        plt.plot(cumulative_energy, 'g-', linewidth=2)
        plt.axhline(y=self.config.energy_threshold, color='r', linestyle='--',
                   label=f'{self.config.energy_threshold*100}% energy')
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'k={optimal_k}')
        plt.title("Cumulative Energy", fontsize=12, fontweight='bold')
        plt.ylabel("Energy Ratio")
        plt.xlabel("Number of Components k")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 4, 3)
        plt.loglog(S, 'm-', linewidth=2)
        plt.title("Singular Values (Log Scale)", fontsize=12, fontweight='bold')
        plt.ylabel("Value (log)")
        plt.xlabel("Index (log)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 4)
        # Energy contribution per component
        energy_contrib = (S**2) / np.sum(S**2)
        plt.bar(range(min(50, len(energy_contrib))), energy_contrib[:min(50, len(energy_contrib))],
                alpha=0.7, color='orange')
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        plt.title("Energy Contribution per Component", fontsize=12, fontweight='bold')
        plt.ylabel("Energy Ratio")
        plt.xlabel("Component Index")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def compress_single_k(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                         k: int, original: np.ndarray) -> Dict:
        """Compress with a single k value"""
        reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        m, n = original.shape

        compressed_size = k * (m + n + 1)
        original_size = m * n
        compression_ratio = compressed_size / original_size

        mse = np.mean((original - reconstructed) ** 2)
        psnr = self.calculate_psnr(original, reconstructed)
        ssim = self.calculate_ssim_simple(original, reconstructed)
        vif = self.calculate_vif(original, reconstructed)

        return {
            "k": k,
            "size_kb": compressed_size / 1024,
            "ratio": compression_ratio,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "vif": vif,
            "reconstructed": reconstructed,
            "energy_preserved": np.sum(S[:k]**2) / np.sum(S**2)
        }

    def compress_single_channel(self, img_array: np.ndarray, k_values: List[int],
                              channel_name: str = "gray") -> List[Dict]:
        """Compress single channel (grayscale) image"""
        try:
            U, S, Vt = self.svd_with_acceleration(img_array)
        except Exception as e:
            raise CompressionError(f"SVD failed for {channel_name}: {e}")

        metrics = []

        if self.config.use_parallel and len(k_values) > 2:
            # Parallel processing for multiple k values
            with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(k_values))) as executor:
                futures = [executor.submit(self.compress_single_k, U, S, Vt, k, img_array) for k in k_values]
                for future in (tqdm(futures, desc=f"Compressing {channel_name}") if TQDM_AVAILABLE else futures):
                    try:
                        result = future.result()
                        metrics.append(result)
                    except Exception as e:
                        print(f"Error processing k={future}: {e}")
        else:
            # Sequential processing
            iterator = tqdm(k_values, desc=f"Compressing {channel_name}") if TQDM_AVAILABLE else k_values
            for k in iterator:
                try:
                    result = self.compress_single_k(U, S, Vt, k, img_array)
                    metrics.append(result)
                except Exception as e:
                    print(f"Error processing k={k}: {e}")

        # Sort by k value
        metrics.sort(key=lambda x: x['k'])
        return metrics

    def compress_color_image(self, img_array: np.ndarray, k_values: List[int]) -> List[Dict]:
        """Compress color image by processing each channel separately"""
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        try:
            U_r, S_r, Vt_r = self.svd_with_acceleration(R)
            U_g, S_g, Vt_g = self.svd_with_acceleration(G)
            U_b, S_b, Vt_b = self.svd_with_acceleration(B)
        except Exception as e:
            raise CompressionError(f"SVD failed for color channels: {e}")

        m, n = R.shape
        original_color_size = m * n * 3

        metrics = []

        iterator = tqdm(k_values, desc="Compressing color") if TQDM_AVAILABLE else k_values

        for k in iterator:
            try:
                R_recon = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]
                G_recon = U_g[:, :k] @ np.diag(S_g[:k]) @ Vt_g[:k, :]
                B_recon = U_b[:, :k] @ np.diag(S_b[:k]) @ Vt_b[:k, :]

                recon_img = np.stack([R_recon, G_recon, B_recon], axis=2)
                recon_img = np.clip(recon_img, 0, 255)

                compressed_color_size = k * (m + n + 1) * 3
                compression_ratio = compressed_color_size / original_color_size

                mse = np.mean((img_array - recon_img) ** 2)
                psnr = self.calculate_psnr(img_array, recon_img)
                ssim = self.calculate_ssim_simple(img_array, recon_img)
                vif = self.calculate_vif(img_array.flatten(), recon_img.flatten())

                # Average energy preserved across channels
                energy_r = np.sum(S_r[:k]**2) / np.sum(S_r**2)
                energy_g = np.sum(S_g[:k]**2) / np.sum(S_g**2)
                energy_b = np.sum(S_b[:k]**2) / np.sum(S_b**2)
                avg_energy = (energy_r + energy_g + energy_b) / 3

                metrics.append({
                    "k": k,
                    "size_kb": compressed_color_size / 1024,
                    "ratio": compression_ratio,
                    "mse": mse,
                    "psnr": psnr,
                    "ssim": ssim,
                    "vif": vif,
                    "reconstructed": recon_img,
                    "energy_preserved": avg_energy
                })
            except Exception as e:
                print(f"Error processing color k={k}: {e}")

        return metrics

    def block_svd_compress(self, img_array: np.ndarray, k: int, block_size: int = None) -> np.ndarray:
        """Compress image using block-based SVD for memory efficiency"""
        if block_size is None:
            block_size = self.config.block_size

        h, w = img_array.shape[:2]
        is_color = len(img_array.shape) == 3

        if is_color:
            compressed = np.zeros_like(img_array)
            for c in range(3):
                compressed[:, :, c] = self._compress_channel_blocks(img_array[:, :, c], k, block_size)
        else:
            compressed = self._compress_channel_blocks(img_array, k, block_size)

        return compressed

    def _compress_channel_blocks(self, channel: np.ndarray, k: int, block_size: int) -> np.ndarray:
        """Compress a single channel using blocks"""
        h, w = channel.shape
        compressed = np.zeros_like(channel)

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = channel[i:i+block_size, j:j+block_size]
                try:
                    U, S, Vt = np.linalg.svd(block, full_matrices=False)
                    k_block = min(k, min(block.shape))
                    compressed_block = U[:, :k_block] @ np.diag(S[:k_block]) @ Vt[:k_block, :]
                    compressed[i:i+block_size, j:j+block_size] = compressed_block
                except Exception as e:
                    print(f"Block compression failed at ({i}, {j}): {e}")
                    compressed[i:i+block_size, j:j+block_size] = block

        return compressed

    def nmf_compress(self, img_array: np.ndarray, k: int) -> np.ndarray:
        """Non-negative Matrix Factorization compression"""
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available, skipping NMF compression")
            return img_array

        original_shape = img_array.shape

        if len(original_shape) == 3:
            # Color image
            compressed = np.zeros_like(img_array)
            for c in range(3):
                channel = img_array[:, :, c]
                # Ensure non-negative values
                channel = np.maximum(channel, 0)

                try:
                    nmf = NMF(n_components=k, random_state=42, max_iter=200)
                    W = nmf.fit_transform(channel)
                    H = nmf.components_
                    compressed[:, :, c] = W @ H
                except Exception as e:
                    print(f"NMF failed for channel {c}: {e}")
                    compressed[:, :, c] = channel
        else:
            # Grayscale image
            img_array = np.maximum(img_array, 0)
            try:
                nmf = NMF(n_components=k, random_state=42, max_iter=200)
                W = nmf.fit_transform(img_array)
                H = nmf.components_
                compressed = W @ H
            except Exception as e:
                print(f"NMF failed: {e}")
                compressed = img_array

        return np.clip(compressed, 0, 255)

    def create_comparison_plot(self, original_gray: np.ndarray, gray_metrics: List[Dict],
                             original_color: np.ndarray = None, color_metrics: List[Dict] = None,
                             save_path: str = None):
        """Create comprehensive comparison plots"""
        k_values = [m['k'] for m in gray_metrics]
        n_k = len(k_values)

        # Calculate grid layout
        fig_cols = min(n_k + 1, 6)
        fig_rows = (n_k + 1 + fig_cols - 1) // fig_cols

        # Grayscale comparison
        plt.figure(figsize=(20, 14))

        # Original grayscale
        plt.subplot(fig_rows, fig_cols, 1)
        plt.imshow(original_gray, cmap="gray")
        plt.title(f"Original Grayscale\nSize: {original_gray.size / 1024:.1f} KB",
                 fontsize=10, fontweight='bold')
        plt.axis("off")

        # Compressed versions
        for i, metrics in enumerate(gray_metrics):
            plt.subplot(fig_rows, fig_cols, i + 2)
            plt.imshow(metrics['reconstructed'], cmap="gray")
            plt.title(f"k = {metrics['k']}\nPSNR: {metrics['psnr']:.1f}dB | SSIM: {metrics['ssim']:.3f}\n"
                     f"Energy: {metrics['energy_preserved']:.1%} | Ratio: {metrics['ratio']:.1%}",
                     fontsize=8)
            plt.axis("off")

            # Save individual images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_compressed_images(metrics['reconstructed'], metrics['k'], "gray", timestamp)

        plt.suptitle("Grayscale SVD Compression Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(f"{save_path}_gray.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Color comparison (if available)
        if original_color is not None and color_metrics is not None:
            plt.figure(figsize=(20, 14))

            # Original color
            plt.subplot(fig_rows, fig_cols, 1)
            plt.imshow(original_color.astype(np.uint8))
            plt.title(f"Original Color\nSize: {original_color.size / 1024:.1f} KB",
                     fontsize=10, fontweight='bold')
            plt.axis("off")

            # Compressed versions
            for i, metrics in enumerate(color_metrics):
                plt.subplot(fig_rows, fig_cols, i + 2)
                plt.imshow(metrics['reconstructed'].astype(np.uint8))
                plt.title(f"k = {metrics['k']}\nPSNR: {metrics['psnr']:.1f}dB | SSIM: {metrics['ssim']:.3f}\n"
                         f"Energy: {metrics['energy_preserved']:.1%} | Ratio: {metrics['ratio']:.1%}",
                         fontsize=8)
                plt.axis("off")

                # Save individual images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_compressed_images(metrics['reconstructed'], metrics['k'], "color", timestamp)

            plt.suptitle("Color SVD Compression Analysis", fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_path and self.config.save_plots:
                plt.savefig(f"{save_path}_color.png", dpi=300, bbox_inches='tight')
            plt.show()

    def create_advanced_metrics_plot(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None,
                                   save_path: str = None):
        """Create advanced metrics analysis plots"""
        k_values = [m['k'] for m in gray_metrics]

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # PSNR comparison
        axes[0,0].plot(k_values, [m['psnr'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[0,0].plot(k_values, [m['psnr'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[0,0].set_xlabel('K values')
        axes[0,0].set_ylabel('PSNR (dB)')
        axes[0,0].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # SSIM comparison
        axes[0,1].plot(k_values, [m['ssim'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[0,1].plot(k_values, [m['ssim'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[0,1].set_xlabel('K values')
        axes[0,1].set_ylabel('SSIM')
        axes[0,1].set_title('Structural Similarity Index', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # VIF comparison
        axes[1,0].plot(k_values, [m['vif'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[1,0].plot(k_values, [m['vif'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[1,0].set_xlabel('K values')
        axes[1,0].set_ylabel('VIF')
        axes[1,0].set_title('Visual Information Fidelity', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Energy preserved
        axes[1,1].plot(k_values, [m['energy_preserved'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[1,1].plot(k_values, [m['energy_preserved'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[1,1].set_xlabel('K values')
        axes[1,1].set_ylabel('Energy Preserved')
        axes[1,1].set_title('Energy Preservation Ratio', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # Compression ratio vs Quality trade-off
        gray_psnr = [m['psnr'] for m in gray_metrics]
        gray_ratio = [m['ratio'] for m in gray_metrics]
        axes[2,0].scatter(gray_ratio, gray_psnr, c='blue', alpha=0.7, s=60, label='Grayscale')
        if color_metrics:
            color_psnr = [m['psnr'] for m in color_metrics]
            color_ratio = [m['ratio'] for m in color_metrics]
            axes[2,0].scatter(color_ratio, color_psnr, c='red', alpha=0.7, s=60, label='Color')
        axes[2,0].set_xlabel('Compression Ratio')
        axes[2,0].set_ylabel('PSNR (dB)')
        axes[2,0].set_title('Rate-Distortion Trade-off', fontweight='bold')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)

        # File size comparison
        axes[2,1].plot(k_values, [m['size_kb'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[2,1].plot(k_values, [m['size_kb'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[2,1].set_xlabel('K values')
        axes[2,1].set_ylabel('File Size (KB)')
        axes[2,1].set_title('Compressed File Size', fontweight='bold')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(f"{save_path}_advanced_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()

    def print_detailed_metrics_table(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None):
        """Print detailed metrics comparison table"""
        print("\n" + "=" * 120)
        print("DETAILED COMPRESSION METRICS")
        print("=" * 120)

        if color_metrics:
            print(f"{'k':<5} {'Gray-PSNR':<10} {'Gray-SSIM':<10} {'Gray-VIF':<10} {'Gray-Energy':<12} "
                  f"{'Color-PSNR':<11} {'Color-SSIM':<11} {'Color-VIF':<11} {'Color-Energy':<13}")
            print("-" * 120)

            for i in range(len(gray_metrics)):
                g = gray_metrics[i]
                c = color_metrics[i] if i < len(color_metrics) else {}
                print(f"{g['k']:<5} {g['psnr']:<10.1f} {g['ssim']:<10.3f} {g['vif']:<10.3f} "
                      f"{g['energy_preserved']:<12.1%} {c.get('psnr', 0):<11.1f} "
                      f"{c.get('ssim', 0):<11.3f} {c.get('vif', 0):<11.3f} {c.get('energy_preserved', 0):<13.1%}")
        else:
            print(f"{'k':<5} {'PSNR':<10} {'SSIM':<10} {'VIF':<10} {'MSE':<10} {'Energy':<10} {'Size(KB)':<10} {'Ratio':<8}")
            print("-" * 80)

            for g in gray_metrics:
                print(f"{g['k']:<5} {g['psnr']:<10.1f} {g['ssim']:<10.3f} {g['vif']:<10.3f} "
                      f"{g['mse']:<10.1f} {g['energy_preserved']:<10.1%} {g['size_kb']:<10.1f} {g['ratio']:<8.1%}")

        # Efficiency analysis
        efficiency_analysis = self.analyze_compression_efficiency(gray_metrics)
        print(f"\nEFFICIENCY ANALYSIS:")
        print(f"Knee point k: {efficiency_analysis['knee_point_k']}")
        print(f"Best efficiency k: {efficiency_analysis['best_efficiency_k']}")
        print(f"Average efficiency score: {efficiency_analysis['avg_efficiency']:.3f}")

    def save_results_json(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None,
                         image_info: Dict = None, filename: str = None):
        """Save compression results to JSON file with robust error handling"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/compression_results_{timestamp}.json"

        # Clean metrics for JSON serialization
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
                "use_parallel": bool(self.config.use_parallel)
            },
            "gray_metrics": clean_metrics(gray_metrics),
            "color_metrics": clean_metrics(color_metrics) if color_metrics else None,
            "efficiency_analysis": self.convert_numpy_types(
                self.analyze_compression_efficiency(gray_metrics)
            )
        }

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Warning: Could not save JSON results: {e}")
            self._save_simplified_results(gray_metrics, color_metrics, filename)

    def _save_simplified_results(self, gray_metrics: List[Dict], color_metrics: List[Dict], filename: str):
        """Save simplified results as fallback"""
        try:
            simplified_results = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "k_values": [int(m["k"]) for m in gray_metrics],
                    "gray_psnr": [float(m["psnr"]) for m in gray_metrics],
                    "gray_ssim": [float(m["ssim"]) for m in gray_metrics],
                },
            }
            if color_metrics:
                simplified_results["summary"]["color_psnr"] = [float(m["psnr"]) for m in color_metrics]
                simplified_results["summary"]["color_ssim"] = [float(m["ssim"]) for m in color_metrics]

            backup_filename = filename.replace(".json", "_simplified.json")
            with open(backup_filename, "w") as f:
                json.dump(simplified_results, f, indent=2)
            print(f"Simplified results saved to: {backup_filename}")
        except Exception as e2:
            print(f"Error: Could not save even simplified results: {e2}")

    def safe_compress_with_fallback(self, img_array: np.ndarray, k_values: List[int],
                                   is_color: bool = False) -> List[Dict]:
        """Compress with automatic fallback strategies"""
        try:
            if is_color:
                return self.compress_color_image(img_array, k_values)
            else:
                return self.compress_single_channel(img_array, k_values)
        except np.linalg.LinAlgError as e:
            print(f"SVD failed, trying with regularization: {e}")
            # Try with regularization
            regularized = img_array + 1e-10 * np.random.randn(*img_array.shape)
            try:
                if is_color:
                    return self.compress_color_image(regularized, k_values)
                else:
                    return self.compress_single_channel(regularized, k_values)
            except Exception as e2:
                print(f"Regularized compression failed: {e2}")
                return []
        except MemoryError as e:
            print(f"Memory error, trying block-based compression: {e}")
            # Try block-based compression
            try:
                metrics = []
                for k in k_values:
                    compressed = self.block_svd_compress(img_array, k)
                    # Calculate basic metrics
                    mse = np.mean((img_array - compressed) ** 2)
                    psnr = self.calculate_psnr(img_array, compressed)
                    ssim = self.calculate_ssim_simple(img_array, compressed)

                    metrics.append({
                        "k": k,
                        "mse": mse,
                        "psnr": psnr,
                        "ssim": ssim,
                        "reconstructed": compressed,
                        "fallback_method": "block_svd"
                    })
                return metrics
            except Exception as e3:
                print(f"Block compression failed: {e3}")
                return []
        except Exception as e:
            print(f"Compression failed: {e}")
            return []

    def compress_image(self, source: str, source_type: str = "url",
                      k_values: List[int] = None, save_results: bool = True,
                      use_advanced_analysis: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Main compression function with enhanced features"""
        start_time = time.time()

        if k_values is None:
            k_values = self.config.default_k_values.copy()

        # Load image
        img_array = self.safe_image_load(source, source_type)
        if img_array is None:
            return None, None

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray_img = np.mean(img_array, axis=2)
        else:
            gray_img = img_array
            img_array = None  # No color version

        # Validate k values
        max_rank = min(gray_img.shape)
        k_values = self.validate_k_values(k_values, max_rank)

        if not k_values:
            print("Error: No valid k values")
            return None, None

        print(f"Processing with k values: {k_values}")
        print(f"Image shape: {gray_img.shape if img_array is None else img_array.shape}")
        print(f"Max possible rank: {max_rank}")

        # SVD analysis
        try:
            U, S, Vt = self.svd_with_acceleration(gray_img)
            optimal_k = self.find_optimal_k(S)
            adaptive_k = self.find_adaptive_k(S) if use_advanced_analysis else optimal_k

            print(f"Optimal k (energy threshold): {optimal_k}")
            print(f"Adaptive k (rate-distortion): {adaptive_k}")

        except Exception as e:
            print(f"SVD analysis failed: {e}")
            return None, None

        # Create singular values plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.output_dir}/svd_analysis_{timestamp}"
        self.create_singular_values_plot(S, optimal_k, f"{plot_path}_singular_values.png")

        # Compress grayscale with fallback
        gray_metrics = self.safe_compress_with_fallback(gray_img, k_values, is_color=False)

        # Compress color (if available)
        color_metrics = None
        if img_array is not None:
            color_metrics = self.safe_compress_with_fallback(img_array, k_values, is_color=True)

        if not gray_metrics:
            print("Error: Compression failed")
            return None, None

        # Create comparison plots
        comparison_path = f"{self.output_dir}/comparison_{timestamp}"
        self.create_comparison_plot(gray_img, gray_metrics, img_array, color_metrics, comparison_path)

        # Create advanced metrics analysis
        if use_advanced_analysis:
            self.create_advanced_metrics_plot(gray_metrics, color_metrics,
                                            f"{self.output_dir}/advanced_metrics_{timestamp}")

        # Print results table
        self.print_detailed_metrics_table(gray_metrics, color_metrics)

        # Save results to JSON
        if save_results:
            image_info = {
                "source": source,
                "source_type": source_type,
                "shape": list(gray_img.shape) if img_array is None else list(img_array.shape),
                "optimal_k": int(optimal_k),
                "adaptive_k": int(adaptive_k) if use_advanced_analysis else int(optimal_k),
                "max_rank": int(max_rank),
                "processing_time": float(time.time() - start_time),
                "gpu_used": GPU_AVAILABLE and self.config.use_gpu,
                "parallel_used": self.config.use_parallel
            }
            self.save_results_json(gray_metrics, color_metrics, image_info)

        # Store in history
        self.compression_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "k_values": k_values,
            "optimal_k": optimal_k,
            "adaptive_k": adaptive_k if use_advanced_analysis else optimal_k,
            "gray_metrics": len(gray_metrics),
            "color_metrics": len(color_metrics) if color_metrics else 0
        })

        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")

        return gray_metrics, color_metrics

    def batch_compress(self, sources: List[Tuple[str, str]], k_values: List[int] = None) -> Dict:
        """Batch compress multiple images"""
        results = {}

        for i, (source, source_type) in enumerate(sources):
            print(f"\n--- Processing image {i+1}/{len(sources)}: {source} ---")
            try:
                gray_metrics, color_metrics = self.compress_image(
                    source, source_type, k_values, save_results=True
                )
                results[f"image_{i+1}"] = {
                    "source": source,
                    "gray_metrics": gray_metrics,
                    "color_metrics": color_metrics,
                    "success": gray_metrics is not None
                }
            except Exception as e:
                print(f"Error processing {source}: {e}")
                results[f"image_{i+1}"] = {
                    "source": source,
                    "error": str(e),
                    "success": False
                }

        return results

def get_user_k_values() -> List[int]:
    """Get k values from user input with enhanced options"""
    print("Choose k values for compression:")
    print("1. Use default values " + str(DEFAULT_k_VALUES))
    print("2. Enter custom values")
    print("3. Quick test: " + str(DEFAULT_k_VALUES_SIMPLE))
    print("4. Extensive analysis: " + str(DEFAULT_k_VALUES_EXTENSIVE))

    choice = input("Choose (1, 2, 3, or 4): ").strip()

    if choice == "1":
        return DEFAULT_k_VALUES
    elif choice == "2":
        while True:
            try:
                k_input = input("Enter k values separated by commas (e.g., 10,30,50,100): ")
                k_values = [int(k.strip()) for k in k_input.split(",")]

                if any(k <= 0 for k in k_values):
                    print("Error: All k values must be greater than 0")
                    continue

                k_values.sort()
                print(f"Selected k values: {k_values}")
                return k_values

            except ValueError:
                print("Error: Please enter valid integers")
    elif choice == "3":
        return DEFAULT_k_VALUES_SIMPLE
    elif choice == "4":
        return DEFAULT_k_VALUES_EXTENSIVE
    else:
        print("Invalid choice. Using default values.")
        return DEFAULT_k_VALUES

def get_image_source():
    """Get image source from user with enhanced options"""
    print("\nChoose image source:")
    print("1. From URL")
    print("2. From local file")
    print("3. Use default image (landscape)")
    print("4. Choose from sample images")

    choice = input("Choose (1, 2, 3, or 4): ").strip()

    if choice == "1":
        url = input("Enter image URL: ").strip()
        return "url", url
    elif choice == "2":
        path = input("Enter image file path: ").strip()
        return "local", path
    elif choice == "4":
        print("\nSample images:")
        for key, url in DEFAULT_IMAGE_URLS.items():
            print(f"  {key}: {url}")
        sample_choice = input("Enter sample name: ").strip().lower()
        if sample_choice in DEFAULT_IMAGE_URLS:
            return "url", DEFAULT_IMAGE_URLS[sample_choice]
        else:
            print("Invalid sample choice, using default.")
            return "url", DEFAULT_IMAGE_URLS["landscape"]
    else:
        return "url", DEFAULT_IMAGE_URLS["landscape"]

def create_cli():
    """Create enhanced command line interface"""
    parser = argparse.ArgumentParser(description='Advanced SVD Image Compression Tool')
    parser.add_argument('--input', '-i', help='Input image path or URL')
    parser.add_argument('--k-values', '-k', nargs='+', type=int,
                       default=DEFAULT_k_VALUES, help='K values for compression')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--format', choices=['png', 'jpg'], default='png', help='Output format')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot display')
    parser.add_argument('--energy-threshold', type=float, default=0.95,
                       help='Energy threshold for optimal k')
    parser.add_argument('--max-size', nargs=2, type=int, default=[2048, 2048],
                       help='Maximum image size (width height)')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--block-size', type=int, default=64, help='Block size for block-based compression')
    parser.add_argument('--batch', nargs='+', help='Batch process multiple images')
    parser.add_argument('--extensive', action='store_true', help='Use extensive k values for analysis')

    return parser.parse_args()

def main():
    """Enhanced main function"""
    print("=== ADVANCED SVD IMAGE COMPRESSION TOOL ===")
    print("Enhanced image compression using Singular Value Decomposition")
    print("Features: GPU acceleration, parallel processing, advanced metrics")
    print()

    # Check if running from command line
    if len(sys.argv) > 1:
        args = create_cli()

        # Create configuration
        config = CompressionConfig(
            default_k_values=DEFAULT_k_VALUES_EXTENSIVE if args.extensive else args.k_values,
            energy_threshold=args.energy_threshold,
            max_image_size=tuple(args.max_size),
            output_format=args.format,
            save_plots=not args.no_plots,
            use_gpu=args.use_gpu and GPU_AVAILABLE,
            use_parallel=not args.no_parallel,
            block_size=args.block_size
        )

        # Create compressor
        compressor = SVDImageCompressor(args.output, config)

        if args.batch:
            # Batch processing
            sources = []
            for source in args.batch:
                source_type = "url" if source.startswith(('http://', 'https://')) else "local"
                sources.append((source, source_type))

            print(f"Batch processing {len(sources)} images...")
            results = compressor.batch_compress(sources, args.k_values)

            successful = sum(1 for r in results.values() if r.get('success', False))
            print(f"\nBatch processing completed: {successful}/{len(sources)} successful")

        elif args.input:
            # Single image processing
            source_type = "url" if args.input.startswith(('http://', 'https://')) else "local"
            result = compressor.compress_image(args.input, source_type, args.k_values)

            if result[0] is not None:
                print("Compression completed successfully!")
            else:
                print("Compression failed!")
        else:
            print("Error: No input specified. Use --input or --batch")

    else:
        # Interactive mode
        print("GPU Acceleration:", "Available" if GPU_AVAILABLE else "Not available")
        print("Parallel Processing:", "Enabled" if multiprocessing.cpu_count() > 1 else "Limited")
        print()

        k_values = get_user_k_values()
        source_type, source_path = get_image_source()

        # Advanced options
        use_gpu = False
        if GPU_AVAILABLE:
            gpu_choice = input("\nEnable GPU acceleration? (y/n): ").strip().lower()
            use_gpu = gpu_choice in ['y', 'yes']

        # Create configuration
        config = CompressionConfig(
            default_k_values=k_values,
            use_gpu=use_gpu,
            use_parallel=True
        )

        # Create compressor
        compressor = SVDImageCompressor("output", config)

        print(f"\nStarting compression with k values: {k_values}")
        if use_gpu:
            print("GPU acceleration enabled")

        # Process image
        result = compressor.compress_image(source_path, source_type, k_values)

        if result[0] is not None:
            print(f"\nCompression completed successfully!")
            print(f"Check the 'output' folder for results")

            # Offer additional analysis
            additional = input("\nWould you like to try alternative compression methods? (y/n): ").strip().lower()
            if additional in ['y', 'yes'] and SKLEARN_AVAILABLE:
                print("Testing NMF compression...")
                try:
                    img_array = compressor.safe_image_load(source_path, source_type)
                    if img_array is not None:
                        gray_img = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                        nmf_result = compressor.nmf_compress(gray_img, k_values[len(k_values)//2])

                        # Save NMF result
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nmf_filename = f"output/nmf_compressed_{timestamp}.png"
                        Image.fromarray(nmf_result.astype(np.uint8), mode='L').save(nmf_filename)
                        print(f"NMF result saved to: {nmf_filename}")
                except Exception as e:
                    print(f"NMF compression failed: {e}")
        else:
            print("Compression failed!")

if __name__ == "__main__":
    main()