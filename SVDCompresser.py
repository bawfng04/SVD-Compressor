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
warnings.filterwarnings('ignore')

# pip install numpy requests pillow matplotlib tqdm
# python SVDCompresser.py

# consts
DEFAULT_IMAGE_URL = "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg"
DEFAULT_k_VALUES = [1, 5, 25, 50, 75, 100]
DEFAULT_k_VALUES_SIMPLE = [10, 50]


try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Tip: Install tqdm for progress bars: pip install tqdm")

@dataclass
class CompressionConfig:
    """Configuration class for SVD compression parameters"""
    default_k_values: List[int] = None
    energy_threshold: float = 0.95
    max_image_size: Tuple[int, int] = (2048, 2048)
    output_format: str = 'png'
    save_plots: bool = True

    def __post_init__(self):
        if self.default_k_values is None:
            self.default_k_values = DEFAULT_k_VALUES

class SVDImageCompressor:
    """SVD-based image compression tool with comprehensive analysis"""

    def __init__(self, output_dir: str = "output", config: CompressionConfig = None):
        self.output_dir = output_dir
        self.config = config or CompressionConfig()
        self.create_output_dir()
        self.compression_history = []

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
        return validated_k

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
        return ssim

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
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                print(f"Loading image from: {source}")
                img = Image.open(source)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.float64)
            img_array, was_resized = self.resize_if_needed(img_array)

            print(f"Image loaded successfully: {img_array.shape}")
            return img_array

        except requests.exceptions.RequestException as e:
            print(f"Error loading from URL: {e}")
            return None
        except FileNotFoundError:
            print(f"Error: File not found: {source}")
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def save_compressed_images(self, recon_array: np.ndarray, k: int, image_type: str,
                             timestamp: str = None) -> str:
        """Save compressed images with timestamp"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        return optimal_k

    def create_singular_values_plot(self, S: np.ndarray, optimal_k: int, save_path: str = None):
        """Create comprehensive singular values analysis plots"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(S, 'b-', linewidth=2)
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        plt.title("Singular Values Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Singular Value")
        plt.xlabel("Index")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 3, 2)
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

        plt.subplot(1, 3, 3)
        plt.loglog(S, 'm-', linewidth=2)
        plt.title("Singular Values (Log Scale)", fontsize=12, fontweight='bold')
        plt.ylabel("Value (log)")
        plt.xlabel("Index (log)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def compress_single_channel(self, img_array: np.ndarray, k_values: List[int],
                              channel_name: str = "gray") -> List[Dict]:
        """Compress single channel (grayscale) image"""
        U, S, Vt = np.linalg.svd(img_array, full_matrices=False)
        m, n = img_array.shape
        original_size = m * n

        metrics = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        iterator = tqdm(k_values, desc=f"Compressing {channel_name}") if TQDM_AVAILABLE else k_values

        for k in iterator:
            reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            compressed_size = k * (m + n + 1)
            compression_ratio = compressed_size / original_size

            mse = np.mean((img_array - reconstructed) ** 2)
            psnr = self.calculate_psnr(img_array, reconstructed)
            ssim = self.calculate_ssim_simple(img_array, reconstructed)

            metrics.append({
                "k": k,
                "size_kb": compressed_size / 1024,
                "ratio": compression_ratio,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "reconstructed": reconstructed
            })

        return metrics

    def compress_color_image(self, img_array: np.ndarray, k_values: List[int]) -> List[Dict]:
        """Compress color image by processing each channel separately"""
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
        U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
        U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)

        m, n = R.shape
        original_color_size = m * n * 3

        metrics = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        iterator = tqdm(k_values, desc="Compressing color") if TQDM_AVAILABLE else k_values

        for k in iterator:
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

            metrics.append({
                "k": k,
                "size_kb": compressed_color_size / 1024,
                "ratio": compression_ratio,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "reconstructed": recon_img
            })

        return metrics

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
        plt.figure(figsize=(18, 12))

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
                     f"Ratio: {metrics['ratio']:.1%}", fontsize=9)
            plt.axis("off")

            # Save individual images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_compressed_images(metrics['reconstructed'], metrics['k'], "gray", timestamp)

        plt.suptitle("Grayscale SVD Compression", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(f"{save_path}_gray.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Color comparison (if available)
        if original_color is not None and color_metrics is not None:
            plt.figure(figsize=(18, 12))

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
                         f"Ratio: {metrics['ratio']:.1%}", fontsize=9)
                plt.axis("off")

                # Save individual images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_compressed_images(metrics['reconstructed'], metrics['k'], "color", timestamp)

            plt.suptitle("Color SVD Compression", fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_path and self.config.save_plots:
                plt.savefig(f"{save_path}_color.png", dpi=300, bbox_inches='tight')
            plt.show()

    def create_metrics_analysis_plot(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None,
                                   save_path: str = None):
        """Create detailed metrics analysis plots"""
        k_values = [m['k'] for m in gray_metrics]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # PSNR comparison
        axes[0,0].plot(k_values, [m['psnr'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[0,0].plot(k_values, [m['psnr'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[0,0].set_xlabel('K values')
        axes[0,0].set_ylabel('PSNR (dB)')
        axes[0,0].set_title('PSNR vs K', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # SSIM comparison
        axes[0,1].plot(k_values, [m['ssim'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[0,1].plot(k_values, [m['ssim'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[0,1].set_xlabel('K values')
        axes[0,1].set_ylabel('SSIM')
        axes[0,1].set_title('SSIM vs K', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Compression ratio
        axes[1,0].plot(k_values, [m['ratio'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[1,0].plot(k_values, [m['ratio'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[1,0].set_xlabel('K values')
        axes[1,0].set_ylabel('Compression Ratio')
        axes[1,0].set_title('Compression Ratio vs K', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # File size
        axes[1,1].plot(k_values, [m['size_kb'] for m in gray_metrics], 'b-o', label='Grayscale', linewidth=2)
        if color_metrics:
            axes[1,1].plot(k_values, [m['size_kb'] for m in color_metrics], 'r-s', label='Color', linewidth=2)
        axes[1,1].set_xlabel('K values')
        axes[1,1].set_ylabel('File Size (KB)')
        axes[1,1].set_title('File Size vs K', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(f"{save_path}_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()

    def print_metrics_table(self, gray_metrics: List[Dict], color_metrics: List[Dict] = None):
        """Print formatted metrics comparison table"""
        print("\n" + "=" * 100)
        print("METRICS COMPARISON TABLE")
        print("=" * 100)

        if color_metrics:
            print(f"{'k':<5} {'Gray-Size(KB)':<13} {'Gray-PSNR':<10} {'Gray-SSIM':<10} "
                  f"{'Color-Size(KB)':<14} {'Color-PSNR':<11} {'Color-SSIM':<11} {'Ratio':<8}")
            print("-" * 100)

            for i in range(len(gray_metrics)):
                g = gray_metrics[i]
                c = color_metrics[i] if i < len(color_metrics) else {}
                print(f"{g['k']:<5} {g['size_kb']:<13.1f} {g['psnr']:<10.1f} {g['ssim']:<10.3f} "
                      f"{c.get('size_kb', 0):<14.1f} {c.get('psnr', 0):<11.1f} "
                      f"{c.get('ssim', 0):<11.3f} {c.get('ratio', 0):<8.1%}")
        else:
            print(f"{'k':<5} {'Size(KB)':<10} {'PSNR':<10} {'SSIM':<10} {'MSE':<10} {'Ratio':<8}")
            print("-" * 60)

            for g in gray_metrics:
                print(f"{g['k']:<5} {g['size_kb']:<10.1f} {g['psnr']:<10.1f} "
                      f"{g['ssim']:<10.3f} {g['mse']:<10.1f} {g['ratio']:<8.1%}")


    def save_results_json(
        self,
        gray_metrics: List[Dict],
        color_metrics: List[Dict] = None,
        image_info: Dict = None,
        filename: str = None,
    ):
        """Save compression results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/compression_results_{timestamp}.json"

        # Clean metrics for JSON serialization
        def clean_metrics(metrics):
            cleaned = []
            for m in metrics:
                clean_m = {k: v for k, v in m.items() if k != "reconstructed"}
                # Convert numpy types to native Python types
                for key, value in clean_m.items():
                    if isinstance(value, np.ndarray):
                        clean_m[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        clean_m[key] = value.item()
                    elif isinstance(value, np.bool_):
                        clean_m[key] = bool(value)
                cleaned.append(clean_m)
            return cleaned

        # Clean image_info dictionary
        def clean_image_info(info):
            if info is None:
                return {}
            cleaned_info = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    cleaned_info[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned_info[key] = value.item()
                elif isinstance(value, np.bool_):
                    cleaned_info[key] = bool(value)
                elif isinstance(value, tuple):
                    # Convert tuple to list for JSON serialization
                    cleaned_info[key] = list(value)
                else:
                    cleaned_info[key] = value
            return cleaned_info

        results = {
            "timestamp": datetime.now().isoformat(),
            "image_info": clean_image_info(image_info),
            "config": {
                "energy_threshold": float(self.config.energy_threshold),
                "max_image_size": list(self.config.max_image_size),
                "output_format": str(self.config.output_format),
            },
            "gray_metrics": clean_metrics(gray_metrics),
            "color_metrics": clean_metrics(color_metrics) if color_metrics else None,
        }

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {filename}")
        except Exception as e:
            print(f"Warning: Could not save JSON results: {e}")
            # Save a simplified version without problematic data
            try:
                simplified_results = {
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "k_values": [m["k"] for m in gray_metrics],
                        "gray_psnr": [float(m["psnr"]) for m in gray_metrics],
                        "gray_ssim": [float(m["ssim"]) for m in gray_metrics],
                    },
                }
                if color_metrics:
                    simplified_results["summary"]["color_psnr"] = [
                        float(m["psnr"]) for m in color_metrics
                    ]
                    simplified_results["summary"]["color_ssim"] = [
                        float(m["ssim"]) for m in color_metrics
                    ]

                backup_filename = filename.replace(".json", "_simplified.json")
                with open(backup_filename, "w") as f:
                    json.dump(simplified_results, f, indent=2)
                print(f"Simplified results saved to: {backup_filename}")
            except Exception as e2:
                print(f"Error: Could not save even simplified results: {e2}")

    def compress_image(self, source: str, source_type: str = "url",
                      k_values: List[int] = None, save_results: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Main compression function"""
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

        # SVD analysis
        U, S, Vt = np.linalg.svd(gray_img, full_matrices=False)
        optimal_k = self.find_optimal_k(S)

        print(f"Optimal k (95% energy): {optimal_k}")
        print(f"Processing with k values: {k_values}")

        # Create singular values plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{self.output_dir}/svd_analysis_{timestamp}"
        self.create_singular_values_plot(S, optimal_k, f"{plot_path}_singular_values.png")

        # Compress grayscale
        gray_metrics = self.compress_single_channel(gray_img, k_values, "grayscale")

        # Compress color (if available)
        color_metrics = None
        if img_array is not None:
            color_metrics = self.compress_color_image(img_array, k_values)

        # Create comparison plots
        comparison_path = f"{self.output_dir}/comparison_{timestamp}"
        self.create_comparison_plot(gray_img, gray_metrics, img_array, color_metrics, comparison_path)

        # Create metrics analysis
        self.create_metrics_analysis_plot(gray_metrics, color_metrics,
                                        f"{self.output_dir}/metrics_analysis_{timestamp}")

        # Print results table
        self.print_metrics_table(gray_metrics, color_metrics)

        # Save results to JSON
        if save_results:
            image_info = {
                "source": source,
                "source_type": source_type,
                "shape": list(gray_img.shape) if img_array is None else list(img_array.shape),
                "optimal_k": int(optimal_k),
                "processing_time": float(time.time() - start_time)
            }
            self.save_results_json(gray_metrics, color_metrics, image_info)

        # Store in history
        self.compression_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "k_values": k_values,
            "optimal_k": optimal_k,
            "gray_metrics": len(gray_metrics),
            "color_metrics": len(color_metrics) if color_metrics else 0
        })

        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")

        return gray_metrics, color_metrics


def get_user_k_values() -> List[int]:
    """Get k values from user input"""
    print("Choose k values for compression:")
    print("1. Use default values " + str(DEFAULT_k_VALUES))
    print("2. Enter custom values")
    print("3. Quick test: " + str(DEFAULT_k_VALUES_SIMPLE))

    choice = input("Choose (1, 2, or 3): ").strip()

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
    else:
        print("Invalid choice. Using default values.")
        return DEFAULT_k_VALUES

def get_image_source():
    """Get image source from user"""
    print("\nChoose image source:")
    print("1. From URL")
    print("2. From local file")
    print("3. Use default image")

    choice = input("Choose (1, 2, or 3): ").strip()

    if choice == "1":
        url = input("Enter image URL: ").strip()
        return "url", url
    elif choice == "2":
        path = input("Enter image file path: ").strip()
        return "local", path
    else:
        return (
            "url",
            DEFAULT_IMAGE_URL
        )

def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(description='SVD Image Compression Tool')
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

    return parser.parse_args()

def main():
    """Main function"""
    print("=== SVD IMAGE COMPRESSION TOOL ===")
    print("Advanced image compression using Singular Value Decomposition")
    print()

    # Check if running from command line
    import sys
    if len(sys.argv) > 1:
        args = create_cli()

        # Create configuration
        config = CompressionConfig(
            default_k_values=args.k_values,
            energy_threshold=args.energy_threshold,
            max_image_size=tuple(args.max_size),
            output_format=args.format,
            save_plots=not args.no_plots
        )

        # Create compressor
        compressor = SVDImageCompressor(args.output, config)

        # Determine source type
        if args.input:
            source_type = "url" if args.input.startswith(('http://', 'https://')) else "local"
            result = compressor.compress_image(args.input, source_type, args.k_values)
        else:
            print("Error: No input specified")
            return

    else:
        # Interactive mode
        k_values = get_user_k_values()
        source_type, source_path = get_image_source()

        # Create configuration
        config = CompressionConfig(default_k_values=k_values)

        # Create compressor
        compressor = SVDImageCompressor("output", config)

        print(f"\nStarting compression with k values: {k_values}")

        # Process image
        result = compressor.compress_image(source_path, source_type, k_values)

        if result[0] is not None:
            print(f"\nCompression completed successfully!")
            print(f"Check the 'output' folder for results")
        else:
            print("Compression failed!")

if __name__ == "__main__":
    main()
