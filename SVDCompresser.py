import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict

# pip install numpy requests pillow matplotlib
# python SVDCompresser.py


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim_simple(original: np.ndarray, compressed: np.ndarray) -> float:
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


def save_compressed_images(
    recon_array: np.ndarray, k: int, image_type: str, output_dir: str = "output"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if image_type == "gray":
        img = Image.fromarray(recon_array.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(recon_array.astype(np.uint8), mode="RGB")

    filename = f"{output_dir}/compressed_{image_type}_k{k}.png"
    img.save(filename)
    return filename


def find_optimal_k(S: np.ndarray, energy_threshold: float = 0.95) -> int:
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    optimal_k = np.argmax(cumulative_energy >= energy_threshold) + 1
    return optimal_k


def print_metrics_table(gray_metrics: List[Dict], color_metrics: List[Dict]):
    print("\n" + "=" * 90)
    print("BẢNG SO SÁNH METRICS")
    print("=" * 90)

    print(
        f"{'k':<5} {'Xám-Size(KB)':<13} {'Xám-PSNR':<10} {'Xám-SSIM':<10} {'Màu-Size(KB)':<13} {'Màu-PSNR':<10} {'Màu-SSIM':<10} {'Ratio':<8}"
    )
    print("-" * 90)

    for i in range(len(gray_metrics)):
        g = gray_metrics[i]
        c = color_metrics[i]
        print(
            f"{g['k']:<5} {g['size_kb']:<13.1f} {g['psnr']:<10.1f} {g['ssim']:<10.3f} {c['size_kb']:<13.1f} {c['psnr']:<10.1f} {c['ssim']:<10.3f} {c['ratio']:<8.1%}"
        )


def get_user_k_values() -> List[int]:
    """
    Cho phép người dùng nhập các giá trị k tùy chỉnh
    """
    print("Nhập các giá trị k để nén ảnh:")
    print("1. Sử dụng giá trị mặc định [5, 25, 50, 75, 100]")
    print("2. Nhập giá trị tùy chỉnh")

    choice = input("Chọn (1 hoặc 2): ").strip()

    if choice == "1":
        return [5, 25, 50, 75, 100]
    elif choice == "2":
        while True:
            try:
                k_input = input(
                    "Nhập các giá trị k cách nhau bởi dấu phẩy (VD: 10,30,50,100): "
                )
                k_values = [int(k.strip()) for k in k_input.split(",")]

                # Kiểm tra giá trị hợp lệ
                if any(k <= 0 for k in k_values):
                    print("Lỗi: Tất cả giá trị k phải lớn hơn 0")
                    continue

                # Sắp xếp các giá trị k
                k_values.sort()
                print(f"Các giá trị k đã chọn: {k_values}")
                return k_values

            except ValueError:
                print("Lỗi: Vui lòng nhập các số nguyên hợp lệ")
    else:
        print("Lựa chọn không hợp lệ. Sử dụng giá trị mặc định.")
        return [5, 25, 50, 75, 100]


def get_image_source():
    """
    Cho phép người dùng chọn nguồn ảnh
    """
    print("\nChọn nguồn ảnh:")
    print("1. Từ URL")
    print("2. Từ file local")
    print("3. Sử dụng ảnh mặc định")

    choice = input("Chọn (1, 2 hoặc 3): ").strip()

    if choice == "1":
        url = input("Nhập URL ảnh: ").strip()
        return "url", url
    elif choice == "2":
        path = input("Nhập đường dẫn file ảnh: ").strip()
        return "local", path
    else:
        return (
            "default",
            "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg?cs=srgb&dl=pexels-christian-heitz-285904-842711.jpg&fm=jpg",
        )


def compress_and_show_images(
    image_url: str, k_values: List[int] = None, save_results: bool = True
):
    if k_values is None:
        k_values = [5, 25, 50, 75, 100]

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        original_img_array = np.array(img, dtype=np.float64)
        print(f"Kích thước ảnh gốc: {original_img_array.shape}")

    except requests.exceptions.RequestException as e:
        print(f"Lỗi tải ảnh: {e}")
        return

    gray_img_array = np.mean(original_img_array, axis=2)
    U, S, Vt = np.linalg.svd(gray_img_array, full_matrices=False)

    optimal_k = find_optimal_k(S, 0.95)
    print(f"K tối ưu (95% năng lượng): {optimal_k}")

    # Điều chỉnh số lượng subplot dựa trên số k_values
    n_k = len(k_values)
    fig_cols = min(n_k + 1, 6)  # Tối đa 6 cột
    fig_rows = (n_k + 1 + fig_cols - 1) // fig_cols  # Tính số hàng cần thiết

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(S)
    plt.title("Phân bố các giá trị suy biến")
    plt.ylabel("Giá trị")
    plt.xlabel("Thứ tự")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    plt.plot(cumulative_energy)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% energy")
    plt.axvline(x=optimal_k, color="r", linestyle="--", label=f"k={optimal_k}")
    plt.title("Năng lượng tích lũy")
    plt.ylabel("Tỷ lệ năng lượng")
    plt.xlabel("Số thành phần k")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.loglog(S)
    plt.title("Singular Values (Log Scale)")
    plt.ylabel("Giá trị (log)")
    plt.xlabel("Thứ tự (log)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    gray_metrics = []
    m, n = gray_img_array.shape
    original_size = m * n

    plt.figure(figsize=(18, 12))
    plt.subplot(fig_rows, fig_cols, 1)
    plt.imshow(gray_img_array, cmap="gray")
    plt.title(f"Ảnh xám gốc\nSize: {original_size / 1024:.1f} KB")
    plt.axis("off")

    for i, k in enumerate(k_values):
        if k > min(m, n):
            print(f"Cảnh báo: k={k} vượt quá rank tối đa của ảnh ({min(m, n)})")
            k = min(m, n)

        reconstructed_array = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        compressed_size = k * (m + n + 1)
        compression_ratio = compressed_size / original_size

        mse = np.mean((gray_img_array - reconstructed_array) ** 2)
        psnr = calculate_psnr(gray_img_array, reconstructed_array)
        ssim = calculate_ssim_simple(gray_img_array, reconstructed_array)

        gray_metrics.append(
            {
                "k": k,
                "size_kb": compressed_size / 1024,
                "ratio": compression_ratio,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

        plt.subplot(fig_rows, fig_cols, i + 2)
        plt.imshow(reconstructed_array, cmap="gray")
        plt.title(
            f"k = {k}\nPSNR: {psnr:.1f}dB | SSIM: {ssim:.3f}\nRatio: {compression_ratio:.1%}"
        )
        plt.axis("off")

        if save_results:
            save_compressed_images(reconstructed_array, k, "gray")

    plt.suptitle("Nén ảnh xám bằng SVD", fontsize=16)
    plt.tight_layout()
    plt.show()

    R, G, B = (
        original_img_array[:, :, 0],
        original_img_array[:, :, 1],
        original_img_array[:, :, 2],
    )

    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)

    color_metrics = []
    original_color_size = original_size * 3

    plt.figure(figsize=(18, 12))
    plt.subplot(fig_rows, fig_cols, 1)
    plt.imshow(original_img_array.astype(np.uint8))
    plt.title(f"Ảnh màu gốc\nSize: {original_color_size / 1024:.1f} KB")
    plt.axis("off")

    for i, k in enumerate(k_values):
        if k > min(m, n):
            k = min(m, n)

        R_recon = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]
        G_recon = U_g[:, :k] @ np.diag(S_g[:k]) @ Vt_g[:k, :]
        B_recon = U_b[:, :k] @ np.diag(S_b[:k]) @ Vt_b[:k, :]

        recon_img_array = np.stack([R_recon, G_recon, B_recon], axis=2)
        recon_img_array = np.clip(recon_img_array, 0, 255)

        compressed_color_size = k * (m + n + 1) * 3
        compression_ratio_color = compressed_color_size / original_color_size

        mse_color = np.mean((original_img_array - recon_img_array) ** 2)
        psnr_color = calculate_psnr(original_img_array, recon_img_array)
        ssim_color = calculate_ssim_simple(original_img_array, recon_img_array)

        color_metrics.append(
            {
                "k": k,
                "size_kb": compressed_color_size / 1024,
                "ratio": compression_ratio_color,
                "mse": mse_color,
                "psnr": psnr_color,
                "ssim": ssim_color,
            }
        )

        plt.subplot(fig_rows, fig_cols, i + 2)
        plt.imshow(recon_img_array.astype(np.uint8))
        plt.title(
            f"k = {k}\nPSNR: {psnr_color:.1f}dB | SSIM: {ssim_color:.3f}\nRatio: {compression_ratio_color:.1%}"
        )
        plt.axis("off")

        if save_results:
            save_compressed_images(recon_img_array, k, "color")

    plt.suptitle("Nén ảnh màu bằng SVD", fontsize=16)
    plt.tight_layout()
    plt.show()

    print_metrics_table(gray_metrics, color_metrics)

    return gray_metrics, color_metrics


def compress_local_image(image_path: str, k_values: List[int] = None):
    if k_values is None:
        k_values = [5, 25, 50, 75, 100]

    try:
        img = Image.open(image_path)
        original_img_array = np.array(img, dtype=np.float64)
        print(f"Đã tải ảnh: {image_path}")
        print(f"Kích thước: {original_img_array.shape}")

        # Chuyển đổi thành grayscale nếu cần
        if len(original_img_array.shape) == 3:
            # Ảnh màu - thực hiện nén như ảnh từ URL
            return compress_from_array(original_img_array, k_values)
        else:
            # Ảnh xám
            print("Ảnh đã là ảnh xám")
            return compress_from_array(
                np.expand_dims(original_img_array, axis=2), k_values
            )

    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None


def compress_from_array(original_img_array: np.ndarray, k_values: List[int]):
    """
    Nén ảnh từ mảng numpy
    """
    print(f"Kích thước ảnh: {original_img_array.shape}")

    # Tiếp tục với logic nén tương tự như compress_and_show_images
    gray_img_array = (
        np.mean(original_img_array, axis=2)
        if len(original_img_array.shape) == 3
        else original_img_array
    )
    U, S, Vt = np.linalg.svd(gray_img_array, full_matrices=False)

    optimal_k = find_optimal_k(S, 0.95)
    print(f"K tối ưu (95% năng lượng): {optimal_k}")

    # Hiển thị singular values
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(S)
    plt.title("Phân bố các giá trị suy biến")
    plt.ylabel("Giá trị")
    plt.xlabel("Thứ tự")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    plt.plot(cumulative_energy)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% energy")
    plt.axvline(x=optimal_k, color="r", linestyle="--", label=f"k={optimal_k}")
    plt.title("Năng lượng tích lũy")
    plt.ylabel("Tỷ lệ năng lượng")
    plt.xlabel("Số thành phần k")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.loglog(S)
    plt.title("Singular Values (Log Scale)")
    plt.ylabel("Giá trị (log)")
    plt.xlabel("Thứ tự (log)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Thực hiện nén với các giá trị k
    gray_metrics = []
    m, n = gray_img_array.shape
    original_size = m * n

    # Điều chỉnh layout dựa trên số lượng k_values
    n_k = len(k_values)
    fig_cols = min(n_k + 1, 6)
    fig_rows = (n_k + 1 + fig_cols - 1) // fig_cols

    plt.figure(figsize=(18, 12))
    plt.subplot(fig_rows, fig_cols, 1)
    plt.imshow(gray_img_array, cmap="gray")
    plt.title(f"Ảnh xám gốc\nSize: {original_size / 1024:.1f} KB")
    plt.axis("off")

    for i, k in enumerate(k_values):
        if k > min(m, n):
            print(f"Cảnh báo: k={k} vượt quá rank tối đa của ảnh ({min(m, n)})")
            k = min(m, n)

        reconstructed_array = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        compressed_size = k * (m + n + 1)
        compression_ratio = compressed_size / original_size

        mse = np.mean((gray_img_array - reconstructed_array) ** 2)
        psnr = calculate_psnr(gray_img_array, reconstructed_array)
        ssim = calculate_ssim_simple(gray_img_array, reconstructed_array)

        gray_metrics.append(
            {
                "k": k,
                "size_kb": compressed_size / 1024,
                "ratio": compression_ratio,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

        plt.subplot(fig_rows, fig_cols, i + 2)
        plt.imshow(reconstructed_array, cmap="gray")
        plt.title(
            f"k = {k}\nPSNR: {psnr:.1f}dB | SSIM: {ssim:.3f}\nRatio: {compression_ratio:.1%}"
        )
        plt.axis("off")

        save_compressed_images(reconstructed_array, k, "gray")

    plt.suptitle("Nén ảnh xám bằng SVD", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Nén ảnh màu nếu có
    color_metrics = []
    if len(original_img_array.shape) == 3:
        R, G, B = (
            original_img_array[:, :, 0],
            original_img_array[:, :, 1],
            original_img_array[:, :, 2],
        )

        U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
        U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
        U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)

        original_color_size = original_size * 3

        plt.figure(figsize=(18, 12))
        plt.subplot(fig_rows, fig_cols, 1)
        plt.imshow(original_img_array.astype(np.uint8))
        plt.title(f"Ảnh màu gốc\nSize: {original_color_size / 1024:.1f} KB")
        plt.axis("off")

        for i, k in enumerate(k_values):
            if k > min(m, n):
                k = min(m, n)

            R_recon = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]
            G_recon = U_g[:, :k] @ np.diag(S_g[:k]) @ Vt_g[:k, :]
            B_recon = U_b[:, :k] @ np.diag(S_b[:k]) @ Vt_b[:k, :]

            recon_img_array = np.stack([R_recon, G_recon, B_recon], axis=2)
            recon_img_array = np.clip(recon_img_array, 0, 255)

            compressed_color_size = k * (m + n + 1) * 3
            compression_ratio_color = compressed_color_size / original_color_size

            mse_color = np.mean((original_img_array - recon_img_array) ** 2)
            psnr_color = calculate_psnr(original_img_array, recon_img_array)
            ssim_color = calculate_ssim_simple(original_img_array, recon_img_array)

            color_metrics.append(
                {
                    "k": k,
                    "size_kb": compressed_color_size / 1024,
                    "ratio": compression_ratio_color,
                    "mse": mse_color,
                    "psnr": psnr_color,
                    "ssim": ssim_color,
                }
            )

            plt.subplot(fig_rows, fig_cols, i + 2)
            plt.imshow(recon_img_array.astype(np.uint8))
            plt.title(
                f"k = {k}\nPSNR: {psnr_color:.1f}dB | SSIM: {ssim_color:.3f}\nRatio: {compression_ratio_color:.1%}"
            )
            plt.axis("off")

            save_compressed_images(recon_img_array, k, "color")

        plt.suptitle("Nén ảnh màu bằng SVD", fontsize=16)
        plt.tight_layout()
        plt.show()

    if color_metrics:
        print_metrics_table(gray_metrics, color_metrics)

    return gray_metrics, color_metrics


def batch_compress_images(image_urls: List[str], k_values: List[int] = None):
    if k_values is None:
        k_values = [25, 50, 100]

    all_results = []
    for i, url in enumerate(image_urls):
        print(f"\n--- Xử lý ảnh {i+1}/{len(image_urls)} ---")
        result = compress_and_show_images(url, k_values, save_results=True)
        if result:
            all_results.append(result)

    return all_results


if __name__ == "__main__":
    print("=== CHƯƠNG TRÌNH NÉN ẢNH BẰNG SVD ===")

    # Lấy giá trị k từ người dùng
    k_values = get_user_k_values()

    # Lấy nguồn ảnh từ người dùng
    source_type, source_path = get_image_source()

    print(f"\nBắt đầu nén ảnh với các giá trị k: {k_values}")

    if source_type == "url":
        gray_metrics, color_metrics = compress_and_show_images(
            source_path, k_values=k_values, save_results=True
        )
    elif source_type == "local":
        result = compress_local_image(source_path, k_values=k_values)
        if result:
            gray_metrics, color_metrics = result
    else:  # default
        gray_metrics, color_metrics = compress_and_show_images(
            source_path, k_values=k_values, save_results=True
        )

    print(f"\nĐã lưu ảnh nén vào thư mục 'output'")
