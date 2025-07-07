import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# pip install numpy requests Pillow matplotlib
# python SVDCompresser.py

def compress_and_show_images(image_url: str):
    """
    Tải ảnh từ URL,
    => thực hiện nén SVD trên cả ảnh xám và ảnh màu,
    => hiển thị kết quả so sánh.
    """
    try:
        # --- Tải ảnh từ URL ---
        response = requests.get(image_url)
        response.raise_for_status()  # Báo lỗi nếu không tải được
        img = Image.open(BytesIO(response.content))

        # Chuyển ảnh sang dạng mảng NumPy để tính toán
        # Chuyển sang float64 để tính toán chính xác hơn
        original_img_array = np.array(img, dtype=np.float64)

        print(f"Kích thước ảnh gốc: {original_img_array.shape}")

    except requests.exceptions.RequestException as e:
        print(f"Lỗi tải ảnh: {e}")
        return

    # --- Phần 1: Nén ảnh thang độ xám ---

    # Chuyển ảnh màu sang ảnh xám
    gray_img_array = np.mean(original_img_array, axis=2)

    # Thực hiện SVD
    U, S, Vt = np.linalg.svd(gray_img_array, full_matrices=False)

    # Trực quan hóa các giá trị suy biến
    plt.figure(figsize=(10, 5))
    plt.plot(S)
    plt.title('Phân bố các giá trị suy biến (Singular Values)')
    plt.ylabel('Giá trị')
    plt.xlabel('Thứ tự giá trị suy biến')
    plt.grid(True)
    plt.show()

    # Tái tạo và hiển thị ảnh với các giá trị k khác nhau
    k_values = [5, 20, 50, 100]

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(gray_img_array, cmap='gray')
    plt.title('Ảnh xám gốc')
    plt.axis('off')

    # Kích thước dữ liệu gốc (m*n)
    m, n = gray_img_array.shape
    original_size = m * n

    for i, k in enumerate(k_values):
        # Tái tạo ảnh với k thành phần
        reconstructed_array = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

        # Kích thước dữ liệu sau khi nén: k*(m+n+1)
        compressed_size = k * (m + n + 1)
        compression_ratio = compressed_size / original_size

        plt.subplot(2, 3, i + 2)
        plt.imshow(reconstructed_array, cmap='gray')
        plt.title(f'k = {k}\nTỉ lệ nén: {compression_ratio:.2%}')
        plt.axis('off')

    plt.suptitle('Nén ảnh thang độ xám bằng SVD', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4)
    plt.show()

    # --- Phần 2: Nén ảnh màu ---

    # Tách 3 kênh màu R, G, B
    R, G, B = original_img_array[:,:,0], original_img_array[:,:,1], original_img_array[:,:,2]

    # Áp dụng SVD cho từng kênh
    U_r, S_r, Vt_r = np.linalg.svd(R, full_matrices=False)
    U_g, S_g, Vt_g = np.linalg.svd(G, full_matrices=False)
    U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)

    # Tái tạo và hiển thị ảnh màu
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    # Cần chuyển đổi lại sang uint8 để hiển thị đúng màu
    plt.imshow(original_img_array.astype(np.uint8))
    plt.title('Ảnh màu gốc')
    plt.axis('off')

    original_color_size = original_size * 3

    for i, k in enumerate(k_values):
        # Tái tạo từng kênh màu
        R_recon = U_r[:, :k] @ np.diag(S_r[:k]) @ Vt_r[:k, :]
        G_recon = U_g[:, :k] @ np.diag(S_g[:k]) @ Vt_g[:k, :]
        B_recon = U_b[:, :k] @ np.diag(S_b[:k]) @ Vt_b[:k, :]

        # Gộp các kênh lại
        recon_img_array = np.stack([R_recon, G_recon, B_recon], axis=2)

        # Xử lý giá trị vượt ngưỡng [0, 255]
        recon_img_array = np.clip(recon_img_array, 0, 255)

        compressed_color_size = k * (m + n + 1) * 3
        compression_ratio_color = compressed_color_size / original_color_size

        plt.subplot(2, 3, i + 2)
        plt.imshow(recon_img_array.astype(np.uint8))
        plt.title(f'k = {k}\nTỉ lệ nén: {compression_ratio_color:.2%}')
        plt.axis('off')

    plt.suptitle('Nén ảnh màu bằng SVD', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4)
    plt.show()


IMAGE_URL = "https://images.unsplash.com/photo-1542838132-92c53300491e?w=800"

compress_and_show_images(IMAGE_URL)
