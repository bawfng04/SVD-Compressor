# SVD-Compresser

A Python script to demonstrate image compression using Singular Value Decomposition (SVD). This script downloads an image, applies SVD compression for both grayscale and color versions, and visualizes the results.

## How it works

Singular Value Decomposition is a matrix factorization technique that can be used to approximate a matrix with a lower-rank matrix. By keeping only the most significant singular values and their corresponding vectors, we can reconstruct an image that is visually similar to the original but requires less data to store.

The script performs the following steps:
1.  Downloads an image from a specified URL.
2.  **Grayscale Compression**:
    *   Converts the image to grayscale.
    *   Performs SVD on the grayscale image matrix.
    *   Reconstructs the image using a varying number of singular values (`k`).
    *   Displays the original and compressed images side-by-side, showing the compression ratio for each `k`.
3.  **Color Compression**:
    *   Splits the image into its Red, Green, and Blue channels.
    *   Applies SVD to each channel independently.
    *   Reconstructs the color image from the compressed channels.
    *   Displays the original and compressed color images with their respective compression ratios.

## Requirements

- Python 3.x
- The following Python libraries:
  - `numpy`
  - `requests`
  - `Pillow`
  - `matplotlib`

You can install them using pip:
```bash
pip install numpy requests Pillow matplotlib
```

## Usage

1.  Clone this repository or download the `SVDCompresser.py` script.
2.  (Optional) Open `SVDCompresser.py` and change the `IMAGE_URL` variable to point to a different image.
3.  Run the script from your terminal:
```bash
python SVDCompresser.py
```
The script will generate and display plots showing the compressed images.