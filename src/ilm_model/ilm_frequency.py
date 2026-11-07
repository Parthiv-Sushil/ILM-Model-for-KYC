# -*- coding: utf-8 -*-
"""
Frequency & Texture Feature Branch.
Reuses the 2D FFT Fourier spectrum generator (generate_FT) to extract high-frequency
energy ratio, spectral roll-off, radial power spectrum, and texture anomaly descriptors.
"""

import numpy as np
import cv2
from src.data_io.dataset_folder import generate_FT


class FrequencyFeatureExtractor:
    """
    Extracts 16-dimensional frequency and texture domain feature vector from input face images.
    Spoof attacks (printed photos, screen replays) exhibit distinct periodic artifacts,
    high-frequency loss (blurring), or grid pattern spikes in Fourier space.
    """
    def __init__(self, target_size=(80, 80)):
        self.target_size = target_size

    def extract_features(self, img):
        """
        Parameters:
            img: BGR image (ndarray) of face crop
        Returns:
            freq_vector: (16,) numpy array
        """
        if img is None or img.size == 0:
            return np.zeros(16, dtype=np.float32)

        # Generate normalized 2D Fourier Spectrum (matches original repository method)
        ft_img = generate_FT(img)
        ft_resized = cv2.resize(ft_img, self.target_size)

        # 1. Energy Distribution in Frequency Bands
        h, w = ft_resized.shape
        cy, cx = h // 2, w // 2

        # Radial mask distance matrix
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max = np.sqrt(cx**2 + cy**2) + 1e-6
        r_norm = r / r_max

        # 4 concentric radial bands (Low, Mid-Low, Mid-High, High frequency)
        band1 = ft_resized[r_norm <= 0.25]
        band2 = ft_resized[(r_norm > 0.25) & (r_norm <= 0.50)]
        band3 = ft_resized[(r_norm > 0.50) & (r_norm <= 0.75)]
        band4 = ft_resized[r_norm > 0.75]

        b1_mean, b1_std = float(np.mean(band1)) if band1.size > 0 else 0.0, float(np.std(band1)) if band1.size > 0 else 0.0
        b2_mean, b2_std = float(np.mean(band2)) if band2.size > 0 else 0.0, float(np.std(band2)) if band2.size > 0 else 0.0
        b3_mean, b3_std = float(np.mean(band3)) if band3.size > 0 else 0.0, float(np.std(band3)) if band3.size > 0 else 0.0
        b4_mean, b4_std = float(np.mean(band4)) if band4.size > 0 else 0.0, float(np.std(band4)) if band4.size > 0 else 0.0

        # High Frequency Energy Ratio (HFER)
        total_energy = np.sum(ft_resized**2) + 1e-6
        high_freq_energy = np.sum(band4**2) + 1e-6
        hfer = float(high_freq_energy / total_energy)

        # 2. Laplacian Blur / Texture Sharpness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 3. Local Binary Pattern / Edge Gradient Magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mean = float(np.mean(grad_mag))
        grad_std = float(np.std(grad_mag))

        # 4. Spectral Entropy
        probs = ft_resized.flatten() / (np.sum(ft_resized) + 1e-6)
        probs = probs[probs > 0]
        spectral_entropy = float(-np.sum(probs * np.log2(probs)))

        freq_vector = np.array([
            b1_mean, b1_std,
            b2_mean, b2_std,
            b3_mean, b3_std,
            b4_mean, b4_std,
            hfer,
            laplacian_var,
            grad_mean,
            grad_std,
            spectral_entropy,
            float(np.max(ft_resized)),
            float(np.min(ft_resized)),
            float(np.median(ft_resized))
        ], dtype=np.float32)

        return freq_vector
