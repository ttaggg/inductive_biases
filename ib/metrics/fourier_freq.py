"""Fourier Frequency metric."""

from pathlib import Path

import numpy as np


def _calculate_fourier_frequency(sdf_volume: np.ndarray) -> float:

    # Compute the 3D discrete Fourier transform.
    fft = np.fft.fftn(sdf_volume)

    # Get the amplitude (absolute value) of the Fourier coefficients.
    fft_amplitude = np.abs(fft)

    # Assume the volume is cubic with resolution R.
    resolution = sdf_volume.shape[0]

    # Create a frequency grid using np.fft.fftfreq.
    freq = np.fft.fftfreq(resolution, d=1.0)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")

    # Compute the Euclidean magnitude of the frequency vector at each grid point.
    k_magnitude = np.sqrt(kx**2 + ky**2 + kz**2)

    # Exclude the zero frequency (DC) component.
    nonzero_mask = k_magnitude > 0

    # Compute the numerator and denominator of the metric.
    numerator = np.sum(fft_amplitude[nonzero_mask] * k_magnitude[nonzero_mask])
    denominator = np.sum(fft_amplitude[nonzero_mask]) + 1e-8  # Avoid division by zero.

    # The final metric is the weighted average frequency.
    return float(numerator / denominator)


class FourierFrequency:
    """Fourier Frequency complexity metric for a given 3D SDF volume.

    C_fourier(f) = (sum_{k != 0} |F(k)| * |k|) / (sum_{k != 0} |F(k)|)
    """

    def __call__(self, predicted_sdf: np.ndarray) -> dict[str, float]:
        predicted_metric = _calculate_fourier_frequency(predicted_sdf)

        return {
            "metrics/predicted_fourier": predicted_metric,
        }
