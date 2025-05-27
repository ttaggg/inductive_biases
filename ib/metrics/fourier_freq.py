"""Fourier Frequency metric."""

import numpy as np

from ib.utils.logging_module import logging


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

    total_amp = fft_amplitude.sum()
    if total_amp == 0:
        logging.info("Constant function, returning 0.0")
        return 0.0

    p = fft_amplitude / total_amp  # p_k = |F_k| / Σ_q|F_q|
    complexity = np.sum(p * k_magnitude)  # Σ_k ‖k‖₂ · p_k
    return float(complexity)


class FourierFrequency:
    """Fourier Frequency complexity metric for a given 3D SDF volume.

    C_fourier(f) = (sum_{k != 0} |F(k)| * |k|) / (sum_{k != 0} |F(k)|)
    """

    def __call__(self, predicted_sdf: np.ndarray) -> dict[str, float]:
        predicted_metric = _calculate_fourier_frequency(predicted_sdf)
        return {"metrics/predicted_fourier": predicted_metric}
