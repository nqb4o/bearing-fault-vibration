"""
Vibration Feature Extraction Module
Extracts 17 features from a single time-domain vibration window for SHAP analysis.
"""
import numpy as np
import math
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from scipy.signal import hilbert

# ─── Feature Names (ordered, used everywhere) ────────────────────────────────
FEATURE_NAMES = [
    # Time-Domain (8)
    "RMS",
    "Kurtosis",
    "Peak Value",
    "Crest Factor",
    "Skewness",
    "Shape Factor",
    "Impulse Factor",
    "Std Dev",
    # Frequency-Domain (3)
    "Dominant Frequency",
    "Spectral Centroid",
    "Spectral Energy",
    # Complexity (3)
    "Shannon Entropy",
    "Sample Entropy",
    "Zero-Crossing Rate",
    # Time-Frequency (2)
    "Wavelet Energy",
    "Envelope Energy",
]

FEATURE_CATEGORIES = {
    "RMS": "Time-Domain", "Kurtosis": "Time-Domain", "Peak Value": "Time-Domain",
    "Crest Factor": "Time-Domain", "Skewness": "Time-Domain", "Shape Factor": "Time-Domain",
    "Impulse Factor": "Time-Domain", "Std Dev": "Time-Domain",
    "Dominant Frequency": "Frequency-Domain", "Spectral Centroid": "Frequency-Domain",
    "Spectral Energy": "Frequency-Domain",
    "Shannon Entropy": "Complexity", "Sample Entropy": "Complexity",
    "Zero-Crossing Rate": "Complexity",
    "Wavelet Energy": "Time-Frequency", "Envelope Energy": "Time-Frequency",
}


# ─── Individual Feature Functions ─────────────────────────────────────────────

def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))

def _kurtosis(x):
    return float(kurtosis(x, fisher=True))

def _peak_value(x):
    return float(np.max(np.abs(x)))

def _crest_factor(x):
    rms = _rms(x)
    if rms == 0:
        return 0.0
    return float(np.max(np.abs(x)) / rms)

def _skewness(x):
    return float(skew(x))

def _shape_factor(x):
    mean_abs = np.mean(np.abs(x))
    if mean_abs == 0:
        return 0.0
    return float(_rms(x) / mean_abs)

def _impulse_factor(x):
    mean_abs = np.mean(np.abs(x))
    if mean_abs == 0:
        return 0.0
    return float(np.max(np.abs(x)) / mean_abs)

def _std_dev(x):
    return float(np.std(x))


def _dominant_frequency(x):
    """Normalized index of the strongest FFT component."""
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    return float(freqs[np.argmax(fft_mag[1:]) + 1])  # skip DC

def _spectral_centroid(x):
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    total = np.sum(fft_mag)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * fft_mag) / total)

def _spectral_energy(x):
    fft_mag = np.abs(np.fft.rfft(x))
    return float(np.sum(fft_mag ** 2))


def _shannon_entropy(x, n_bins=50):
    """Shannon entropy of the signal amplitude histogram."""
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return float(scipy_entropy(hist, base=2))

def _sample_entropy(x, m=3, delay=1):
    """Fast permutation entropy (O(N)) — captures signal complexity/regularity.
    Uses ordinal patterns instead of the O(N²) pairwise-distance approach."""
    N = len(x)
    if N < m * delay + 1:
        return 0.0
    
    # Build ordinal patterns
    n_patterns = N - (m - 1) * delay
    # Create indices for each pattern
    indices = np.arange(m) * delay
    patterns = np.array([x[i + indices] for i in range(n_patterns)])
    # Convert to ordinal ranks
    ordinals = np.argsort(np.argsort(patterns, axis=1), axis=1)
    # Hash each pattern to a single integer
    multipliers = m ** np.arange(m)
    pattern_ids = ordinals @ multipliers
    # Count unique pattern frequencies
    _, counts = np.unique(pattern_ids, return_counts=True)
    probs = counts / counts.sum()
    # Normalized permutation entropy (0 = perfectly regular, 1 = maximally complex)
    max_entropy = np.log2(float(np.prod(np.arange(1, m + 1))))  # log2(m!)
    if max_entropy == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)) / max_entropy)

def _zero_crossing_rate(x):
    return float(np.sum(np.abs(np.diff(np.sign(x))) > 0) / len(x))


def _wavelet_energy(x):
    """Energy from DWT detail coefficients. Falls back to FFT band energy if pywt unavailable."""
    try:
        import pywt
        coeffs = pywt.wavedec(x, 'db4', level=4)
        # Sum energy of detail coefficients (skip approximation)
        detail_energy = sum(float(np.sum(c ** 2)) for c in coeffs[1:])
        return detail_energy
    except ImportError:
        # Fallback: high-frequency energy from FFT
        fft_mag = np.abs(np.fft.rfft(x))
        half = len(fft_mag) // 2
        return float(np.sum(fft_mag[half:] ** 2))

def _envelope_energy(x):
    """Energy of the analytic signal envelope (Hilbert transform)."""
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    return float(np.sum(envelope ** 2) / len(x))


# ─── Main Extraction Function ────────────────────────────────────────────────

def extract_features(window):
    """
    Extract all 17 features from a 1D vibration window.
    
    Args:
        window: 1D numpy array of vibration samples
        
    Returns:
        numpy array of shape (17,) with feature values in FEATURE_NAMES order
    """
    x = window.flatten().astype(np.float64)
    
    features = np.array([
        # Time-Domain
        _rms(x),
        _kurtosis(x),
        _peak_value(x),
        _crest_factor(x),
        _skewness(x),
        _shape_factor(x),
        _impulse_factor(x),
        _std_dev(x),
        # Frequency-Domain
        _dominant_frequency(x),
        _spectral_centroid(x),
        _spectral_energy(x),
        # Complexity
        _shannon_entropy(x),
        _sample_entropy(x),
        _zero_crossing_rate(x),
        # Time-Frequency
        _wavelet_energy(x),
        _envelope_energy(x),
    ], dtype=np.float64)
    
    return features


def extract_features_batch(windows):
    """
    Extract features from multiple windows.
    
    Args:
        windows: numpy array of shape (N, window_size) or (N, window_size, 1)
        
    Returns:
        numpy array of shape (N, 17)
    """
    if windows.ndim == 3:
        windows = windows.squeeze(axis=-1)
    
    return np.array([extract_features(w) for w in windows])
