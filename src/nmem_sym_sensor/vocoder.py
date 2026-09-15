"""
Griffin-Lim vocoder: convert mel spectrograms to waveforms.

This is the "vocal cord" — takes a mel spectrogram (what the decoder
produces) and converts it to an audible waveform. Uses the iterative
Griffin-Lim phase reconstruction algorithm.

The quality won't be studio-grade, but it doesn't need to be.
A baby's first words are barely intelligible. What matters is that
the self-monitoring loop can re-encode the output and compare it
to the target sound unit embedding. The waveform is an intermediate
step, not the final product.

No external dependencies beyond numpy and scipy.
"""
import numpy as np

try:
    from scipy.signal import istft as scipy_istft
except ImportError:
    scipy_istft = None


def mel_filterbank_inverse(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Build a pseudo-inverse mel filterbank for mel → linear spectrogram.

    The forward mel filterbank maps linear frequency bins to mel bins.
    The pseudo-inverse approximately reverses this (lossy — mel is a
    many-to-one mapping, so the inverse is underdetermined).

    Returns: (n_fft // 2 + 1, n_mels) matrix
    """
    # Build forward mel filterbank (same as audio.py's _get_mel_filterbank)
    fmin, fmax = 0.0, sample_rate / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Triangular filterbank: (n_mels, n_freqs)
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        for j, f in enumerate(freqs):
            if low <= f <= center:
                filterbank[i, j] = (f - low) / max(center - low, 1e-8)
            elif center < f <= high:
                filterbank[i, j] = (high - f) / max(high - center, 1e-8)

    # Pseudo-inverse: (n_freqs, n_mels)
    # Use pinv for numerical stability
    return np.linalg.pinv(filterbank)


def mel_to_linear(mel_spectrogram: np.ndarray, n_fft: int, sample_rate: int) -> np.ndarray:
    """Convert mel spectrogram to approximate linear spectrogram.

    Args:
        mel_spectrogram: (n_mels, T) log-mel spectrogram
        n_fft: FFT size (determines number of frequency bins)
        sample_rate: Audio sample rate in Hz

    Returns:
        (n_fft // 2 + 1, T) linear magnitude spectrogram
    """
    n_mels = mel_spectrogram.shape[0]

    # Undo log scaling: log(mel + 1e-8) → mel
    mel_power = np.exp(mel_spectrogram)

    # Invert mel filterbank
    inv_filterbank = mel_filterbank_inverse(n_mels, n_fft, sample_rate)
    linear = inv_filterbank @ mel_power

    # Clip negative values (artefacts of pseudo-inverse)
    linear = np.maximum(linear, 0.0)

    return linear


def griffin_lim(
    mel_spectrogram: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 256,
    sample_rate: int = 16000,
    n_iter: int = 32,
) -> np.ndarray:
    """Convert mel spectrogram to waveform using Griffin-Lim.

    Iterative phase reconstruction algorithm:
    1. Start with random phase
    2. iSTFT → waveform
    3. STFT → replace magnitude with target, keep estimated phase
    4. Repeat until convergence

    Args:
        mel_spectrogram: (n_mels, T) log-mel spectrogram from decoder
        n_fft: FFT size
        hop_length: STFT hop length in samples
        sample_rate: Audio sample rate in Hz
        n_iter: Number of Griffin-Lim iterations (more = better quality)

    Returns:
        1D float32 waveform array at sample_rate Hz
    """
    # Convert mel to linear magnitude spectrogram
    magnitude = mel_to_linear(mel_spectrogram, n_fft, sample_rate)

    n_freqs, n_frames = magnitude.shape

    # Random initial phase
    phase = np.exp(2j * np.pi * np.random.rand(n_freqs, n_frames))

    # Iterative reconstruction
    window = np.hanning(n_fft)

    for i in range(n_iter):
        # Complex spectrogram = magnitude × phase
        stft_complex = magnitude * phase

        # iSTFT → waveform
        waveform = _istft(stft_complex, hop_length, window)

        # STFT → new complex spectrogram
        new_stft = _stft(waveform, n_fft, hop_length, window)

        # Keep new phase, replace magnitude with target
        new_magnitude = np.abs(new_stft)
        new_magnitude[new_magnitude < 1e-8] = 1e-8
        phase = new_stft / new_magnitude

    # Final reconstruction
    final_stft = magnitude * phase
    waveform = _istft(final_stft, hop_length, window)

    # Normalise to [-1, 1]
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.9

    return waveform.astype(np.float32)


def _stft(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Short-time Fourier Transform.

    Returns complex spectrogram (n_fft // 2 + 1, n_frames).
    """
    # Pad signal
    pad_length = n_fft // 2
    padded = np.pad(waveform, (pad_length, pad_length), mode="reflect")

    n_frames = 1 + (len(padded) - n_fft) // hop_length
    n_freqs = n_fft // 2 + 1

    stft = np.zeros((n_freqs, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = spectrum

    return stft


def _istft(
    stft_complex: np.ndarray,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Inverse Short-time Fourier Transform.

    Uses overlap-add reconstruction.
    """
    n_freqs, n_frames = stft_complex.shape
    n_fft = (n_freqs - 1) * 2

    # Output length
    output_length = n_fft + (n_frames - 1) * hop_length
    waveform = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i in range(n_frames):
        frame = np.fft.irfft(stft_complex[:, i])
        start = i * hop_length
        waveform[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2

    # Normalise by window overlap
    window_sum[window_sum < 1e-8] = 1e-8
    waveform /= window_sum

    # Remove padding
    pad = n_fft // 2
    return waveform[pad:-pad] if pad > 0 else waveform


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = 16000):
    """Save waveform to WAV file."""
    try:
        import soundfile as sf
        sf.write(path, waveform, sample_rate)
    except ImportError:
        # Fallback to scipy
        from scipy.io import wavfile
        # Convert float32 [-1,1] to int16
        int_waveform = (waveform * 32767).astype(np.int16)
        wavfile.write(path, sample_rate, int_waveform)
