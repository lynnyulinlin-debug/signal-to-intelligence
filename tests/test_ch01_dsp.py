"""Tests for Chapter 1: DSP Basics"""
import numpy as np


class TestFFTSpectrum:
    """Test FFT spectrum analysis"""

    def test_fft_output_shape(self, seed, load_code_module):
        """Test that FFT output has correct shape"""
        fft_spectrum = load_code_module("code/ch01_dsp/fft_spectrum.py")
        signal_length = 1000

        result = fft_spectrum.run_experiment(signal_length=signal_length, seed=42)
        fft_result = result["fft_result"]

        assert fft_result.shape == (signal_length,)

    def test_fft_symmetry(self, seed, load_code_module):
        """Test FFT symmetry property for real signals"""
        fft_spectrum = load_code_module("code/ch01_dsp/fft_spectrum.py")
        signal = np.random.randn(100)
        _, _, fft_result = fft_spectrum.compute_fft_spectrum(signal, sampling_rate=100)

        # For real signals, FFT should be symmetric
        assert np.allclose(fft_result[1:50], np.conj(fft_result[-1:-50:-1]))

    def test_fft_frequency_detection(self, seed, load_code_module):
        """Test that FFT correctly detects signal frequencies"""
        fft_spectrum = load_code_module("code/ch01_dsp/fft_spectrum.py")
        sampling_rate = 100
        signal_length = 1000
        freq1, freq2 = 5, 10

        result = fft_spectrum.run_experiment(
            signal_length=signal_length,
            sampling_rate=sampling_rate,
            signal_freq=[freq1, freq2],
            noise_level=0.0,
            seed=42,
        )
        detected_freqs = result["detected_freqs"]

        # Check if detected frequencies are close to actual frequencies
        assert np.any(np.abs(detected_freqs - freq1) < 1)
        assert np.any(np.abs(detected_freqs - freq2) < 1)

    def test_fft_noise_robustness(self, seed, load_code_module):
        """Test FFT with noisy signal"""
        fft_spectrum = load_code_module("code/ch01_dsp/fft_spectrum.py")
        signal_length = 1000
        _, _, signal_noisy = fft_spectrum.generate_signal(
            signal_length=signal_length,
            signal_freq=[5],
            noise_level=0.5,
            seed=42,
        )

        _, _, fft_result = fft_spectrum.compute_fft_spectrum(signal_noisy)
        assert fft_result.shape == (signal_length,)
        assert not np.any(np.isnan(fft_result))


class TestPositionalEncoding:
    """Test positional encoding"""

    def test_pe_shape(self, load_code_module):
        """Test positional encoding shape"""
        pe_module = load_code_module("code/ch01_dsp/positional_encoding.py")
        seq_length = 100
        d_model = 64

        pe = pe_module.positional_encoding(seq_length, d_model)

        assert pe.shape == (seq_length, d_model)

    def test_pe_norm_stability(self, load_code_module):
        """Test that PE norm is approximately constant"""
        pe_module = load_code_module("code/ch01_dsp/positional_encoding.py")
        seq_length = 100
        d_model = 64

        pe = pe_module.positional_encoding(seq_length, d_model)
        norms = np.linalg.norm(pe, axis=1)

        # Norms should be relatively stable
        assert np.std(norms) < 0.5

    def test_pe_periodicity(self, load_code_module):
        """Test that PE has periodic structure"""
        pe_module = load_code_module("code/ch01_dsp/positional_encoding.py")
        seq_length = 200
        d_model = 64

        pe = pe_module.positional_encoding(seq_length, d_model)

        # Check periodicity in first dimension
        # Period should be approximately 10000
        period = 10000
        idx1 = 10
        idx2 = idx1 + int(period)

        if idx2 < seq_length:
            # Values should be similar after one period
            assert np.abs(pe[idx1, 0] - pe[idx2, 0]) < 0.1

    def test_pe_no_nan(self, load_code_module):
        """Test that PE doesn't contain NaN values"""
        pe_module = load_code_module("code/ch01_dsp/positional_encoding.py")
        seq_length = 100
        d_model = 64

        pe = pe_module.positional_encoding(seq_length, d_model)

        assert not np.any(np.isnan(pe))
        assert not np.any(np.isinf(pe))
