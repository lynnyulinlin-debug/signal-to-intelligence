"""Tests for Chapter 1: DSP Basics"""
import numpy as np
import pytest
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


class TestFFTSpectrum:
    """Test FFT spectrum analysis"""

    def test_fft_output_shape(self, seed):
        """Test that FFT output has correct shape"""
        signal_length = 1000
        signal = np.sin(2 * np.pi * 5 * np.arange(signal_length) / 100)

        fft_result = np.fft.fft(signal)
        assert fft_result.shape == (signal_length,)

    def test_fft_symmetry(self, seed):
        """Test FFT symmetry property for real signals"""
        signal = np.random.randn(100)
        fft_result = np.fft.fft(signal)

        # For real signals, FFT should be symmetric
        assert np.allclose(fft_result[1:50], np.conj(fft_result[-1:-50:-1]))

    def test_fft_frequency_detection(self, seed):
        """Test that FFT correctly detects signal frequencies"""
        sampling_rate = 100
        signal_length = 1000
        freq1, freq2 = 5, 10

        t = np.arange(signal_length) / sampling_rate
        signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)

        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(signal_length, 1 / sampling_rate)
        magnitude = np.abs(fft_result)

        # Find peaks
        positive_freq_idx = frequencies > 0
        frequencies_positive = frequencies[positive_freq_idx]
        magnitude_positive = magnitude[positive_freq_idx]

        top_indices = np.argsort(magnitude_positive)[-2:][::-1]
        detected_freqs = frequencies_positive[top_indices]

        # Check if detected frequencies are close to actual frequencies
        assert np.any(np.abs(detected_freqs - freq1) < 1)
        assert np.any(np.abs(detected_freqs - freq2) < 1)

    def test_fft_noise_robustness(self, seed):
        """Test FFT with noisy signal"""
        signal_length = 1000
        signal = np.sin(2 * np.pi * 5 * np.arange(signal_length) / 100)
        noise = 0.5 * np.random.randn(signal_length)
        signal_noisy = signal + noise

        fft_result = np.fft.fft(signal_noisy)
        assert fft_result.shape == (signal_length,)
        assert not np.any(np.isnan(fft_result))


class TestPositionalEncoding:
    """Test positional encoding"""

    def test_pe_shape(self):
        """Test positional encoding shape"""
        seq_length = 100
        d_model = 64

        pe = np.zeros((seq_length, d_model))
        position = np.arange(seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        assert pe.shape == (seq_length, d_model)

    def test_pe_norm_stability(self):
        """Test that PE norm is approximately constant"""
        seq_length = 100
        d_model = 64

        pe = np.zeros((seq_length, d_model))
        position = np.arange(seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        norms = np.linalg.norm(pe, axis=1)

        # Norms should be relatively stable
        assert np.std(norms) < 0.5

    def test_pe_periodicity(self):
        """Test that PE has periodic structure"""
        seq_length = 200
        d_model = 64

        pe = np.zeros((seq_length, d_model))
        position = np.arange(seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Check periodicity in first dimension
        # Period should be approximately 10000
        period = 10000
        idx1 = 10
        idx2 = idx1 + int(period)

        if idx2 < seq_length:
            # Values should be similar after one period
            assert np.abs(pe[idx1, 0] - pe[idx2, 0]) < 0.1

    def test_pe_no_nan(self):
        """Test that PE doesn't contain NaN values"""
        seq_length = 100
        d_model = 64

        pe = np.zeros((seq_length, d_model))
        position = np.arange(seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        assert not np.any(np.isnan(pe))
        assert not np.any(np.isinf(pe))
