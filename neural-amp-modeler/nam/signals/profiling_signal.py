# File: profiling_signal.py
# Multi-phase profiling signal generator for parametric amp capture
#
# The signal is designed so that different phases excite different aspects
# of the amp's behavior. The training pipeline extracts characterization
# from the signal+response pair alone (no knob metadata needed).
#
# Signal structure:
#   1. Calibration blips (latency detection, v3-compatible)
#   2. Rising amplitude white noise (frequency response across gain range)
#   3. Pulsating white noise (attack/release, compression onset)
#   4. Exponential chirp groups (harmonic intermodulation)
#   5. Musical content (realistic dynamics)
#   6. Silence + validation signal

import math
from typing import Optional

import numpy as np


class ProfilingSignalGenerator:
    """
    Generates the multi-phase test signal for parametric amp profiling.

    The signal sweeps all components of the amplifier from preamp through
    EQ into power-amp. Different phases target different behaviors:

    - Rising noise maps frequency response at every gain level
    - Pulsating noise characterizes compression and transient response
    - Swept sines capture harmonic intermodulation
    - Musical content ensures realistic dynamic behavior

    Total duration: ~162.5 seconds at 48kHz
    Output: mono float64 array, peak normalized to [-1, 1]
    """

    SAMPLE_RATE = 48_000

    # Section durations in seconds
    BLIP_DURATION = 2.0
    RISING_NOISE_DURATION = 30.0
    PULSATING_NOISE_DURATION = 30.0
    CHIRP_DURATION = 30.0
    MUSICAL_DURATION = 60.0
    SILENCE_DURATION = 0.5
    VALIDATION_DURATION = 9.0

    # Blip parameters (compatible with NAM v3 latency calibration)
    BLIP_LOCATIONS = (24_000, 72_000)  # Sample indices within blip section
    BLIP_WIDTH = 100  # Samples per blip pulse

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    @property
    def total_duration(self) -> float:
        return (
            self.BLIP_DURATION
            + self.RISING_NOISE_DURATION
            + self.PULSATING_NOISE_DURATION
            + self.CHIRP_DURATION
            + self.MUSICAL_DURATION
            + self.SILENCE_DURATION
            + self.VALIDATION_DURATION
        )

    @property
    def total_samples(self) -> int:
        return int(self.total_duration * self.SAMPLE_RATE)

    def generate(self) -> np.ndarray:
        """Generate the complete profiling signal as a float64 array in [-1, 1]."""
        sections = [
            self._generate_blips(),
            self._generate_rising_noise(),
            self._generate_pulsating_noise(),
            self._generate_chirp_groups(),
            self._generate_musical_content(),
            self._generate_silence(),
            self._generate_validation_signal(),
        ]
        signal = np.concatenate(sections)
        # Peak normalize to 0.95 to leave headroom
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal * (0.95 / peak)
        return signal

    def save(self, filepath: str):
        """Generate and save the profiling signal as a 24-bit 48kHz WAV."""
        from ..data import np_to_wav

        signal = self.generate()
        np_to_wav(signal, filepath, rate=self.SAMPLE_RATE, sampwidth=3)

    # ---- Section generators ----

    def _generate_blips(self) -> np.ndarray:
        """
        Calibration blips for latency detection.
        Compatible with NAM v3 data format conventions.
        """
        n_samples = int(self.BLIP_DURATION * self.SAMPLE_RATE)
        signal = np.zeros(n_samples)

        for loc in self.BLIP_LOCATIONS:
            # Short half-sine pulse
            t = np.linspace(0, math.pi, self.BLIP_WIDTH)
            pulse = np.sin(t) * 0.8
            end = min(loc + self.BLIP_WIDTH, n_samples)
            signal[loc:end] = pulse[: end - loc]

        return signal

    def _generate_rising_noise(self) -> np.ndarray:
        """
        White noise with amplitude rising from -60dBFS to 0dBFS.

        Maps the amp's frequency response at every gain level.
        Early samples excite clean behavior, late samples excite driven behavior.
        """
        n_samples = int(self.RISING_NOISE_DURATION * self.SAMPLE_RATE)
        noise = self._rng.standard_normal(n_samples)

        # Amplitude envelope: exponential rise from -60dBFS to 0dBFS
        db_start = -60.0
        db_end = 0.0
        db_envelope = np.linspace(db_start, db_end, n_samples)
        amplitude = 10.0 ** (db_envelope / 20.0)

        return noise * amplitude

    def _generate_pulsating_noise(self) -> np.ndarray:
        """
        Alternating noise bursts at 6 gain levels.

        Burst pattern: 100ms on, 50ms off
        Each gain level gets ~5 seconds of bursts.
        Characterizes attack/release behavior and compression onset.
        """
        n_samples = int(self.PULSATING_NOISE_DURATION * self.SAMPLE_RATE)
        signal = np.zeros(n_samples)

        burst_on = int(0.100 * self.SAMPLE_RATE)  # 100ms
        burst_off = int(0.050 * self.SAMPLE_RATE)  # 50ms
        burst_cycle = burst_on + burst_off

        # 6 gain levels spanning the dynamic range
        gain_levels_db = [-48.0, -36.0, -24.0, -12.0, -6.0, 0.0]
        samples_per_level = n_samples // len(gain_levels_db)

        for i, db in enumerate(gain_levels_db):
            amplitude = 10.0 ** (db / 20.0)
            start = i * samples_per_level
            end = start + samples_per_level

            # Generate burst pattern within this level
            for j in range(start, end, burst_cycle):
                burst_end = min(j + burst_on, end)
                n_burst = burst_end - j
                if n_burst > 0:
                    signal[j:burst_end] = (
                        self._rng.standard_normal(n_burst) * amplitude
                    )

        return signal

    def _generate_chirp_groups(self) -> np.ndarray:
        """
        Groups of exponential sine sweeps at increasing amplitudes.

        Each group: exponential chirp from 20Hz to 20kHz over 2 seconds.
        Multiple groups at different amplitudes.
        Captures harmonic intermodulation and frequency-dependent distortion.

        The exponential chirp is chosen because deconvolution separates
        harmonic orders cleanly in the time domain (Novak et al. 2010).
        """
        n_samples = int(self.CHIRP_DURATION * self.SAMPLE_RATE)
        signal = np.zeros(n_samples)

        f_start = 20.0
        f_end = 20000.0
        chirp_duration = 2.0
        chirp_samples = int(chirp_duration * self.SAMPLE_RATE)

        # Amplitude levels for chirp groups
        amplitudes_db = [-36.0, -24.0, -12.0, -6.0, -3.0, 0.0]
        gap_samples = int(0.2 * self.SAMPLE_RATE)  # 200ms silence between chirps

        offset = 0
        for db in amplitudes_db:
            if offset + chirp_samples > n_samples:
                break
            amplitude = 10.0 ** (db / 20.0)
            chirp = self._generate_exponential_chirp(
                f_start, f_end, chirp_samples
            )
            end = min(offset + chirp_samples, n_samples)
            signal[offset:end] = chirp[: end - offset] * amplitude
            offset = end + gap_samples

        # Fill remaining time with one more full-amplitude chirp
        if offset + chirp_samples <= n_samples:
            chirp = self._generate_exponential_chirp(f_start, f_end, chirp_samples)
            end = min(offset + chirp_samples, n_samples)
            signal[offset:end] = chirp[: end - offset]

        return signal

    def _generate_exponential_chirp(
        self, f_start: float, f_end: float, n_samples: int
    ) -> np.ndarray:
        """
        Generate an exponential (logarithmic) sine sweep.

        Frequency increases exponentially from f_start to f_end.
        """
        t = np.arange(n_samples) / self.SAMPLE_RATE
        T = n_samples / self.SAMPLE_RATE
        k = (f_end / f_start) ** (1.0 / T)

        phase = 2.0 * math.pi * f_start * (k**t - 1.0) / math.log(k)

        # Apply fade-in/fade-out to avoid clicks
        fade_samples = int(0.01 * self.SAMPLE_RATE)  # 10ms fade
        envelope = np.ones(n_samples)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            envelope[:fade_samples] = fade_in
            envelope[-fade_samples:] = fade_out

        return np.sin(phase) * envelope

    def _generate_musical_content(self) -> np.ndarray:
        """
        Synthesized guitar-like signals with varying dynamics.

        Uses Karplus-Strong-style plucked string synthesis to create
        realistic guitar signal dynamics (sharp attack, exponential decay,
        varying pitch and intensity).
        """
        n_samples = int(self.MUSICAL_DURATION * self.SAMPLE_RATE)
        signal = np.zeros(n_samples)

        # Note events: (start_time_sec, frequency_hz, duration_sec, velocity)
        events = self._generate_note_events()

        for start_sec, freq, duration_sec, velocity in events:
            start_sample = int(start_sec * self.SAMPLE_RATE)
            n_note = int(duration_sec * self.SAMPLE_RATE)
            if start_sample + n_note > n_samples:
                n_note = n_samples - start_sample
            if n_note <= 0:
                continue

            note = self._synthesize_plucked_string(freq, n_note, velocity)
            signal[start_sample : start_sample + n_note] += note

        return signal

    def _generate_note_events(self):
        """
        Generate a sequence of note events spanning various dynamics.

        Covers: single notes (quiet to loud), power chords, fast runs,
        palm-muted chugs, sustained bends.
        """
        events = []
        t = 0.0

        # Guitar frequencies (standard tuning, various positions)
        low_e = 82.41
        a_string = 110.0
        d_string = 146.83
        g_string = 196.0
        b_string = 246.94
        high_e = 329.63

        # Section 1: Single notes at increasing dynamics (0-15s)
        for i, velocity in enumerate(np.linspace(0.1, 1.0, 12)):
            freq = [low_e, a_string, d_string, g_string, b_string, high_e][i % 6]
            freq *= (2 ** (self._rng.integers(0, 12) / 12.0))  # Random fret
            events.append((t, freq, 1.0, velocity))
            t += 1.2

        # Section 2: Power chords (15-30s)
        for i in range(10):
            root = low_e * (2 ** (self._rng.integers(0, 7) / 12.0))
            fifth = root * 1.5
            octave = root * 2.0
            velocity = 0.5 + self._rng.random() * 0.5
            events.append((t, root, 1.2, velocity * 0.5))
            events.append((t, fifth, 1.2, velocity * 0.35))
            events.append((t, octave, 1.2, velocity * 0.3))
            t += 1.5

        # Section 3: Fast picking / runs (30-42s)
        for i in range(48):
            freq = a_string * (2 ** (self._rng.integers(0, 24) / 12.0))
            velocity = 0.3 + self._rng.random() * 0.7
            events.append((t, freq, 0.15, velocity))
            t += 0.25

        # Section 4: Palm-muted chugs — short, heavily damped (42-50s)
        for i in range(32):
            freq = low_e * (2 ** (self._rng.integers(0, 5) / 12.0))
            velocity = 0.6 + self._rng.random() * 0.4
            events.append((t, freq, 0.08, velocity))
            t += 0.25

        # Section 5: Sustained notes with varying dynamics (50-60s)
        for i in range(5):
            freq = g_string * (2 ** (self._rng.integers(0, 12) / 12.0))
            velocity = 0.4 + self._rng.random() * 0.6
            events.append((t, freq, 1.8, velocity))
            t += 2.0

        return events

    def _synthesize_plucked_string(
        self, frequency: float, n_samples: int, velocity: float = 1.0
    ) -> np.ndarray:
        """
        Simple Karplus-Strong plucked string synthesis.

        :param frequency: fundamental frequency in Hz
        :param n_samples: number of output samples
        :param velocity: pick intensity (0 to 1), controls initial noise amplitude
        """
        period = int(self.SAMPLE_RATE / frequency)
        if period < 2:
            period = 2

        # Initialize delay line with filtered noise (velocity controls amplitude)
        delay_line = self._rng.uniform(-velocity, velocity, period).astype(np.float64)

        output = np.zeros(n_samples)
        idx = 0

        # Low-pass filtering coefficient (controls decay/brightness)
        # Higher frequency strings decay faster
        decay = 0.996 - (frequency / 20000.0) * 0.01

        for n in range(n_samples):
            output[n] = delay_line[idx]
            # Average adjacent samples (low-pass) with decay
            next_idx = (idx + 1) % period
            delay_line[idx] = decay * 0.5 * (delay_line[idx] + delay_line[next_idx])
            idx = next_idx

        return output

    def _generate_silence(self) -> np.ndarray:
        """Short silence before validation section."""
        return np.zeros(int(self.SILENCE_DURATION * self.SAMPLE_RATE))

    def _generate_validation_signal(self) -> np.ndarray:
        """
        Validation signal for model evaluation.

        Mix of swept sine, noise bursts, and musical content at moderate level.
        Placed at the end of the signal to match NAM convention (validation last).
        """
        n_samples = int(self.VALIDATION_DURATION * self.SAMPLE_RATE)
        signal = np.zeros(n_samples)

        third = n_samples // 3

        # 1/3: Mid-level exponential chirp
        chirp = self._generate_exponential_chirp(20.0, 20000.0, third)
        signal[:third] = chirp * 0.5

        # 1/3: Mid-level noise
        signal[third : 2 * third] = (
            self._rng.standard_normal(third) * 0.3
        )

        # 1/3: Musical content at moderate dynamics
        remaining = n_samples - 2 * third
        notes = np.zeros(remaining)
        freqs = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]
        note_len = remaining // len(freqs)
        for i, freq in enumerate(freqs):
            start = i * note_len
            end = min(start + note_len, remaining)
            n = end - start
            if n > 0:
                notes[start:end] = self._synthesize_plucked_string(
                    freq, n, velocity=0.5
                )
        signal[2 * third :] = notes[: n_samples - 2 * third]

        return signal
