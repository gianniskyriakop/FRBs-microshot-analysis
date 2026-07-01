"""
will_microshot_pipeline.py

WILL + FLITS-ready workflow for Giannis' FRB microshot-complexity project.

Goal
----
Simulate FRB-like bursts that have comparable individual peak timescales but
varying morphological complexity, then quantify the differences with metrics
that go beyond a single ACF/PSD timescale.

Intended scientific workflow
----------------------------
1. Define an intrinsic microshot forest: shot times, widths, amplitudes, and
   optional frequency structure.
2. Use WILL as the simulation layer to make a dynamic spectrum.
3. Dedisperse/collapse the dynamic spectrum, or pass the dynamic spectrum to
   FLITS and read the FLITS product back in.
4. Measure ACF timescale + complexity metrics.
5. Select bursts with similar ACF timescale but different complexity.

Why the fallback exists
-----------------------
The preferred simulator is WILL. However, many analysis environments do not
have WILL installed. This module therefore contains a small numpy fallback that
mimics the same input/output shape. The fallback is useful for debugging the
metrics, but the science runs should use WILL.

Install WILL in your science environment with:
    pip install git+https://github.com/josephwkania/will.git

The module tries to import:
    from will import create, detect
If that succeeds, `simulate_microshot_dynamic_spectrum(..., prefer_will=True)`
uses WILL's `create.TwoDimensionalPulse` by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import correlate, find_peaks, peak_widths, periodogram

try:  # WILL is optional at import time, but preferred at simulation time.
    from will import create as will_create
    from will import detect as will_detect

    WILL_AVAILABLE = True
except Exception:  # pragma: no cover - this depends on the local environment.
    will_create = None
    will_detect = None
    WILL_AVAILABLE = False


ArrayLike = Union[np.ndarray, Sequence[float]]
ComplexityMode = Literal["sparse", "moderate", "dense", "clustered", "quasiperiodic"]
WillMethod = Literal["twodim", "gausspulse"]


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class InstrumentConfig:
    """
    Basic dynamic-spectrum/instrument setup.

    Parameters
    ----------
    nfreq
        Number of frequency channels.
    freq_low_mhz, freq_high_mhz
        Frequency band edges in MHz.
    tsamp_ms
        Time resolution in milliseconds.
    duration_ms
        Zero-DM simulated burst window before dispersion padding.
    dm
        Dispersion measure in pc cm^-3. For FRB 20220912A-like tests you can
        start around ~220 pc cm^-3, but keep this as an explicit parameter.
    noise_std
        Gaussian noise standard deviation per dynamic-spectrum pixel.
    peak_snr
        Approximate peak S/N of the frequency-averaged profile after scaling.
    """

    nfreq: int = 128
    freq_low_mhz: float = 1100.0
    freq_high_mhz: float = 1800.0
    tsamp_ms: float = 0.016
    duration_ms: float = 12.0
    dm: float = 220.0
    noise_std: float = 1.0
    peak_snr: float = 25.0

    @property
    def tsamp_s(self) -> float:
        return self.tsamp_ms / 1000.0

    @property
    def chan_freqs_mhz(self) -> np.ndarray:
        # Descending order is common in filterbank data and works with WILL.
        return np.linspace(self.freq_high_mhz, self.freq_low_mhz, self.nfreq)

    @property
    def nsamps(self) -> int:
        return int(np.round(self.duration_ms / self.tsamp_ms))


@dataclass
class MicroshotForestConfig:
    """
    Intrinsic burst morphology.

    The important experimental control is `shot_sigma_ms`: keep this fixed
    while changing `n_peaks` and `complexity_mode`.
    """

    n_peaks: int
    shot_sigma_ms: float = 0.080
    envelope_sigma_ms: float = 1.2
    min_sep_ms: Optional[float] = None
    complexity_mode: ComplexityMode = "moderate"
    amplitude_lognormal_sigma: float = 0.45
    freq_sigma_mhz: float = 180.0
    center_freq_mhz: Optional[float] = None
    center_freq_jitter_mhz: float = 70.0
    drift_mhz_per_ms: float = -40.0
    theta_rad: float = 0.0
    spectral_index_alpha: float = 0.0
    nscint: int = 0
    scint_phi: float = 0.0
    tau_will_samples: float = 0.0
    total_counts: int = 250_000
    seed: Optional[int] = None


@dataclass
class SimulatedBurst:
    """One simulated burst and its truth metadata."""

    burst_id: str
    t_ms: np.ndarray
    dynamic_spectrum: np.ndarray
    profile: np.ndarray
    shot_times_ms: np.ndarray
    amplitudes: np.ndarray
    center_freqs_mhz: np.ndarray
    shot_sigma_ms: float
    input_n_peaks: int
    used_will: bool
    will_method: str
    instrument: InstrumentConfig
    forest: MicroshotForestConfig
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def robust_sigma(x: ArrayLike) -> float:
    """Median-absolute-deviation standard deviation estimate."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad <= 0 or not np.isfinite(mad):
        std = np.nanstd(x)
        return float(std if std > 0 and np.isfinite(std) else 1.0)
    return float(1.4826 * mad)


def normalise_profile(profile: ArrayLike, offpulse: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Return a 1D profile in S/N-like units."""
    p = np.asarray(profile, dtype=float).copy()
    if offpulse is not None:
        off = p[offpulse[0] : offpulse[1]]
    else:
        edge = max(1, int(0.20 * p.size))
        off = np.concatenate([p[:edge], p[-edge:]])
    mu = np.nanmedian(off)
    sig = robust_sigma(off)
    return (p - mu) / sig


def positive_part(profile: ArrayLike) -> np.ndarray:
    """Baseline-subtracted positive part of a profile."""
    p = np.asarray(profile, dtype=float)
    return np.clip(p - np.nanmedian(p), 0, None)


def gini_coefficient(x: ArrayLike) -> float:
    """Gini coefficient for non-negative values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0 or np.all(x == 0):
        return np.nan
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) @ x) / (n * np.sum(x)) - (n + 1) / n)


def shannon_entropy_fraction(x: ArrayLike) -> float:
    """
    Normalized Shannon entropy in [0, 1].

    For peak amplitudes, low values mean one/few peaks dominate; high values
    mean the burst energy is more evenly distributed across peaks.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size <= 1:
        return 0.0
    p = x / np.sum(x)
    H = -np.sum(p * np.log(p))
    return float(H / np.log(x.size))


def t90_width_ms(profile: ArrayLike, tsamp_ms: float) -> float:
    """Fluence T90 width in milliseconds."""
    y = positive_part(profile)
    total = np.nansum(y)
    if total <= 0:
        return np.nan
    cdf = np.cumsum(y) / total
    i5 = int(np.searchsorted(cdf, 0.05))
    i95 = int(np.searchsorted(cdf, 0.95))
    return float((i95 - i5) * tsamp_ms)


def ensure_2d_time_frequency(ds: np.ndarray) -> np.ndarray:
    """
    Ensure dynamic spectrum is shaped (time, frequency).

    Dante/FLITS-style arrays are sometimes stored as (frequency, time). This
    function does not guess aggressively; use it only when you know your axes.
    """
    arr = np.asarray(ds, dtype=float)
    if arr.ndim != 2:
        raise ValueError("dynamic spectrum must be 2D")
    return arr


# ---------------------------------------------------------------------------
# Dispersion helpers for fallback and FLITS products
# ---------------------------------------------------------------------------


def dispersion_delay_ms(dm: float, chan_freqs_mhz: np.ndarray, ref_freq_mhz: Optional[float] = None) -> np.ndarray:
    """
    Cold-plasma dispersion delay in milliseconds relative to `ref_freq_mhz`.

    Frequencies are in MHz. Delays are positive for lower frequencies when the
    reference is the highest frequency in the band.
    """
    freqs = np.asarray(chan_freqs_mhz, dtype=float)
    if ref_freq_mhz is None:
        ref_freq_mhz = float(np.nanmax(freqs))
    k_dm_ms = 4.148808e6  # ms MHz^2 pc^-1 cm^3
    return k_dm_ms * dm * (freqs ** -2 - ref_freq_mhz ** -2)


def shift_columns_integer(ds: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Shift each frequency channel by an integer number of time bins.

    Positive shift moves signal to later time samples. Values shifted outside
    the array are replaced with zero.
    """
    arr = np.asarray(ds, dtype=float)
    out = np.zeros_like(arr)
    nt, nf = arr.shape
    shifts = np.asarray(shifts, dtype=int)
    if shifts.size != nf:
        raise ValueError("number of shifts must match number of frequency channels")

    for j, s in enumerate(shifts):
        if s == 0:
            out[:, j] = arr[:, j]
        elif s > 0:
            if s < nt:
                out[s:, j] = arr[: nt - s, j]
        else:
            if -s < nt:
                out[: nt + s, j] = arr[-s:, j]
    return out


def disperse_or_dedisperse_integer(
    ds: np.ndarray,
    dm: float,
    tsamp_ms: float,
    chan_freqs_mhz: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    """
    Simple integer-bin dispersion/dedispersion.

    This is not intended to replace WILL/JESS/FLITS; it just makes the notebook
    runnable when WILL is unavailable.
    """
    delays = dispersion_delay_ms(dm, chan_freqs_mhz)
    shifts = np.rint(delays / tsamp_ms).astype(int)
    if inverse:
        shifts = -shifts
    return shift_columns_integer(ds, shifts)


def profile_from_dynamic_spectrum(
    dynamic_spectrum: np.ndarray,
    instrument: InstrumentConfig,
    dedisperse: bool = True,
    use_will_dedispersion: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """
    Convert dynamic spectrum with shape (time, frequency) to a 1D profile.

    If WILL is installed and `use_will_dedispersion=True`, use
    `will.detect.dedisped_time_series`. Otherwise use the integer-bin fallback.
    """
    ds = ensure_2d_time_frequency(dynamic_spectrum)

    if dedisperse:
        if use_will_dedispersion and WILL_AVAILABLE:
            try:
                prof = will_detect.dedisped_time_series(
                    ds,
                    dm=instrument.dm,
                    tsamp=instrument.tsamp_s,
                    chan_freqs=instrument.chan_freqs_mhz,
                )
                prof = np.asarray(prof, dtype=float)
            except Exception:
                ds_dd = disperse_or_dedisperse_integer(
                    ds, instrument.dm, instrument.tsamp_ms, instrument.chan_freqs_mhz, inverse=True
                )
                prof = np.nanmean(ds_dd, axis=1)
        else:
            ds_dd = disperse_or_dedisperse_integer(
                ds, instrument.dm, instrument.tsamp_ms, instrument.chan_freqs_mhz, inverse=True
            )
            prof = np.nanmean(ds_dd, axis=1)
    else:
        prof = np.nanmean(ds, axis=1)

    return normalise_profile(prof) if normalise else prof


# ---------------------------------------------------------------------------
# Intrinsic microshot forest generation
# ---------------------------------------------------------------------------


def draw_microshot_times(
    rng: np.random.Generator,
    n_peaks: int,
    duration_ms: float,
    mode: ComplexityMode = "moderate",
    envelope_sigma_ms: float = 1.2,
    min_sep_ms: Optional[float] = None,
) -> np.ndarray:
    """Draw zero-DM microshot arrival times inside the simulated window."""
    if min_sep_ms is None:
        min_sep_ms = 0.0
    centre = 0.5 * duration_ms
    lo = max(0.5, centre - 4 * envelope_sigma_ms)
    hi = min(duration_ms - 0.5, centre + 4 * envelope_sigma_ms)

    if mode == "sparse":
        times = rng.uniform(lo, hi, size=n_peaks)
    elif mode == "moderate":
        times = rng.normal(centre, envelope_sigma_ms, size=n_peaks)
    elif mode == "dense":
        times = rng.normal(centre, 0.55 * envelope_sigma_ms, size=n_peaks)
    elif mode == "clustered":
        n_clusters = max(2, min(5, int(np.sqrt(n_peaks))))
        cluster_centres = rng.normal(centre, envelope_sigma_ms, size=n_clusters)
        assignments = rng.integers(0, n_clusters, size=n_peaks)
        times = cluster_centres[assignments] + rng.normal(0, 0.18 * envelope_sigma_ms, size=n_peaks)
    elif mode == "quasiperiodic":
        period = (hi - lo) / max(n_peaks - 1, 1)
        times = lo + period * np.arange(n_peaks)
        times += rng.normal(0, 0.18 * period, size=n_peaks)
    else:
        raise ValueError(f"Unknown complexity mode: {mode}")

    times = np.clip(times, 0.2, duration_ms - 0.2)
    times = np.sort(times)

    if min_sep_ms > 0 and times.size > 1:
        kept = [float(times[0])]
        for t in times[1:]:
            if t - kept[-1] >= min_sep_ms:
                kept.append(float(t))
            else:
                kept.append(float(min(duration_ms - 0.2, kept[-1] + min_sep_ms)))
        times = np.asarray(kept)

    return np.sort(times)


def draw_microshot_amplitudes(
    rng: np.random.Generator,
    n_peaks: int,
    sigma: float = 0.45,
) -> np.ndarray:
    """Draw relative microshot amplitudes."""
    amps = rng.lognormal(mean=0.0, sigma=sigma, size=n_peaks)
    return amps / np.nanmax(amps)


def draw_microshot_center_freqs(
    rng: np.random.Generator,
    shot_times_ms: np.ndarray,
    instrument: InstrumentConfig,
    forest: MicroshotForestConfig,
) -> np.ndarray:
    """
    Draw central frequencies for the shots.

    A negative `drift_mhz_per_ms` makes later shots lower in frequency.
    """
    if forest.center_freq_mhz is None:
        centre = 0.5 * (instrument.freq_low_mhz + instrument.freq_high_mhz)
    else:
        centre = forest.center_freq_mhz
    t0 = np.nanmedian(shot_times_ms)
    freqs = centre + forest.drift_mhz_per_ms * (shot_times_ms - t0)
    freqs += rng.normal(0, forest.center_freq_jitter_mhz, size=shot_times_ms.size)
    return np.clip(freqs, instrument.freq_low_mhz, instrument.freq_high_mhz)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Unit-height Gaussian PDF-like profile."""
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_zero_dm_microshot_pdf(
    shot_times_ms: np.ndarray,
    amplitudes: np.ndarray,
    center_freqs_mhz: np.ndarray,
    instrument: InstrumentConfig,
    forest: MicroshotForestConfig,
) -> np.ndarray:
    """
    Build a zero-DM dynamic-spectrum pulse PDF with shape (time, frequency).

    This PDF is passed into WILL's `create.TwoDimensionalPulse` when WILL is
    available. The same PDF is also used by the fallback simulator.
    """
    nt = instrument.nsamps
    freqs = instrument.chan_freqs_mhz
    t_ms = np.arange(nt) * instrument.tsamp_ms
    pdf = np.zeros((nt, freqs.size), dtype=float)

    for t0, amp, f0 in zip(shot_times_ms, amplitudes, center_freqs_mhz):
        time_prof = gaussian_pdf(t_ms, t0, forest.shot_sigma_ms)
        freq_prof = gaussian_pdf(freqs, f0, forest.freq_sigma_mhz)
        pdf += amp * time_prof[:, None] * freq_prof[None, :]

    if forest.spectral_index_alpha != 0:
        fref = np.nanmedian(freqs)
        pdf *= (freqs / fref) ** forest.spectral_index_alpha

    if forest.nscint and forest.nscint > 0:
        if WILL_AVAILABLE:
            scint = will_create.scintillation(
                chan_freqs=freqs,
                freq_ref=np.nanmedian(freqs),
                nscint=forest.nscint,
                phi=forest.scint_phi,
            )
        else:
            scint = np.abs(np.cos(2 * np.pi * forest.nscint * (freqs / np.nanmedian(freqs)) ** 2 + forest.scint_phi))
        pdf *= scint[None, :]

    pdf = np.clip(pdf, 0, None)
    if np.nanmax(pdf) > 0:
        pdf /= np.nanmax(pdf)
    return pdf


# ---------------------------------------------------------------------------
# WILL simulation layer
# ---------------------------------------------------------------------------


def _scale_pulse_to_profile_snr(pulse: np.ndarray, noise_std: float, peak_snr: float) -> np.ndarray:
    """Scale pulse so its frequency-averaged profile has approximately peak_snr."""
    pulse = np.asarray(pulse, dtype=float)
    prof = np.nanmean(pulse, axis=1)
    prof_peak = np.nanmax(prof)
    if not np.isfinite(prof_peak) or prof_peak <= 0:
        return pulse
    return pulse * (peak_snr * noise_std / prof_peak)


def simulate_microshot_dynamic_spectrum(
    forest: MicroshotForestConfig,
    instrument: InstrumentConfig = InstrumentConfig(),
    prefer_will: bool = True,
    will_method: WillMethod = "twodim",
    burst_id: Optional[str] = None,
) -> SimulatedBurst:
    """
    Simulate one microshot burst dynamic spectrum.

    Preferred path
    --------------
    If WILL is installed and `prefer_will=True`, this uses WILL. The default
    `will_method="twodim"` passes a custom multi-microshot PDF to
    `will.create.TwoDimensionalPulse`, which is robust for many microshots.

    Alternative
    -----------
    `will_method="gausspulse"` uses WILL's `create.GaussPulse` directly. This
    is closest to the official WILL example for multi-component bursts, but can
    be slower for many components.

    Fallback
    --------
    If WILL is not installed, a numpy fallback creates a dispersed dynamic
    spectrum. Use this only for metric development.
    """
    rng = np.random.default_rng(forest.seed)
    if burst_id is None:
        burst_id = f"{forest.complexity_mode}_n{forest.n_peaks}_seed{forest.seed}"

    shot_times = draw_microshot_times(
        rng,
        n_peaks=forest.n_peaks,
        duration_ms=instrument.duration_ms,
        mode=forest.complexity_mode,
        envelope_sigma_ms=forest.envelope_sigma_ms,
        min_sep_ms=forest.min_sep_ms,
    )
    amps = draw_microshot_amplitudes(rng, forest.n_peaks, forest.amplitude_lognormal_sigma)
    center_freqs = draw_microshot_center_freqs(rng, shot_times, instrument, forest)
    pdf = build_zero_dm_microshot_pdf(shot_times, amps, center_freqs, instrument, forest)

    used_will = bool(prefer_will and WILL_AVAILABLE)

    if used_will and will_method == "twodim":
        pulse_obj = will_create.TwoDimensionalPulse(
            pulse_pdf=pdf,
            chan_freqs=instrument.chan_freqs_mhz,
            tsamp=instrument.tsamp_s,
            dm=instrument.dm,
        )
        pulse = pulse_obj.sample_pulse(nsamp=int(forest.total_counts), dtype=np.float64)
    elif used_will and will_method == "gausspulse":
        # Direct WILL component model. Offsets are in seconds from start of window.
        pulse_obj = will_create.GaussPulse(
            relative_intensities=amps,
            sigma_times=np.full(forest.n_peaks, forest.shot_sigma_ms / 1000.0),
            sigma_freqs=np.full(forest.n_peaks, forest.freq_sigma_mhz),
            center_freqs=center_freqs,
            pulse_thetas=np.full(forest.n_peaks, forest.theta_rad),
            offsets=shot_times / 1000.0,
            dm=instrument.dm,
            tau=forest.tau_will_samples,
            chan_freqs=instrument.chan_freqs_mhz,
            tsamp=instrument.tsamp_s,
            spectral_index_alpha=forest.spectral_index_alpha,
            nscint=forest.nscint,
            phi=forest.scint_phi,
            bandpass=None,
            dm_interchan_smear=False,
        )
        pulse = pulse_obj.sample_pulse(nsamp=int(forest.total_counts), dtype=np.float64)
    else:
        # Fallback: scale PDF, pad by the maximum dispersion delay, then
        # apply integer-bin dispersion. This keeps low-frequency channels from
        # being shifted entirely out of the window when DM is large.
        pulse_zero_dm = pdf * instrument.peak_snr * instrument.noise_std
        shifts = np.rint(
            dispersion_delay_ms(instrument.dm, instrument.chan_freqs_mhz) / instrument.tsamp_ms
        ).astype(int)
        pad = int(max(0, np.nanmax(shifts)))
        if pad > 0:
            pulse_zero_dm = np.pad(pulse_zero_dm, ((0, pad), (0, 0)), mode="constant")
        pulse = shift_columns_integer(pulse_zero_dm, shifts)

    pulse = _scale_pulse_to_profile_snr(pulse, instrument.noise_std, instrument.peak_snr)
    noise = rng.normal(0.0, instrument.noise_std, size=pulse.shape)
    dynamic_spectrum = noise + pulse
    profile = profile_from_dynamic_spectrum(
        dynamic_spectrum,
        instrument=instrument,
        dedisperse=True,
        use_will_dedispersion=used_will,
        normalise=True,
    )
    t_ms = np.arange(profile.size) * instrument.tsamp_ms

    return SimulatedBurst(
        burst_id=burst_id,
        t_ms=t_ms,
        dynamic_spectrum=dynamic_spectrum,
        profile=profile,
        shot_times_ms=shot_times,
        amplitudes=amps,
        center_freqs_mhz=center_freqs,
        shot_sigma_ms=forest.shot_sigma_ms,
        input_n_peaks=forest.n_peaks,
        used_will=used_will,
        will_method=will_method if used_will else "numpy_fallback",
        instrument=instrument,
        forest=forest,
        metadata={
            "WILL_AVAILABLE": WILL_AVAILABLE,
            "prefer_will": prefer_will,
            "actual_dynamic_shape": tuple(dynamic_spectrum.shape),
        },
    )


def make_microshot_bank(
    n_peaks_grid: Sequence[int] = (3, 6, 12, 24, 36),
    complexity_modes: Sequence[ComplexityMode] = ("sparse", "moderate", "dense", "clustered"),
    n_realizations: int = 8,
    instrument: InstrumentConfig = InstrumentConfig(),
    shot_sigma_ms: float = 0.080,
    envelope_sigma_ms: float = 1.2,
    prefer_will: bool = True,
    will_method: WillMethod = "twodim",
    seed: int = 12345,
) -> List[SimulatedBurst]:
    """Create a bank of simulated bursts."""
    bursts: List[SimulatedBurst] = []
    idx = 0
    for mode in complexity_modes:
        for n_peaks in n_peaks_grid:
            for r in range(n_realizations):
                idx += 1
                forest = MicroshotForestConfig(
                    n_peaks=int(n_peaks),
                    shot_sigma_ms=shot_sigma_ms,
                    envelope_sigma_ms=envelope_sigma_ms,
                    complexity_mode=mode,
                    seed=seed + idx,
                )
                burst_id = f"{mode}_n{n_peaks:02d}_r{r:02d}"
                bursts.append(
                    simulate_microshot_dynamic_spectrum(
                        forest=forest,
                        instrument=instrument,
                        prefer_will=prefer_will,
                        will_method=will_method,
                        burst_id=burst_id,
                    )
                )
    return bursts


# ---------------------------------------------------------------------------
# ACF, PSD, and peak-complexity metrics
# ---------------------------------------------------------------------------


def autocorrelation_1d(profile: ArrayLike, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized one-sided autocorrelation function."""
    y = np.asarray(profile, dtype=float)
    y = y - np.nanmean(y)
    y = np.nan_to_num(y, nan=0.0)
    acf_full = correlate(y, y, mode="full", method="fft")
    mid = acf_full.size // 2
    acf = acf_full[mid:]
    if acf[0] != 0:
        acf = acf / acf[0]
    lags = np.arange(acf.size)
    if max_lag is not None:
        acf = acf[: max_lag + 1]
        lags = lags[: max_lag + 1]
    return lags, acf


def gaussian_acf_model(lag_ms: np.ndarray, sigma_ms: float, floor: float) -> np.ndarray:
    return (1.0 - floor) * np.exp(-0.5 * (lag_ms / sigma_ms) ** 2) + floor


def fit_acf_timescale_ms(
    profile: ArrayLike,
    tsamp_ms: float,
    fit_max_lag_ms: float = 2.5,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Gaussian model to the central ACF peak.

    Returns
    -------
    sigma_ms, fwhm_ms, lag_ms, acf, model
    """
    max_lag = max(3, int(np.round(fit_max_lag_ms / tsamp_ms)))
    lags, acf = autocorrelation_1d(profile, max_lag=max_lag)
    lag_ms = lags * tsamp_ms

    mask = (lag_ms > 0) & np.isfinite(acf)
    if np.sum(mask) < 5:
        return np.nan, np.nan, lag_ms, acf, np.full_like(acf, np.nan)

    # Initial guess from half-maximum crossing.
    below_half = np.where(acf < 0.5)[0]
    if below_half.size > 0:
        sigma0 = max(tsamp_ms, lag_ms[below_half[0]] / np.sqrt(2 * np.log(2)))
    else:
        sigma0 = max(tsamp_ms, 0.2 * fit_max_lag_ms)

    try:
        popt, _ = curve_fit(
            gaussian_acf_model,
            lag_ms[mask],
            acf[mask],
            p0=[sigma0, 0.0],
            bounds=([0.25 * tsamp_ms, -0.5], [fit_max_lag_ms * 2, 0.5]),
            maxfev=10000,
        )
        sigma_ms = float(popt[0])
        model = gaussian_acf_model(lag_ms, *popt)
    except Exception:
        sigma_ms = np.nan
        model = np.full_like(acf, np.nan)

    fwhm_ms = float(2 * np.sqrt(2 * np.log(2)) * sigma_ms) if np.isfinite(sigma_ms) else np.nan
    return sigma_ms, fwhm_ms, lag_ms, acf, model


def estimate_psd_slope(
    profile: ArrayLike,
    tsamp_ms: float,
    fmin_hz: Optional[float] = None,
    fmax_hz: Optional[float] = None,
) -> float:
    """
    Estimate log-log PSD slope over a simple frequency interval.

    This is only a compact descriptor. Use Stingray for publication-quality PSDs.
    """
    y = np.asarray(profile, dtype=float)
    y = y - np.nanmean(y)
    fs_hz = 1000.0 / tsamp_ms
    f, pxx = periodogram(y, fs=fs_hz, scaling="density")
    mask = (f > 0) & (pxx > 0) & np.isfinite(pxx)
    if fmin_hz is not None:
        mask &= f >= fmin_hz
    if fmax_hz is not None:
        mask &= f <= fmax_hz
    if np.sum(mask) < 5:
        return np.nan
    coeff = np.polyfit(np.log10(f[mask]), np.log10(pxx[mask]), deg=1)
    return float(coeff[0])


def analyse_profile(
    profile: ArrayLike,
    tsamp_ms: float,
    burst_id: str = "burst",
    input_n_peaks: Optional[int] = None,
    shot_sigma_ms: Optional[float] = None,
    peak_sigma_threshold: float = 5.0,
    min_peak_distance_ms: Optional[float] = None,
    acf_fit_max_lag_ms: float = 2.5,
) -> Dict[str, float]:
    """Measure ACF timescale and complexity metrics for a 1D profile."""
    p = normalise_profile(profile)
    nt = p.size
    if min_peak_distance_ms is None:
        min_peak_distance_ms = max(tsamp_ms, 0.5 * (shot_sigma_ms or tsamp_ms))
    min_dist = max(1, int(np.round(min_peak_distance_ms / tsamp_ms)))

    peaks, props = find_peaks(p, height=peak_sigma_threshold, distance=min_dist)
    heights = props.get("peak_heights", np.array([], dtype=float))
    if peaks.size > 0:
        widths_samples = peak_widths(p, peaks, rel_height=0.5)[0]
        widths_ms = widths_samples * tsamp_ms
    else:
        widths_ms = np.array([], dtype=float)

    sep_ms = np.diff(peaks) * tsamp_ms if peaks.size >= 2 else np.array([], dtype=float)
    t90 = t90_width_ms(p, tsamp_ms)
    duration_for_rate = t90 if np.isfinite(t90) and t90 > 0 else nt * tsamp_ms

    acf_sigma, acf_fwhm, lag_ms, acf, acf_model = fit_acf_timescale_ms(
        p, tsamp_ms=tsamp_ms, fit_max_lag_ms=acf_fit_max_lag_ms
    )
    if np.all(np.isfinite(acf_model)):
        resid_region = (lag_ms > 0) & (lag_ms <= acf_fit_max_lag_ms) & np.isfinite(acf)
        acf_resid_rms = float(np.sqrt(np.nanmean((acf[resid_region] - acf_model[resid_region]) ** 2)))
    else:
        acf_resid_rms = np.nan

    return {
        "burst_id": burst_id,
        "input_n_peaks": np.nan if input_n_peaks is None else int(input_n_peaks),
        "input_shot_sigma_ms": np.nan if shot_sigma_ms is None else float(shot_sigma_ms),
        "acf_sigma_ms": acf_sigma,
        "acf_fwhm_ms": acf_fwhm,
        "acf_resid_rms": acf_resid_rms,
        "peak_count": int(peaks.size),
        "peak_rate_per_ms": float(peaks.size / duration_for_rate) if duration_for_rate > 0 else np.nan,
        "t90_ms": t90,
        "median_peak_width_ms": float(np.nanmedian(widths_ms)) if widths_ms.size else np.nan,
        "mean_peak_width_ms": float(np.nanmean(widths_ms)) if widths_ms.size else np.nan,
        "median_sep_ms": float(np.nanmedian(sep_ms)) if sep_ms.size else np.nan,
        "mean_sep_ms": float(np.nanmean(sep_ms)) if sep_ms.size else np.nan,
        "sep_cv": float(np.nanstd(sep_ms) / np.nanmean(sep_ms)) if sep_ms.size and np.nanmean(sep_ms) > 0 else np.nan,
        "amplitude_entropy": shannon_entropy_fraction(heights),
        "amplitude_gini": gini_coefficient(heights),
        "profile_modulation_index": float(np.nanstd(positive_part(p)) / np.nanmean(positive_part(p)))
        if np.nanmean(positive_part(p)) > 0
        else np.nan,
        "psd_slope": estimate_psd_slope(p, tsamp_ms=tsamp_ms),
        "peak_threshold_sigma": peak_sigma_threshold,
        "min_peak_distance_ms": min_peak_distance_ms,
    }


def analyse_burst_bank(
    bursts: Sequence[SimulatedBurst],
    peak_sigma_threshold: float = 5.0,
    min_peak_distance_ms: Optional[float] = None,
    acf_fit_max_lag_ms: float = 2.5,
) -> pd.DataFrame:
    """Analyse a list of simulated bursts and return a metrics table."""
    rows = []
    for b in bursts:
        row = analyse_profile(
            b.profile,
            tsamp_ms=b.instrument.tsamp_ms,
            burst_id=b.burst_id,
            input_n_peaks=b.input_n_peaks,
            shot_sigma_ms=b.shot_sigma_ms,
            peak_sigma_threshold=peak_sigma_threshold,
            min_peak_distance_ms=min_peak_distance_ms,
            acf_fit_max_lag_ms=acf_fit_max_lag_ms,
        )
        row.update(
            {
                "complexity_mode": b.forest.complexity_mode,
                "used_will": b.used_will,
                "will_method": b.will_method,
                "duration_ms": b.instrument.duration_ms,
                "tsamp_ms": b.instrument.tsamp_ms,
                "dm": b.instrument.dm,
                "nfreq": b.instrument.nfreq,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_population_complexity_score(
    metrics: pd.DataFrame,
    cols: Sequence[str] = ("peak_count", "peak_rate_per_ms", "acf_resid_rms", "profile_modulation_index"),
) -> pd.DataFrame:
    """
    Add a simple standardized complexity score to a metrics table.

    This score is not a physical observable. It is a convenient ranking index
    for selecting contrasting examples from the same bank.
    """
    out = metrics.copy()
    zsum = np.zeros(len(out), dtype=float)
    n_used = 0
    for col in cols:
        if col not in out:
            continue
        x = out[col].astype(float).to_numpy()
        med = np.nanmedian(x)
        sig = robust_sigma(x)
        if not np.isfinite(sig) or sig <= 0:
            continue
        zsum += np.nan_to_num((x - med) / sig, nan=0.0)
        n_used += 1
    out["complexity_score"] = zsum / max(1, n_used)
    return out


def select_similar_timescale_different_complexity(
    metrics: pd.DataFrame,
    tau_col: str = "acf_sigma_ms",
    tolerance_fraction: float = 0.20,
    complexity_col: str = "complexity_score",
    min_group_size: int = 3,
) -> pd.DataFrame:
    """
    Select bursts with similar measured ACF timescale but different complexity.

    For each possible seed burst, this finds all bursts with ACF sigma within
    `tolerance_fraction` and reports the group whose complexity range is largest.
    """
    df = metrics.copy()
    if complexity_col not in df.columns:
        df = add_population_complexity_score(df)
    df = df[np.isfinite(df[tau_col]) & np.isfinite(df[complexity_col])].copy()
    if df.empty:
        return df

    best = None
    best_range = -np.inf
    for _, row in df.iterrows():
        tau = row[tau_col]
        lo = tau * (1 - tolerance_fraction)
        hi = tau * (1 + tolerance_fraction)
        group = df[(df[tau_col] >= lo) & (df[tau_col] <= hi)].copy()
        if len(group) < min_group_size:
            continue
        crange = group[complexity_col].max() - group[complexity_col].min()
        if crange > best_range:
            best = group
            best_range = crange

    if best is None:
        return df.iloc[0:0].copy()
    return best.sort_values(complexity_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FLITS integration helpers
# ---------------------------------------------------------------------------


def analyse_flits_profile(
    profile: ArrayLike,
    tsamp_ms: float,
    burst_id: str = "flits_burst",
    peak_sigma_threshold: float = 5.0,
    min_peak_distance_ms: Optional[float] = None,
    acf_fit_max_lag_ms: float = 2.5,
) -> Dict[str, float]:
    """
    Run the same metrics on a 1D profile produced by FLITS.

    Use this when FLITS has already dedispersed and collapsed the dynamic
    spectrum for you.
    """
    return analyse_profile(
        profile,
        tsamp_ms=tsamp_ms,
        burst_id=burst_id,
        peak_sigma_threshold=peak_sigma_threshold,
        min_peak_distance_ms=min_peak_distance_ms,
        acf_fit_max_lag_ms=acf_fit_max_lag_ms,
    )


def analyse_flits_dynamic_spectrum(
    dynamic_spectrum: np.ndarray,
    instrument: InstrumentConfig,
    burst_id: str = "flits_burst",
    input_axis_order: Literal["time_frequency", "frequency_time"] = "time_frequency",
    dedisperse: bool = False,
    peak_sigma_threshold: float = 5.0,
    min_peak_distance_ms: Optional[float] = None,
    acf_fit_max_lag_ms: float = 2.5,
) -> Dict[str, float]:
    """
    Run metrics on a FLITS dynamic spectrum.

    If FLITS has already dedispersed the dynamic spectrum, set `dedisperse=False`.
    If you are giving a raw dispersed dynamic spectrum, set `dedisperse=True`.
    """
    ds = np.asarray(dynamic_spectrum, dtype=float)
    if input_axis_order == "frequency_time":
        ds = ds.T
    prof = profile_from_dynamic_spectrum(
        ds,
        instrument=instrument,
        dedisperse=dedisperse,
        use_will_dedispersion=False,
        normalise=True,
    )
    return analyse_profile(
        prof,
        tsamp_ms=instrument.tsamp_ms,
        burst_id=burst_id,
        peak_sigma_threshold=peak_sigma_threshold,
        min_peak_distance_ms=min_peak_distance_ms,
        acf_fit_max_lag_ms=acf_fit_max_lag_ms,
    )


def save_burst_npz(burst: SimulatedBurst, path: Union[str, Path]) -> None:
    """Save a simulated burst in a FLITS-friendly npz container."""
    path = Path(path)
    np.savez_compressed(
        path,
        burst_id=burst.burst_id,
        t_ms=burst.t_ms,
        dynamic_spectrum=burst.dynamic_spectrum,
        profile=burst.profile,
        shot_times_ms=burst.shot_times_ms,
        amplitudes=burst.amplitudes,
        center_freqs_mhz=burst.center_freqs_mhz,
        chan_freqs_mhz=burst.instrument.chan_freqs_mhz,
        tsamp_ms=burst.instrument.tsamp_ms,
        dm=burst.instrument.dm,
        used_will=burst.used_will,
        will_method=burst.will_method,
    )


def load_profile_from_npz(path: Union[str, Path], profile_key: str = "profile") -> Tuple[np.ndarray, float]:
    """Load a profile and tsamp_ms from an npz product."""
    data = np.load(path)
    if profile_key not in data:
        raise KeyError(f"{profile_key!r} not found in {path}")
    tsamp_ms = float(data["tsamp_ms"]) if "tsamp_ms" in data else np.nan
    return np.asarray(data[profile_key], dtype=float), tsamp_ms


# ---------------------------------------------------------------------------
# Optional plotting helpers
# ---------------------------------------------------------------------------


def plot_burst_summary(burst: SimulatedBurst, ax_profile=None, ax_ds=None):
    """Plot dynamic spectrum and dedispersed profile for one burst."""
    import matplotlib.pyplot as plt

    if ax_ds is None or ax_profile is None:
        fig, (ax_profile, ax_ds) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    else:
        fig = ax_profile.figure

    ax_profile.plot(burst.t_ms, burst.profile)
    ax_profile.set_title(f"{burst.burst_id}: profile")
    ax_profile.set_xlabel("Time [ms]")
    ax_profile.set_ylabel("S/N-like intensity")

    extent = [burst.t_ms[0], burst.t_ms[-1], burst.instrument.freq_low_mhz, burst.instrument.freq_high_mhz]
    im = ax_ds.imshow(
        burst.dynamic_spectrum.T,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    ax_ds.set_title(f"dynamic spectrum; WILL={burst.used_will}, method={burst.will_method}")
    ax_ds.set_xlabel("Time [ms]")
    ax_ds.set_ylabel("Frequency [MHz]")
    fig.colorbar(im, ax=ax_ds, label="Intensity")
    fig.tight_layout()
    return fig


def plot_profile_acf(profile: ArrayLike, tsamp_ms: float, title: str = "Profile + ACF"):
    """Plot a profile and its fitted ACF."""
    import matplotlib.pyplot as plt

    p = normalise_profile(profile)
    t = np.arange(p.size) * tsamp_ms
    sigma, fwhm, lag_ms, acf, model = fit_acf_timescale_ms(p, tsamp_ms)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(t, p)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("S/N-like intensity")
    axes[0].set_title(title)

    axes[1].plot(lag_ms, acf, label="ACF")
    if np.all(np.isfinite(model)):
        axes[1].plot(lag_ms, model, label=f"Gaussian fit: sigma={sigma:.3f} ms")
    axes[1].set_xlabel("Lag [ms]")
    axes[1].set_ylabel("ACF")
    axes[1].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Small command-line demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    instrument = InstrumentConfig(duration_ms=12.0, tsamp_ms=0.016, dm=220.0)
    bursts = make_microshot_bank(
        n_peaks_grid=(3, 6, 12, 24),
        complexity_modes=("sparse", "dense", "clustered"),
        n_realizations=3,
        instrument=instrument,
        shot_sigma_ms=0.080,
        prefer_will=True,
        seed=123,
    )
    metrics = analyse_burst_bank(bursts, peak_sigma_threshold=5.0)
    metrics = add_population_complexity_score(metrics)
    selected = select_similar_timescale_different_complexity(metrics)
    print("WILL available:", WILL_AVAILABLE)
    print(metrics.head())
    print("\nSelected similar-timescale group:")
    print(selected[["burst_id", "acf_sigma_ms", "peak_count", "complexity_score"]])
