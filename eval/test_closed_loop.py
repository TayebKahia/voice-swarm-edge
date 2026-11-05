"""eval/closed_loop.py: onset rule, sample clock, rest detection."""

import numpy as np

from eval.closed_loop import FRAME, RATE, find_onset, rest_index, sample_clock


def _speech(n, rng, level=3000):
    return (rng.standard_normal(n) * level).astype(np.int16)


def test_onset_bridges_the_comma_and_stops_at_silence():
    rng = np.random.default_rng(1)
    silence = lambda s: (rng.standard_normal(int(s * RATE)) * 20).astype(np.int16)
    audio = np.concatenate([
        silence(2.0), _speech(int(0.8 * RATE), rng),     # an earlier command
        silence(1.5),
        _speech(int(0.35 * RATE), rng), silence(0.2),     # "swarm," pause
        _speech(int(0.4 * RATE), rng), silence(0.3),      # "hold", then the trigger
    ])
    onset_true = int((2.0 + 0.8 + 1.5) * RATE)
    trigger = audio.size - 1
    onset, region = find_onset(audio, trigger)
    assert abs(onset - onset_true) <= 160
    assert 1.1 < region < 1.4


def test_sample_clock_sits_on_the_lower_envelope():
    rng = np.random.default_rng(2)
    n = 500
    true_end = 100.0 + ((np.arange(n) + 1) * FRAME - 1) / RATE
    tdue = true_end + 0.002 + rng.exponential(0.01, n)
    t0, slope = sample_clock(tdue)
    est = t0 + slope * ((np.arange(n) + 1) * FRAME - 1)
    assert np.all(est <= tdue + 1e-9)
    assert np.abs(est - true_end).max() < 0.01


def test_rest_needs_half_a_second_still():
    tick = 1 / 24
    s = np.concatenate([np.full(10, 2.0), np.full(5, 0.05), np.full(3, 0.2), np.full(30, 0.05)])
    speeds = np.column_stack([s] * 5)
    assert rest_index(speeds, 0, tick) == 18
