"""
Proves the statistical drift detection logic works correctly.
"""

import numpy as np
from scipy.stats import ks_2samp

DRIFT_THRESHOLD = 0.05


def test_no_drift_on_identical_distributions():
    """Identical distributions should NOT trigger drift."""
    np.random.seed(42)
    baseline = np.random.normal(5000, 500, 1000)
    recent = np.random.normal(5000, 500, 1000)

    stat, p_value = ks_2samp(baseline, recent)
    assert p_value > DRIFT_THRESHOLD, (
        "False positive: Drift detected on identical distributions!"
    )


def test_drift_detected_on_shifted_distributions():
    """Shifted distributions MUST trigger drift."""
    np.random.seed(42)
    baseline = np.random.normal(5000, 500, 1000)
    # Simulate the 1.5x drift we injected in Phase 5
    recent = np.random.normal(7500, 500, 1000)

    stat, p_value = ks_2samp(baseline, recent)
    assert p_value < DRIFT_THRESHOLD, (
        "False negative: Failed to detect massive distribution shift!"
    )
