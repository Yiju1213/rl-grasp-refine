from __future__ import annotations

import unittest

import numpy as np

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from tests.fakes import make_calibration_cfg


class TestCalibrator(unittest.TestCase):
    def test_predict_and_update(self):
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())
        prob_before, uncertainty_before = calibrator.predict(0.5)
        calibrator.update(np.asarray([-1.0, 0.0, 1.0]), np.asarray([0, 0, 1]))
        prob_after, uncertainty_after = calibrator.predict(0.5)
        self.assertGreaterEqual(prob_before, 0.0)
        self.assertGreaterEqual(uncertainty_before, 0.0)
        self.assertNotEqual(calibrator.a, 1.0)
        self.assertGreaterEqual(prob_after, 0.0)
        self.assertGreaterEqual(uncertainty_after, 0.0)

    def test_state_roundtrip_restores_predictions(self):
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())
        calibrator.update(np.asarray([-1.0, 0.0, 1.0]), np.asarray([0, 0, 1]))
        state = calibrator.get_state()
        restored = OnlineLogitCalibrator(make_calibration_cfg())
        restored.load_state(state)

        prob_a, uncertainty_a = calibrator.predict(0.5)
        prob_b, uncertainty_b = restored.predict(0.5)

        self.assertAlmostEqual(prob_a, prob_b, places=7)
        self.assertAlmostEqual(uncertainty_a, uncertainty_b, places=7)


if __name__ == "__main__":
    unittest.main()
