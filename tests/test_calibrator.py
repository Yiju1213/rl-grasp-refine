from __future__ import annotations

import unittest

import numpy as np

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from tests.fakes import make_calibration_cfg


class TestCalibrator(unittest.TestCase):
    def test_predict_and_update(self):
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())
        prob_before = calibrator.predict(0.5)
        trace_before = calibrator.posterior_trace()
        calibrator.update(np.asarray([-1.0, 0.0, 1.0]), np.asarray([0, 0, 1]))
        prob_after = calibrator.predict(0.5)
        trace_after = calibrator.posterior_trace()
        self.assertGreaterEqual(prob_before, 0.0)
        self.assertGreaterEqual(trace_before, 0.0)
        self.assertNotEqual(calibrator.a, 1.0)
        self.assertGreaterEqual(prob_after, 0.0)
        self.assertGreaterEqual(trace_after, 0.0)
        self.assertNotEqual(trace_before, trace_after)

    def test_state_roundtrip_restores_predictions(self):
        calibrator = OnlineLogitCalibrator(make_calibration_cfg())
        calibrator.update(np.asarray([-1.0, 0.0, 1.0]), np.asarray([0, 0, 1]))
        state = calibrator.get_state()
        restored = OnlineLogitCalibrator(make_calibration_cfg())
        restored.load_state(state)

        prob_a = calibrator.predict(0.5)
        prob_b = restored.predict(0.5)

        self.assertAlmostEqual(prob_a, prob_b, places=7)
        self.assertAlmostEqual(calibrator.posterior_trace(), restored.posterior_trace(), places=7)

    def test_update_can_be_disabled_without_changing_state(self):
        calibration_cfg = make_calibration_cfg()
        calibration_cfg["online_update_enabled"] = False
        calibrator = OnlineLogitCalibrator(calibration_cfg)
        state_before = calibrator.get_state()

        calibrator.update(np.asarray([-1.0, 0.0, 1.0]), np.asarray([0, 0, 1]))
        state_after = calibrator.get_state()

        self.assertEqual(float(state_before["a"]), float(state_after["a"]))
        self.assertEqual(float(state_before["b"]), float(state_after["b"]))
        np.testing.assert_allclose(state_before["posterior_cov"], state_after["posterior_cov"])

    def test_identity_probability_mode_uses_sigmoid_without_uncertainty(self):
        calibration_cfg = make_calibration_cfg()
        calibration_cfg["online_update_enabled"] = False
        calibration_cfg["signal_mode"] = "identity_probability"
        calibration_cfg["uncertainty_discount_enabled"] = False
        calibrator = OnlineLogitCalibrator(calibration_cfg)

        self.assertAlmostEqual(calibrator.predict(0.0), 0.5)
        np.testing.assert_allclose(
            calibrator.predict(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)),
            1.0 / (1.0 + np.exp(-np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(calibrator.posterior_trace(), 0.0)


if __name__ == "__main__":
    unittest.main()
