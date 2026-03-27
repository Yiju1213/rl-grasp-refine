from __future__ import annotations

import argparse
import json

import numpy as np

from _common import load_experiment_bundle
from src.calibration.online_logit_calibrator import OnlineLogitCalibrator


def main():
    parser = argparse.ArgumentParser(description="Debug calibrator predict/update behavior.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    args = parser.parse_args()

    _, config_bundle = load_experiment_bundle(args.experiment)
    calibrator_cfg = config_bundle["calibration"]
    calibrator = OnlineLogitCalibrator(calibrator_cfg)

    logits = np.asarray([-1.0, -0.3, 0.0, 0.5, 1.2], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.float32)
    before = [calibrator.predict(value) for value in logits]
    calibrator.update(logits, labels)
    after = [calibrator.predict(value) for value in logits]
    payload = {
        "before": before,
        "after": after,
        "state": {
            "a": calibrator.a,
            "b": calibrator.b,
            "posterior_trace": calibrator.posterior_trace(),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
