"""Independent calibration utilities for compact-group scheduling."""

import math


RESOURCE_KEYS = (
    "input_bw",
    "weight_bw",
    "output_bw",
    "dram_input_bw",
    "dram_weight_bw",
    "dram_output_bw",
)


def build_calibration_scales(calibration_profiles):
    """Build fixed normalization scales from calibration profiles only.

    The caller is responsible for separating calibration data from evaluation
    data before calling this function.
    """
    profiles = list(calibration_profiles)
    if not profiles:
        raise ValueError("Calibration profile collection is empty")

    scales = {}
    for key in RESOURCE_KEYS:
        values = []
        for index, profile in enumerate(profiles):
            try:
                raw_value = (
                    profile[key] if isinstance(profile, dict)
                    else getattr(profile, key)
                )
            except (KeyError, AttributeError) as exc:
                raise ValueError(
                    f"Calibration profile {index} is missing {key}"
                ) from exc

            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid calibration value for {key} at profile {index}: {value}"
                )
            values.append(value)

        maximum = max(values)
        if maximum <= 0:
            raise ValueError(
                f"Calibration maximum must be positive for {key}"
            )
        scales[key] = maximum

    return scales
