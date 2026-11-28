import numpy as np


class GeoAgent:
    """Needs columns loc_dev_km and device_mismatch (from the synthetic generator)."""

    name = "geo"

    def __init__(self, cfg):
        a = cfg["agents"]["geo"]
        self.enabled = a["enabled"]
        self.alpha = a["alpha"]
        self.norm_km = a["location_norm_km"]

    @staticmethod
    def combine(delta_location, delta_device, alpha):
        d_loc = np.asarray(delta_location, float)
        d_dev = np.asarray(delta_device, float)
        return np.clip(alpha * d_loc + (1 - alpha) * d_dev, 0.0, 1.0)

    def score(self, df):
        if not self.enabled:
            return np.zeros(len(df))
        loc = np.clip(df["loc_dev_km"].to_numpy(float) / self.norm_km, 0.0, 1.0)
        dev = df["device_mismatch"].to_numpy(float)
        return self.combine(loc, dev, self.alpha)