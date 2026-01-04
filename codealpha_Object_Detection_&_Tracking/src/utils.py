"""Utility functions for the project."""

import os
import time
from typing import Any
import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with configuration values.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


class FPSCounter:
    """FPS counter with smoothing."""

    def __init__(self, smoothing: float = 0.9):
        """
        Initialize FPS counter.

        Args:
            smoothing: Exponential smoothing factor (0-1).
        """
        self.smoothing = smoothing
        self.fps = 0.0
        self.last_time = time.time()

    def update(self) -> float:
        """
        Update FPS calculation.

        Returns:
            Current smoothed FPS.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps

        return self.fps

    def get_fps(self) -> float:
        """Get current FPS value."""
        return self.fps

    def reset(self):
        """Reset FPS counter."""
        self.fps = 0.0
        self.last_time = time.time()
