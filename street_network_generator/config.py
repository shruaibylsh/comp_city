"""
Configuration management for street network generator.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GeneratorConfig:
    """Configuration for street network generation."""

    # Reproducibility
    seed: int = 42

    # Window parameters
    window_size_m: int = 500

    # Performance tuning
    syntax_recompute_interval: int = 80
    candidate_per_step: int = 12

    # Simulated annealing
    initial_temp: float = 5.0
    cooling_rate: float = 0.997

    # Iteration bounds
    min_iterations: int = 250
    max_iterations: int = 2500

    # Geometry constraints
    snap_tolerance_m: float = 1.5
    min_seg_len_m: float = 12.0
    max_seg_len_m: float = 90.0

    # Weight scheduling
    weights: Dict[str, float] = field(default_factory=lambda: {
        "morph_start": 1.0,
        "morph_end": 0.7,
        "syntax_start": 0.0,
        "syntax_end": 0.3,
    })

    # Metric weights (alpha_i for morphology)
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "degree_dist": 0.3,
        "segment_length": 0.3,
        "orientation": 0.15,
        "density": 0.15,
        "dead_end_ratio": 0.1,
    })

    # Syntax weights (beta_j)
    syntax_weights: Dict[str, float] = field(default_factory=lambda: {
        "mean_depth": 0.4,
        "local_integration": 0.3,
        "choice": 0.2,
        "intelligibility": 0.1,
    })

    # Penalty weights
    penalty_weights: Dict[str, float] = field(default_factory=lambda: {
        "disconnected": 10.0,
        "small_component": 5.0,
        "boundary_dead_end": 2.0,
    })

    # Stopping criteria
    morph_divergence_threshold: float = 0.10  # 10%
    no_improvement_audits: int = 5

    # Boundary constraints
    min_boundary_spines: int = 2
    max_boundary_spines: int = 4
    boundary_dead_end_distance_m: float = 30.0

    @classmethod
    def from_json(cls, filepath: str) -> "GeneratorConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        data = {
            "seed": self.seed,
            "window_size_m": self.window_size_m,
            "syntax_recompute_interval": self.syntax_recompute_interval,
            "candidate_per_step": self.candidate_per_step,
            "initial_temp": self.initial_temp,
            "cooling_rate": self.cooling_rate,
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "snap_tolerance_m": self.snap_tolerance_m,
            "min_seg_len_m": self.min_seg_len_m,
            "max_seg_len_m": self.max_seg_len_m,
            "weights": self.weights,
            "metric_weights": self.metric_weights,
            "syntax_weights": self.syntax_weights,
            "penalty_weights": self.penalty_weights,
            "morph_divergence_threshold": self.morph_divergence_threshold,
            "no_improvement_audits": self.no_improvement_audits,
            "min_boundary_spines": self.min_boundary_spines,
            "max_boundary_spines": self.max_boundary_spines,
            "boundary_dead_end_distance_m": self.boundary_dead_end_distance_m,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_weight_at_progress(self, progress: float, weight_type: str) -> float:
        """
        Get interpolated weight based on generation progress.

        Args:
            progress: Generation progress [0, 1]
            weight_type: 'morph' or 'syntax'

        Returns:
            Interpolated weight value
        """
        if weight_type == "morph":
            start = self.weights["morph_start"]
            end = self.weights["morph_end"]
        elif weight_type == "syntax":
            start = self.weights["syntax_start"]
            end = self.weights["syntax_end"]
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        # Linear interpolation with schedule:
        # First 60%: start weight
        # Next 20%: ramp start -> mid
        # Final 20%: ramp mid -> end
        if progress < 0.6:
            return start
        elif progress < 0.8:
            # Ramp from start to midpoint
            local_progress = (progress - 0.6) / 0.2
            mid = (start + end) / 2
            return start + local_progress * (mid - start)
        else:
            # Ramp from midpoint to end
            local_progress = (progress - 0.8) / 0.2
            mid = (start + end) / 2
            return mid + local_progress * (end - mid)
