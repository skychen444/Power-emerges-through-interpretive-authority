"""
interpretation.py

Interpretation Intent v2.3

Interpretation translates Judgement into an interpretive intent.

Direction is derived from the principal axis of the recent selected
trajectory rather than only from its first and last point.
"""

import math

import numpy as np


class Interpretation:

    def __init__(
        self,
        trajectory_window=30,
        minimum_direction_spread=12.0
    ):
        self.trajectory_window = trajectory_window
        self.minimum_direction_spread = (
            minimum_direction_spread
        )

        self.rewrite_grammar_map = {
            "dwell": "overwrite",
            "revisit": "cross_out",
            "density": "compression",
            "occupancy": "enclosure",
            "trajectory_instability": "correction"
        }

    def update(
        self,
        selected_evidence,
        judgement
    ):
        if not judgement.has_interpretive_authority():
            return None

        stage = judgement.get_stage()

        if stage is None:
            return None

        if not selected_evidence:
            return None

        origin = selected_evidence[-1]["position"]

        direction = self._remembered_motion_direction(
            selected_evidence
        )

        if direction is None:
            direction = self._trajectory_angle(
                selected_evidence,
                last_n=self.trajectory_window
            )

        dominant_component = (
            judgement.get_dominant_component()
        )

        grammar = self._grammar_from_judgement(
            stage=stage,
            dominant_component=dominant_component
        )

        return {
            "type": f"{stage}_intent",
            "stage": stage,
            "grammar": grammar,
            "confidence": judgement.get_belief(),
            "origin": origin,
            "direction": direction,
            "evidence_count": len(selected_evidence),
            "components": judgement.get_components(),
            "dominant_component": dominant_component
        }

    def _grammar_from_judgement(
        self,
        stage,
        dominant_component
    ):
        if stage == "probe":
            return "probe"

        if stage == "commit":
            return "commit"

        if stage != "rewrite":
            return None

        if dominant_component is None:
            return "overwrite"

        component_name = dominant_component.get(
            "name"
        )

        return self.rewrite_grammar_map.get(
            component_name,
            "overwrite"
        )

    def _remembered_motion_direction(
        self,
        selected_evidence
    ):
        """
        Reads the latest reliable motion direction preserved by
        Evidence.

        Selected evidence may consist mostly of stationary observations,
        but each accepted item can still carry a recent motion memory.
        """

        for item in reversed(selected_evidence):
            direction = item.get(
                "motion_direction"
            )

            if direction is None:
                continue

            age = item.get(
                "motion_direction_age"
            )

            if age is not None and age > 6.0:
                continue

            return float(direction)

        return None

    def _trajectory_angle(
        self,
        selected_evidence,
        last_n=30
    ):
        """
        Returns the principal direction of the recent selected
        trajectory.

        PCA preserves the dominant spatial axis even when a visitor
        moves back and forth and finishes near the starting point.

        Returns None when there is no reliable directional spread.
        """

        if len(selected_evidence) < 3:
            return None

        sample_size = min(
            len(selected_evidence),
            last_n
        )

        recent = selected_evidence[
            -sample_size:
        ]

        positions = np.array(
            [
                evidence["position"]
                for evidence in recent
            ],
            dtype=np.float64
        )

        centred = positions - positions.mean(
            axis=0
        )

        covariance = np.cov(
            centred,
            rowvar=False
        )

        if covariance.shape != (2, 2):
            return None

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        principal_index = int(
            np.argmax(eigenvalues)
        )

        principal_value = float(
            eigenvalues[principal_index]
        )

        if principal_value <= 0.0:
            return None

        principal_spread = math.sqrt(
            principal_value
        )

        if (
            principal_spread
            < self.minimum_direction_spread
        ):
            return None

        principal_vector = eigenvectors[
            :,
            principal_index
        ]

        net_displacement = (
            positions[-1] - positions[0]
        )

        if np.linalg.norm(net_displacement) >= 4.0:
            if (
                np.dot(
                    principal_vector,
                    net_displacement
                )
                < 0.0
            ):
                principal_vector = (
                    -principal_vector
                )
        else:
            if (
                principal_vector[0] < 0.0
                or (
                    abs(principal_vector[0]) < 1e-9
                    and principal_vector[1] < 0.0
                )
            ):
                principal_vector = (
                    -principal_vector
                )

        return math.atan2(
            float(principal_vector[1]),
            float(principal_vector[0])
        )