"""
spatial.py

Finds where selected evidence becomes spatially dense.

This module does not judge.
It only describes where evidence accumulates.
"""

import math


class SpatialAnalysis:

    def __init__(self):
        self.radius = 100
        self.minimum_points = 30

    def find_dense_area(self, history):
        if len(history) < self.minimum_points:
            return None

        best_center = None
        best_count = 0

        for item in history:
            center = item["position"]
            count = 0

            for other in history:
                if self._distance(center, other["position"]) <= self.radius:
                    count += 1

            if count > best_count:
                best_count = count
                best_center = center

        density = best_count / len(history)

        return {
            "center": best_center,
            "radius": self.radius,
            "count": best_count,
            "density": density
        }

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )