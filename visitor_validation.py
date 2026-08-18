"""Stable candidate validation for multi-person tracking."""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class CandidateState:
    track_id: int
    first_seen: float
    last_seen: float
    observation_count: int = 0
    geometry_count: int = 0
    nose_visible_count: int = 0
    active: bool = False
    activation_reason: Optional[str] = None
    positions: List[Point] = field(default_factory=list)
    position_times: List[float] = field(default_factory=list)
    box_areas: List[float] = field(default_factory=list)


class VisitorValidator:
    def __init__(
        self,
        probation_seconds=0.8,
        minimum_observations=6,
        minimum_geometry_ratio=0.60,
        minimum_motion_pixels=25.0,
        minimum_box_change_ratio=0.12,
        track_timeout_seconds=2.0,
        maximum_history=90,
        minimum_nose_visible_ratio=0.30,
        minimum_motion_observations=12,
        minimum_observation_rate=6.0,
        maximum_motion_box_change_ratio=2.0,
        minimum_motion_step_count=3,
        minimum_motion_step_pixels=2.0,
        maximum_motion_step_pixels=80.0,
        minimum_motion_consistency_ratio=0.25,
        recent_motion_window_seconds=2.0,
        minimum_recent_displacement_pixels=40.0,
    ):
        self.probation_seconds = float(probation_seconds)
        self.minimum_observations = int(minimum_observations)
        self.minimum_geometry_ratio = float(minimum_geometry_ratio)
        self.minimum_motion_pixels = float(minimum_motion_pixels)
        self.minimum_box_change_ratio = float(minimum_box_change_ratio)
        self.track_timeout_seconds = float(track_timeout_seconds)
        self.maximum_history = int(maximum_history)
        self.minimum_nose_visible_ratio = float(minimum_nose_visible_ratio)
        self.minimum_motion_observations = int(minimum_motion_observations)
        self.minimum_observation_rate = float(minimum_observation_rate)
        self.maximum_motion_box_change_ratio = float(maximum_motion_box_change_ratio)
        self.minimum_motion_step_count = int(minimum_motion_step_count)
        self.minimum_motion_step_pixels = float(minimum_motion_step_pixels)
        self.maximum_motion_step_pixels = float(maximum_motion_step_pixels)
        self.minimum_motion_consistency_ratio = float(minimum_motion_consistency_ratio)
        self.recent_motion_window_seconds = float(recent_motion_window_seconds)
        self.minimum_recent_displacement_pixels = float(minimum_recent_displacement_pixels)
        self.candidates: Dict[int, CandidateState] = {}

    def update(self, track_id, representation, bounding_box, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()

        track_id = int(track_id)
        state = self.candidates.get(track_id)

        if state is None:
            state = CandidateState(track_id, timestamp, timestamp)
            self.candidates[track_id] = state

        state.last_seen = timestamp
        state.observation_count += 1

        position = representation.get("position")
        visible = representation.get("visible", {})

        geometry_valid = position is not None and (
            visible.get("shoulders", False)
            or visible.get("hips", False)
        )

        if geometry_valid:
            state.geometry_count += 1

        if visible.get("nose", False):
            state.nose_visible_count += 1

        if position is not None:
            state.positions.append((float(position[0]), float(position[1])))
            state.position_times.append(float(timestamp))

        box_area = self._box_area(bounding_box)
        if box_area > 0:
            state.box_areas.append(box_area)

        self._trim_history(state)
        metrics = self._metrics(state, timestamp)

        if not state.active:
            reason = self._activation_reason(metrics)
            if reason is not None:
                state.active = True
                state.activation_reason = reason
                print(
                    "Visitor activated |",
                    "track_id:", track_id,
                    "| reason:", reason,
                    "| metrics:", metrics,
                )

        return {
            "track_id": track_id,
            "status": "active" if state.active else "candidate",
            "active": state.active,
            "activation_reason": state.activation_reason,
            "metrics": metrics,
        }

    def is_active(self, track_id):
        state = self.candidates.get(int(track_id))
        return bool(state is not None and state.active)

    def mark_frame_complete(self, seen_track_ids, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()

        seen = {int(track_id) for track_id in seen_track_ids}
        expired = []

        for track_id, state in list(self.candidates.items()):
            if track_id in seen:
                continue

            missing_duration = timestamp - state.last_seen
            if missing_duration >= self.track_timeout_seconds:
                expired.append({
                    "track_id": track_id,
                    "was_active": state.active,
                    "missing_duration": missing_duration,
                })
                del self.candidates[track_id]

        return expired

    def get_state(self, track_id):
        return self.candidates.get(int(track_id))

    def _activation_reason(self, metrics):
        if metrics["duration"] < self.probation_seconds:
            return None
        if metrics["observation_count"] < self.minimum_observations:
            return None
        if metrics["geometry_ratio"] < self.minimum_geometry_ratio:
            return None

        if metrics["nose_visible_ratio"] >= self.minimum_nose_visible_ratio:
            return "nose_confirmed"

        stable_body_motion = (
            metrics["maximum_displacement"] >= self.minimum_motion_pixels
            and metrics["observation_count"] >= self.minimum_motion_observations
            and metrics["observation_rate"] >= self.minimum_observation_rate
            and metrics["box_change_ratio"] <= self.maximum_motion_box_change_ratio
            and metrics["credible_motion_step_count"] >= self.minimum_motion_step_count
            and metrics["motion_consistency_ratio"] >= self.minimum_motion_consistency_ratio
            and metrics["recent_displacement"] >= self.minimum_recent_displacement_pixels
        )

        if stable_body_motion:
            return "body_motion"

        return None

    def _metrics(self, state, timestamp):
        duration = max(0.0, timestamp - state.first_seen)
        observation_count = state.observation_count
        geometry_ratio = state.geometry_count / observation_count if observation_count else 0.0
        nose_visible_ratio = state.nose_visible_count / observation_count if observation_count else 0.0
        observation_rate = observation_count / duration if duration > 0 else 0.0
        maximum_displacement = self._maximum_displacement(state.positions)
        box_change_ratio = self._box_change_ratio(state.box_areas)
        steps = self._motion_step_metrics(state.positions)
        recent_displacement = self._recent_displacement(
            state.positions,
            state.position_times,
            timestamp,
        )

        return {
            "duration": round(duration, 3),
            "observation_count": observation_count,
            "observation_rate": round(observation_rate, 2),
            "geometry_ratio": round(geometry_ratio, 3),
            "nose_visible_count": state.nose_visible_count,
            "nose_visible_ratio": round(nose_visible_ratio, 3),
            "maximum_displacement": round(maximum_displacement, 2),
            "box_change_ratio": round(box_change_ratio, 3),
            "credible_motion_step_count": steps["credible_step_count"],
            "motion_consistency_ratio": round(steps["consistency_ratio"], 3),
            "median_step_distance": round(steps["median_step_distance"], 2),
            "recent_displacement": round(recent_displacement, 2),
        }

    def _maximum_displacement(self, positions):
        if len(positions) < 2:
            return 0.0
        origin = positions[0]
        return max(
            math.hypot(point[0] - origin[0], point[1] - origin[1])
            for point in positions[1:]
        )

    def _recent_displacement(self, positions, position_times, timestamp):
        """
        Measures net movement inside the recent time window.

        Static objects may accumulate small pose jitter over many seconds.
        This metric requires a meaningful change between the earliest and
        latest usable positions within the most recent window.
        """
        if len(positions) < 2 or len(position_times) < 2:
            return 0.0

        window_start = float(timestamp) - self.recent_motion_window_seconds

        recent_points = [
            point
            for point, point_time in zip(positions, position_times)
            if point_time >= window_start
        ]

        if len(recent_points) < 2:
            return 0.0

        first = recent_points[0]
        last = recent_points[-1]

        return math.hypot(
            last[0] - first[0],
            last[1] - first[1],
        )

    def _motion_step_metrics(self, positions):
        if len(positions) < 2:
            return {
                "credible_step_count": 0,
                "consistency_ratio": 0.0,
                "median_step_distance": 0.0,
            }

        distances = [
            math.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(positions, positions[1:])
        ]

        credible = [
            distance
            for distance in distances
            if self.minimum_motion_step_pixels <= distance <= self.maximum_motion_step_pixels
        ]

        sorted_distances = sorted(distances)
        middle = len(sorted_distances) // 2
        if len(sorted_distances) % 2:
            median = sorted_distances[middle]
        else:
            median = (sorted_distances[middle - 1] + sorted_distances[middle]) / 2.0

        return {
            "credible_step_count": len(credible),
            "consistency_ratio": len(credible) / len(distances),
            "median_step_distance": median,
        }

    @staticmethod
    def _box_change_ratio(areas):
        if len(areas) < 2:
            return 0.0
        minimum_area = min(areas)
        maximum_area = max(areas)
        if minimum_area <= 0:
            return 0.0
        return (maximum_area - minimum_area) / minimum_area

    @staticmethod
    def _box_area(bounding_box):
        if bounding_box is None:
            return 0.0
        x1, y1, x2, y2 = [float(value) for value in bounding_box]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _trim_history(self, state):
        if len(state.positions) > self.maximum_history:
            state.positions = state.positions[-self.maximum_history:]
            state.position_times = state.position_times[-self.maximum_history:]
        if len(state.box_areas) > self.maximum_history:
            state.box_areas = state.box_areas[-self.maximum_history:]