"""
drawing_placement.py

Separates behavioural tracking position from drawing position.

Horizontal placement retains the visitor's relative left/right
position. Vertical placement is selected from five fixed anchors.

The selection is random but constrained:
- five anchors remain evenly distributed across the drawable area;
- a mild centre preference protects overall composition;
- the most recently used anchor receives a temporary penalty;
- one lane is cached for each session/cycle, so placement does not
  change from frame to frame during the same interpretation cycle.
"""

import random


class DrawingPlacement:

    def __init__(
        self,
        horizontal_margin_ratio=0.12,
        vertical_lanes=(
            0.18,
            0.34,
            0.50,
            0.66,
            0.82
        ),
        lane_weights=(
            0.16,
            0.22,
            0.24,
            0.22,
            0.16
        ),
        recent_lane_penalty=0.25
    ):
        self.horizontal_margin_ratio = float(
            horizontal_margin_ratio
        )

        self.vertical_lanes = tuple(
            float(value)
            for value in vertical_lanes
        )

        self.lane_weights = tuple(
            float(value)
            for value in lane_weights
        )

        if (
            len(self.vertical_lanes)
            != len(self.lane_weights)
        ):
            raise ValueError(
                "vertical_lanes and lane_weights "
                "must have the same length."
            )

        self.recent_lane_penalty = max(
            0.0,
            min(float(recent_lane_penalty), 1.0)
        )

        # One stable lane per (session, completed cycle).
        self.assigned_lanes = {}

        # Global recent lane history for the current sheet.
        self.last_lane_index = None

    def place(
        self,
        intent,
        session_id,
        frame_shape,
        cycle_index=0
    ):
        if intent is None:
            return None

        frame_height, frame_width = frame_shape[:2]

        source_x = float(
            intent["origin"][0]
        )

        normalised_x = source_x / max(
            1.0,
            frame_width
        )

        margin = self.horizontal_margin_ratio

        placed_x_ratio = (
            margin
            + normalised_x
            * (1.0 - 2.0 * margin)
        )

        placement_key = (
            int(session_id),
            int(cycle_index)
        )

        lane_index = self.assigned_lanes.get(
            placement_key
        )

        if lane_index is None:
            lane_index = self._choose_lane()
            self.assigned_lanes[
                placement_key
            ] = lane_index
            self.last_lane_index = lane_index

            print(
                "Drawing lane assigned |",
                "session_id:",
                session_id,
                "| cycle:",
                cycle_index,
                "| lane:",
                lane_index,
                "| y ratio:",
                self.vertical_lanes[lane_index]
            )

        placed_y_ratio = self.vertical_lanes[
            lane_index
        ]

        placed = intent.copy()
        placed["source_origin"] = intent["origin"]
        placed["origin"] = (
            int(round(
                placed_x_ratio * frame_width
            )),
            int(round(
                placed_y_ratio * frame_height
            ))
        )
        placed["placement_lane"] = lane_index
        placed["placement_cycle_index"] = int(
            cycle_index
        )

        return placed

    def reset_sheet(self):
        """
        Clears all lane assignments for a replacement sheet.

        The next interpretation begins with a fresh constrained
        random choice rather than repeating the previous layout.
        """

        self.assigned_lanes.clear()
        self.last_lane_index = None

        print("Drawing placement reset for new sheet.")

    def _choose_lane(self):
        weights = list(
            self.lane_weights
        )

        if self.last_lane_index is not None:
            weights[
                self.last_lane_index
            ] *= self.recent_lane_penalty

        # Protect against an invalid all-zero configuration.
        if sum(weights) <= 0:
            weights = [
                1.0
                for _ in self.vertical_lanes
            ]

        return random.choices(
            population=range(
                len(self.vertical_lanes)
            ),
            weights=weights,
            k=1
        )[0]