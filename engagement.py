"""
engagement.py

Classifies each visible visitor as foreground or background.

Background visitors remain valid evidence sources and may activate
the system when no foreground visitor is present. Foreground visitors
always have higher arbitration priority.
"""


class Engagement:

    def __init__(
        self,
        foreground_box_height_ratio=0.48,
        foreground_bottom_ratio=0.92,
        background_weight=0.35,
        foreground_weight=1.0
    ):
        self.foreground_box_height_ratio = float(
            foreground_box_height_ratio
        )
        self.foreground_bottom_ratio = float(
            foreground_bottom_ratio
        )
        self.background_weight = float(
            background_weight
        )
        self.foreground_weight = float(
            foreground_weight
        )

    def classify(
        self,
        observation,
        representation,
        frame_shape
    ):
        frame_height, frame_width = frame_shape[:2]

        x1, y1, x2, y2 = observation[
            "bounding_box"
        ]

        box_height_ratio = max(
            0.0,
            float(y2 - y1) / max(1.0, frame_height)
        )

        bottom_ratio = float(y2) / max(
            1.0,
            frame_height
        )

        # A visitor is treated as foreground when their body occupies
        # enough of the image or reaches the near part of the scene.
        foreground = (
            box_height_ratio
            >= self.foreground_box_height_ratio
            or bottom_ratio
            >= self.foreground_bottom_ratio
        )

        level = (
            "foreground"
            if foreground
            else "background"
        )

        weight = (
            self.foreground_weight
            if foreground
            else self.background_weight
        )

        return {
            "level": level,
            "weight": weight,
            "box_height_ratio": box_height_ratio,
            "bottom_ratio": bottom_ratio,
            "eligible": True
        }