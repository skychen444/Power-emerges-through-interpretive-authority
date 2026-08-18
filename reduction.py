"""
reduction.py

The system reduces one validated YOLO body observation
into one selected point.

Observation establishes that a visitor session is credible.
Reduction decides which part of that body represents the person.
"""


class BodyCenterReduction:
    def __init__(self):
        self.bias = {
            "hip_center": 1.0,
            "nose": 0.0,
            "shoulder_center": 0.0
        }

        self.keypoint_confidence_threshold = 0.35
        self.minimum_selected_confidence = 0.35

        # Relative to the detected person's bounding box.
        self.minimum_shoulder_width_ratio = 0.10
        self.minimum_hip_width_ratio = 0.08
        self.minimum_torso_height_ratio = 0.10

        self.minimum_width_ratio = 0.20
        self.maximum_width_ratio = 5.0

        self.coordinate_margin_pixels = 40.0

    def get_position(
        self,
        observation,
        frame_shape
    ):
        if observation is None:
            return None

        keypoints = observation.get(
            "keypoints"
        )

        bounding_box = observation.get(
            "bounding_box"
        )

        if (
            keypoints is None
            or bounding_box is None
        ):
            return None

        if not self._valid_body(
            keypoints,
            bounding_box,
            frame_shape
        ):
            return None

        candidates = {
            "nose": self._single_candidate(
                keypoints[
                    "nose"
                ]
            ),

            "hip_center": self._center_candidate(
                keypoints[
                    "left_hip"
                ],
                keypoints[
                    "right_hip"
                ]
            ),

            "shoulder_center": (
                self._center_candidate(
                    keypoints[
                        "left_shoulder"
                    ],
                    keypoints[
                        "right_shoulder"
                    ]
                )
            )
        }

        selected_method = (
            self._select_candidate(
                candidates
            )
        )

        if selected_method is None:
            return None

        selected = candidates[
            selected_method
        ]

        if (
            selected is None
            or selected[
                "confidence"
            ]
            < self.minimum_selected_confidence
        ):
            return None

        return {
            "id": observation[
                "session_id"
            ],
            "session_id": observation[
                "session_id"
            ],
            "track_id": observation[
                "track_id"
            ],
            "time": observation[
                "time"
            ],
            "position": selected[
                "position"
            ],
            "confidence": selected[
                "confidence"
            ],
            "method": selected_method
        }

    def _valid_body(
        self,
        keypoints,
        bounding_box,
        frame_shape
    ):
        required_names = (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip"
        )

        for name in required_names:
            if name not in keypoints:
                return False

        required = [
            keypoints[
                name
            ]
            for name
            in required_names
        ]

        valid_count = sum(
            self._usable(
                data
            )
            for data
            in required
        )

        if valid_count < 3:
            return False

        # A hip-centred reduction requires both hips.
        if (
            not self._usable(
                keypoints[
                    "left_hip"
                ]
            )
            or not self._usable(
                keypoints[
                    "right_hip"
                ]
            )
        ):
            return False

        for data in required:
            if (
                self._usable(
                    data
                )
                and not self._coordinate_is_reasonable(
                    data[
                        "position"
                    ],
                    frame_shape
                )
            ):
                return False

        x1, y1, x2, y2 = [
            float(
                value
            )
            for value
            in bounding_box
        ]

        box_width = max(
            1.0,
            x2 - x1
        )

        box_height = max(
            1.0,
            y2 - y1
        )

        shoulders_available = (
            self._usable(
                keypoints[
                    "left_shoulder"
                ]
            )
            and self._usable(
                keypoints[
                    "right_shoulder"
                ]
            )
        )

        if shoulders_available:
            left_shoulder = keypoints[
                "left_shoulder"
            ][
                "position"
            ]

            right_shoulder = keypoints[
                "right_shoulder"
            ][
                "position"
            ]

            left_hip = keypoints[
                "left_hip"
            ][
                "position"
            ]

            right_hip = keypoints[
                "right_hip"
            ][
                "position"
            ]

            shoulder_width = abs(
                left_shoulder[0]
                - right_shoulder[0]
            )

            hip_width = abs(
                left_hip[0]
                - right_hip[0]
            )

            shoulder_center_y = (
                left_shoulder[1]
                + right_shoulder[1]
            ) / 2.0

            hip_center_y = (
                left_hip[1]
                + right_hip[1]
            ) / 2.0

            torso_height = (
                hip_center_y
                - shoulder_center_y
            )

            if (
                shoulder_width
                < box_width
                * self.minimum_shoulder_width_ratio
            ):
                return False

            if (
                hip_width
                < box_width
                * self.minimum_hip_width_ratio
            ):
                return False

            if (
                torso_height
                < box_height
                * self.minimum_torso_height_ratio
            ):
                return False

            width_ratio = (
                shoulder_width
                / max(
                    hip_width,
                    1.0
                )
            )

            if (
                width_ratio
                < self.minimum_width_ratio
                or width_ratio
                > self.maximum_width_ratio
            ):
                return False

        return True

    def _usable(
        self,
        data
    ):
        return (
            data is not None
            and data.get(
                "valid",
                False
            )
            and data.get(
                "position"
            )
            is not None
            and float(
                data.get(
                    "confidence",
                    0.0
                )
            )
            >= self.keypoint_confidence_threshold
        )

    def _coordinate_is_reasonable(
        self,
        point,
        frame_shape
    ):
        height, width = frame_shape[
            :2
        ]

        margin = (
            self.coordinate_margin_pixels
        )

        return (
            -margin
            <= point[0]
            <= width + margin
            and -margin
            <= point[1]
            <= height + margin
        )

    def _single_candidate(
        self,
        data
    ):
        if not self._usable(
            data
        ):
            return None

        return {
            "position": (
                int(
                    round(
                        data[
                            "position"
                        ][0]
                    )
                ),
                int(
                    round(
                        data[
                            "position"
                        ][1]
                    )
                )
            ),
            "confidence": float(
                data[
                    "confidence"
                ]
            )
        }

    def _center_candidate(
        self,
        data_a,
        data_b
    ):
        if (
            not self._usable(
                data_a
            )
            or not self._usable(
                data_b
            )
        ):
            return None

        position_a = data_a[
            "position"
        ]

        position_b = data_b[
            "position"
        ]

        return {
            "position": (
                int(
                    round(
                        (
                            position_a[0]
                            + position_b[0]
                        )
                        / 2.0
                    )
                ),

                int(
                    round(
                        (
                            position_a[1]
                            + position_b[1]
                        )
                        / 2.0
                    )
                )
            ),

            "confidence": (
                float(
                    data_a[
                        "confidence"
                    ]
                )
                + float(
                    data_b[
                        "confidence"
                    ]
                )
            ) / 2.0
        }

    def _select_candidate(
        self,
        candidates
    ):
        best_method = None
        best_score = -1.0

        for method, data in (
            candidates.items()
        ):
            if data is None:
                continue

            score = (
                data[
                    "confidence"
                ]
                * self.bias[
                    method
                ]
            )

            if score > best_score:
                best_score = score
                best_method = method

        return best_method