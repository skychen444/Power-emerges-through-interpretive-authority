"""
observation.py

Multi-person Observation layer.

Responsibilities
----------------
- Read camera frames.
- Detect people with YOLO Pose.
- Track temporary identities with ByteTrack.
- Validate credible visitors.
- Reconnect short tracking interruptions into visitor sessions.
- Return raw keypoints for every ACTIVE visitor session.

Reduction, behavioural evidence, judgement, and interpretation
remain outside this module.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import CAMERA_ID
from visitor_validation import VisitorValidator
from visitor_session import VisitorSessionRegistry


MODEL_PATH = "yolo11n-pose.pt"
TRACKER_CONFIG = "bytetrack.yaml"

PERSON_CONFIDENCE = 0.35
KEYPOINT_CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class Observation:
    def __init__(
        self,
        camera_id=CAMERA_ID,
        model_path=MODEL_PATH
    ):
        self.cap = self._open_camera(
            camera_id
        )

        print(
            "Loading YOLO pose model..."
        )

        self.model = YOLO(
            model_path
        )

        self.validator = VisitorValidator(
            probation_seconds=0.8,
            minimum_observations=6,
            minimum_geometry_ratio=0.60,
            minimum_motion_pixels=25.0,
            minimum_box_change_ratio=0.12,
            track_timeout_seconds=2.0
        )

        self.session_registry = (
            VisitorSessionRegistry(
                reconnect_window_seconds=1.90,
                confirmed_exit_seconds=2.20,
                maximum_reconnect_distance=180.0,
                maximum_shoulder_ratio_difference=0.60,
                pending_candidate_timeout_seconds=1.50
            )
        )

    def read(self):
        """
        Returns:
            frame:
                Annotated camera frame.

            observations:
                List of raw observations belonging only to
                validated ACTIVE visitor sessions.

            session_events:
                Session lifecycle events, including confirmed exits.
        """

        success, frame = self.cap.read()

        if (
            not success
            or frame is None
        ):
            print(
                "Camera frame read failed |",
                "opened:",
                self.cap.isOpened()
            )

            return None, [], []

        frame = cv2.flip(
            frame,
            1
        )

        timestamp = time.monotonic()

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=PERSON_CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=[0],
            verbose=False
        )

        frame_records = []
        seen_track_ids = set()
        seen_active_track_ids = set()

        if results:
            result = results[0]

            boxes = result.boxes
            keypoints = result.keypoints

            if (
                boxes is not None
                and keypoints is not None
                and len(boxes) > 0
                and boxes.id is not None
            ):
                boxes_xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                boxes_conf = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

                keypoints_xy = (
                    keypoints.xy
                    .cpu()
                    .numpy()
                )

                if keypoints.conf is not None:
                    keypoints_conf = (
                        keypoints.conf
                        .cpu()
                        .numpy()
                    )
                else:
                    keypoints_conf = np.ones(
                        (
                            len(keypoints_xy),
                            keypoints_xy.shape[1]
                        ),
                        dtype=np.float32
                    )

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                detection_count = min(
                    len(boxes_xyxy),
                    len(boxes_conf),
                    len(keypoints_xy),
                    len(track_ids)
                )

                # Phase 1:
                # Validate and collect all detections.
                for index in range(
                    detection_count
                ):
                    track_id = int(
                        track_ids[index]
                    )

                    seen_track_ids.add(
                        track_id
                    )

                    raw_keypoints = (
                        self._extract_keypoints(
                            keypoints_xy[index],
                            keypoints_conf[index]
                        )
                    )

                    tracking_representation = (
                        self._tracking_representation(
                            raw_keypoints
                        )
                    )

                    validation = (
                        self.validator.update(
                            track_id=track_id,
                            representation=(
                                tracking_representation
                            ),
                            bounding_box=(
                                boxes_xyxy[index]
                            ),
                            timestamp=timestamp
                        )
                    )

                    frame_records.append(
                        {
                            "track_id": track_id,
                            "bounding_box": tuple(
                                map(
                                    float,
                                    boxes_xyxy[index]
                                )
                            ),
                            "person_confidence": float(
                                boxes_conf[index]
                            ),
                            "keypoints": raw_keypoints,
                            "tracking_representation": (
                                tracking_representation
                            ),
                            "validation": validation,
                            "session_id": None
                        }
                    )

        # Phase 2A:
        # Protect all ACTIVE tracks first.
        for record in frame_records:
            if not record[
                "validation"
            ][
                "active"
            ]:
                continue

            track_id = record[
                "track_id"
            ]

            seen_active_track_ids.add(
                track_id
            )

            session_result = (
                self.session_registry.assign(
                    track_id=track_id,
                    representation=record[
                        "tracking_representation"
                    ],
                    timestamp=timestamp
                )
            )

            record[
                "session_id"
            ] = session_result[
                "session_id"
            ]

        # Phase 2B:
        # Only probation candidates may reserve LOST sessions.
        for record in frame_records:
            if record[
                "validation"
            ][
                "active"
            ]:
                continue

            self.session_registry.observe_candidate(
                track_id=record[
                    "track_id"
                ],
                representation=record[
                    "tracking_representation"
                ],
                timestamp=timestamp
            )

        expired_tracks = (
            self.validator.mark_frame_complete(
                seen_track_ids=seen_track_ids,
                timestamp=timestamp
            )
        )

        for expired in expired_tracks:
            print(
                "Track expired |",
                expired
            )

        session_events = (
            self.session_registry.mark_frame_complete(
                seen_active_track_ids=(
                    seen_active_track_ids
                ),
                seen_candidate_track_ids=(
                    seen_track_ids
                ),
                timestamp=timestamp
            )
        )

        observations = []

        for record in frame_records:
            self._draw_debug_record(
                frame,
                record
            )

            session_id = record[
                "session_id"
            ]

            if session_id is None:
                continue

            observations.append(
                {
                    "session_id": int(
                        session_id
                    ),
                    "track_id": int(
                        record[
                            "track_id"
                        ]
                    ),
                    "time": timestamp,
                    "bounding_box": record[
                        "bounding_box"
                    ],
                    "person_confidence": record[
                        "person_confidence"
                    ],
                    "keypoints": record[
                        "keypoints"
                    ]
                }
            )

        return (
            frame,
            observations,
            session_events
        )

    def release(self):
        self.cap.release()

    def _open_camera(
        self,
        camera_id
    ):
        print(
            "Opening camera ID:",
            camera_id
        )

        capture = cv2.VideoCapture(
            camera_id,
            cv2.CAP_DSHOW
        )

        print(
            "DirectShow opened:",
            capture.isOpened()
        )

        if not capture.isOpened():
            capture.release()

            capture = cv2.VideoCapture(
                camera_id
            )

            print(
                "Default backend opened:",
                capture.isOpened()
            )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        print(
            "Actual camera resolution:",
            int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            ),
            "x",
            int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )
        )

        return capture

    def _extract_keypoints(
        self,
        keypoints_xy,
        keypoints_conf
    ):
        return {
            "nose": self._keypoint_data(
                keypoints_xy[NOSE],
                keypoints_conf[NOSE]
            ),

            "left_shoulder": self._keypoint_data(
                keypoints_xy[
                    LEFT_SHOULDER
                ],
                keypoints_conf[
                    LEFT_SHOULDER
                ]
            ),

            "right_shoulder": self._keypoint_data(
                keypoints_xy[
                    RIGHT_SHOULDER
                ],
                keypoints_conf[
                    RIGHT_SHOULDER
                ]
            ),

            "left_hip": self._keypoint_data(
                keypoints_xy[
                    LEFT_HIP
                ],
                keypoints_conf[
                    LEFT_HIP
                ]
            ),

            "right_hip": self._keypoint_data(
                keypoints_xy[
                    RIGHT_HIP
                ],
                keypoints_conf[
                    RIGHT_HIP
                ]
            )
        }

    def _keypoint_data(
        self,
        point,
        confidence
    ):
        x = float(
            point[0]
        )

        y = float(
            point[1]
        )

        confidence = float(
            confidence
        )

        valid = (
            np.isfinite(x)
            and np.isfinite(y)
            and x > 0
            and y > 0
            and confidence
            >= KEYPOINT_CONFIDENCE
        )

        return {
            "position": (
                (x, y)
                if valid
                else None
            ),
            "confidence": confidence,
            "valid": bool(
                valid
            )
        }

    def _tracking_representation(
        self,
        keypoints
    ):
        nose = self._point(
            keypoints[
                "nose"
            ]
        )

        left_shoulder = self._point(
            keypoints[
                "left_shoulder"
            ]
        )

        right_shoulder = self._point(
            keypoints[
                "right_shoulder"
            ]
        )

        left_hip = self._point(
            keypoints[
                "left_hip"
            ]
        )

        right_hip = self._point(
            keypoints[
                "right_hip"
            ]
        )

        shoulder_center = self._center(
            left_shoulder,
            right_shoulder
        )

        hip_center = self._center(
            left_hip,
            right_hip
        )

        torso_center = None
        position_source = None

        if (
            shoulder_center is not None
            and hip_center is not None
        ):
            torso_center = (
                shoulder_center[0] * 0.4
                + hip_center[0] * 0.6,

                shoulder_center[1] * 0.4
                + hip_center[1] * 0.6
            )

            position_source = "torso"

        elif hip_center is not None:
            torso_center = hip_center
            position_source = "hips"

        elif shoulder_center is not None:
            torso_center = (
                shoulder_center
            )

            position_source = (
                "shoulders"
            )

        shoulder_width = None
        shoulder_angle = None

        if (
            left_shoulder is not None
            and right_shoulder is not None
        ):
            dx = (
                right_shoulder[0]
                - left_shoulder[0]
            )

            dy = (
                right_shoulder[1]
                - left_shoulder[1]
            )

            shoulder_width = float(
                np.hypot(
                    dx,
                    dy
                )
            )

            shoulder_angle = float(
                np.arctan2(
                    dy,
                    dx
                )
            )

        nose_offset = None

        if (
            nose is not None
            and shoulder_center is not None
        ):
            nose_offset = (
                nose[0]
                - shoulder_center[0],

                nose[1]
                - shoulder_center[1]
            )

        return {
            "position": torso_center,
            "torso_center": torso_center,
            "position_source": position_source,
            "nose": nose,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "shoulder_center": shoulder_center,
            "hip_center": hip_center,
            "shoulder_angle": shoulder_angle,
            "shoulder_width": shoulder_width,
            "nose_offset": nose_offset,
            "visible": {
                "nose": nose is not None,
                "shoulders": (
                    left_shoulder is not None
                    and right_shoulder is not None
                ),
                "hips": (
                    left_hip is not None
                    and right_hip is not None
                )
            }
        }

    def _point(
        self,
        data
    ):
        if not data[
            "valid"
        ]:
            return None

        return data[
            "position"
        ]

    def _center(
        self,
        point_a,
        point_b
    ):
        if (
            point_a is None
            or point_b is None
        ):
            return None

        return (
            (
                point_a[0]
                + point_b[0]
            )
            / 2.0,

            (
                point_a[1]
                + point_b[1]
            )
            / 2.0
        )

    def _draw_debug_record(
        self,
        frame,
        record
    ):
        x1, y1, x2, y2 = [
            int(
                round(
                    value
                )
            )
            for value
            in record[
                "bounding_box"
            ]
        ]

        active = record[
            "validation"
        ][
            "active"
        ]

        session_id = record[
            "session_id"
        ]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            (
                3
                if active
                else 1
            )
        )

        status = (
            "ACTIVE"
            if active
            else "CANDIDATE"
        )

        session_text = (
            f" | S{session_id}"
            if session_id is not None
            else ""
        )

        cv2.putText(
            frame,
            (
                f"ID {record['track_id']} | "
                f"{status}{session_text}"
            ),
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )