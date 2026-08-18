"""
evidence.py

Stores accepted observations and calculates behavioural evidence.

Rule 1: Dwell Time
Rule 2: Trajectory
Rule 3: Closed Area
Rule 4: Speed

Evidence also manages the lifecycle of one visitor's
accumulated evidence.
"""

import math

import numpy as np


class Evidence:

    def __init__(self):
        self.history = []

        # Consecutive frames without accepted evidence.
        self.missing_frames = 0
        self.max_missing_frames = 20

        # Protects the evidence history from sudden tracking jumps.
        self.rejected_jump_count = 0
        self.max_rejected_jumps = 15
        self.maximum_jump = 220

        # A repeated large displacement can be a real relocation,
        # not a tracking glitch. A compact cluster of consecutive
        # rejected positions is accepted as a new reference.
        self.pending_relocation = []
        self.relocation_confirmation_count = 4
        self.relocation_cluster_radius = 75.0
        self.relocation_maximum_age = 1.25

        # Independent motion-direction memory.
        #
        # Attention Selection may temporarily reject moving frames.
        # Direction therefore remains available after the visitor
        # stops and later reaches Commit.
        self.motion_positions = []
        self.motion_direction_window = 30
        self.motion_direction_max_age = 6.0
        self.motion_minimum_step = 8.0
        self.motion_minimum_spread = 10.0

        # Trajectory analysis thresholds.
        #
        # Small frame-to-frame movement is treated as tracking jitter,
        # not as meaningful direction change or repeated engagement.
        self.minimum_trajectory_step = 10.0
        self.direction_reversal_cosine = -0.20

        # A return is counted only when the visitor:
        # 1. occupied an earlier position,
        # 2. moved clearly away from it,
        # 3. later returned near it.
        self.return_radius = 50.0
        self.return_departure_radius = 100.0
        self.return_minimum_gap = 15
        self.return_cooldown = 12

    def add(self, representation):
        if representation is None:
            return False

        current = representation["position"]

        if len(self.history) == 0:
            accepted = self._prepare_representation(
                representation,
                previous_position=None,
                relocation=False
            )

            self.history.append(accepted)
            self.missing_frames = 0
            self.rejected_jump_count = 0
            self.pending_relocation.clear()
            return True

        last = self.history[-1]["position"]

        jump = self._distance(
            current,
            last
        )

        if jump > self.maximum_jump:
            self.rejected_jump_count += 1

            print(
                "Evidence jump rejected:",
                round(jump, 2),
                "| rejected:",
                self.rejected_jump_count
            )

            if self._register_pending_relocation(
                representation
            ):
                accepted = self._prepare_representation(
                    representation,
                    previous_position=last,
                    relocation=True
                )

                print(
                    "Evidence relocation accepted:",
                    accepted["position"],
                    "| confirmations:",
                    self.relocation_confirmation_count
                )

                self.history.append(accepted)
                self.missing_frames = 0
                self.rejected_jump_count = 0
                self.pending_relocation.clear()
                return True

            if (
                self.rejected_jump_count
                >= self.max_rejected_jumps
            ):
                accepted = self._prepare_representation(
                    representation,
                    previous_position=last,
                    relocation=True
                )

                print(
                    "Evidence reference reset:",
                    accepted["position"]
                )

                self.history.append(accepted)
                self.missing_frames = 0
                self.rejected_jump_count = 0
                self.pending_relocation.clear()
                return True

            return False

        accepted = self._prepare_representation(
            representation,
            previous_position=last,
            relocation=False
        )

        self.history.append(accepted)
        self.missing_frames = 0
        self.rejected_jump_count = 0
        self.pending_relocation.clear()
        return True

    def _prepare_representation(
        self,
        representation,
        previous_position,
        relocation
    ):
        accepted = representation.copy()

        current_position = accepted["position"]
        current_time = float(
            accepted.get("time", 0.0)
        )

        if previous_position is not None:
            step = self._distance(
                previous_position,
                current_position
            )

            if (
                step >= self.motion_minimum_step
                and (
                    step <= self.maximum_jump
                    or relocation
                )
            ):
                if len(self.motion_positions) == 0:
                    self.motion_positions.append(
                        {
                            "position": previous_position,
                            "time": current_time
                        }
                    )

                self.motion_positions.append(
                    {
                        "position": current_position,
                        "time": current_time
                    }
                )

                if (
                    len(self.motion_positions)
                    > self.motion_direction_window
                ):
                    self.motion_positions = (
                        self.motion_positions[
                            -self.motion_direction_window:
                        ]
                    )

        direction_data = self.recent_motion_direction(
            current_time=current_time
        )

        if direction_data is None:
            accepted["motion_direction"] = None
            accepted["motion_direction_age"] = None
            accepted["motion_direction_spread"] = 0.0
        else:
            accepted["motion_direction"] = (
                direction_data["angle"]
            )
            accepted["motion_direction_age"] = (
                direction_data["age"]
            )
            accepted["motion_direction_spread"] = (
                direction_data["spread"]
            )

        accepted["relocation"] = bool(
            relocation
        )

        return accepted

    def _register_pending_relocation(
        self,
        representation
    ):
        current = representation["position"]
        current_time = float(
            representation.get("time", 0.0)
        )

        if len(self.pending_relocation) == 0:
            self.pending_relocation = [
                {
                    "position": current,
                    "time": current_time
                }
            ]
            return False

        first_time = self.pending_relocation[0][
            "time"
        ]

        age = current_time - first_time

        cluster_center = self._cluster_center(
            self.pending_relocation
        )

        distance_to_cluster = self._distance(
            current,
            cluster_center
        )

        if (
            age > self.relocation_maximum_age
            or distance_to_cluster
            > self.relocation_cluster_radius
        ):
            self.pending_relocation = [
                {
                    "position": current,
                    "time": current_time
                }
            ]
            return False

        self.pending_relocation.append(
            {
                "position": current,
                "time": current_time
            }
        )

        return (
            len(self.pending_relocation)
            >= self.relocation_confirmation_count
        )

    def _cluster_center(
        self,
        items
    ):
        x = sum(
            item["position"][0]
            for item in items
        ) / len(items)

        y = sum(
            item["position"][1]
            for item in items
        ) / len(items)

        return (
            x,
            y
        )

    def recent_motion_direction(
        self,
        current_time=None
    ):
        """
        Returns the principal axis of recent meaningful movement.

        This memory is independent from Selection, so a visitor may
        move first, stop, and still receive a Commit aligned with the
        recently observed movement.
        """

        if len(self.motion_positions) < 3:
            return None

        if current_time is None:
            current_time = self.motion_positions[-1][
                "time"
            ]

        latest_motion_time = self.motion_positions[-1][
            "time"
        ]

        age = current_time - latest_motion_time

        if age > self.motion_direction_max_age:
            return None

        positions = np.array(
            [
                item["position"]
                for item in self.motion_positions[
                    -self.motion_direction_window:
                ]
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

        spread = math.sqrt(
            principal_value
        )

        if spread < self.motion_minimum_spread:
            return None

        vector = eigenvectors[
            :,
            principal_index
        ]

        net_displacement = (
            positions[-1] - positions[0]
        )

        if np.linalg.norm(net_displacement) >= 4.0:
            if np.dot(
                vector,
                net_displacement
            ) < 0.0:
                vector = -vector
        else:
            if (
                vector[0] < 0.0
                or (
                    abs(vector[0]) < 1e-9
                    and vector[1] < 0.0
                )
            ):
                vector = -vector

        return {
            "angle": math.atan2(
                float(vector[1]),
                float(vector[0])
            ),
            "spread": spread,
            "age": max(0.0, age)
        }

    def register_missing_frame(self):
        if len(self.history) == 0:
            return

        self.missing_frames += 1

    def episode_finished(self):
        return (
            len(self.history) > 0
            and self.missing_frames
            >= self.max_missing_frames
        )

    def reset_episode(self):
        print(
            "Evidence episode cleared |",
            "Evidence removed:",
            len(self.history)
        )

        self.history.clear()
        self.missing_frames = 0
        self.rejected_jump_count = 0
        self.pending_relocation.clear()
        self.motion_positions.clear()

    def get_history(self):
        return self.history

    def latest_speed(self):
        if len(self.history) < 2:
            return 0

        current = self.history[-1]
        previous = self.history[-2]

        x1, y1 = previous["position"]
        x2, y2 = current["position"]

        t1 = previous["time"]
        t2 = current["time"]

        dt = t2 - t1

        if dt <= 0.03:
            return 0

        distance = (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

        speed = distance / dt

        if speed > 500:
            return 500

        return speed

    def average_speed(self, last_n=30):
        if len(self.history) < 2:
            return 0

        recent = self.history[-last_n:]

        if len(recent) < 2:
            return 0

        speeds = []

        for index in range(1, len(recent)):
            current = recent[index]
            previous = recent[index - 1]

            x1, y1 = previous["position"]
            x2, y2 = current["position"]

            t1 = previous["time"]
            t2 = current["time"]

            dt = t2 - t1

            if dt <= 0:
                continue

            distance = (
                (x2 - x1) ** 2
                + (y2 - y1) ** 2
            ) ** 0.5

            speed = distance / dt

            if speed <= 500:
                speeds.append(speed)

        if len(speeds) == 0:
            return 0

        return sum(speeds) / len(speeds)

    def trajectory(self, last_n=120):
        if len(self.history) < 2:
            return {
                "distance": 0,
                "revisit": 0,
                "loop": False,
                "direction_changes": 0
            }

        recent = self.history[-last_n:]
        positions = [
            item["position"]
            for item in recent
        ]

        distance = 0.0
        direction_changes = 0

        previous_vector = None

        for index in range(1, len(positions)):
            previous_position = positions[index - 1]
            current_position = positions[index]

            dx = (
                current_position[0]
                - previous_position[0]
            )

            dy = (
                current_position[1]
                - previous_position[1]
            )

            step_distance = (
                dx ** 2
                + dy ** 2
            ) ** 0.5

            distance += step_distance

            # Ignore small tracking jitter when measuring
            # meaningful changes of direction.
            if step_distance < self.minimum_trajectory_step:
                continue

            current_vector = (
                dx / step_distance,
                dy / step_distance
            )

            if previous_vector is not None:
                cosine = (
                    previous_vector[0]
                    * current_vector[0]
                    + previous_vector[1]
                    * current_vector[1]
                )

                if cosine < self.direction_reversal_cosine:
                    direction_changes += 1

            previous_vector = current_vector

        revisit = self._count_returns(
            positions
        )

        start = positions[0]
        end = positions[-1]

        loop_distance = (
            (end[0] - start[0]) ** 2
            + (end[1] - start[1]) ** 2
        ) ** 0.5

        loop = (
            loop_distance < 80
            and distance > 300
        )

        return {
            "distance": distance,
            "revisit": revisit,
            "loop": loop,
            "direction_changes": direction_changes
        }

    def _count_returns(self, positions):
        """
        Counts actual leave-and-return events.

        Remaining near an earlier point does not count as a return.
        Small tracking jitter does not create repeated engagement.
        """

        if len(positions) < (
            self.return_minimum_gap + 2
        ):
            return 0

        return_count = 0
        last_return_index = -self.return_cooldown

        for current_index in range(
            self.return_minimum_gap,
            len(positions)
        ):
            if (
                current_index
                - last_return_index
                < self.return_cooldown
            ):
                continue

            current = positions[current_index]

            latest_possible_origin = (
                current_index
                - self.return_minimum_gap
            )

            return_found = False

            for origin_index in range(
                latest_possible_origin
            ):
                origin = positions[origin_index]

                current_distance = self._distance(
                    current,
                    origin
                )

                if current_distance > self.return_radius:
                    continue

                moved_away = any(
                    self._distance(
                        positions[middle_index],
                        origin
                    )
                    >= self.return_departure_radius
                    for middle_index in range(
                        origin_index + 1,
                        current_index
                    )
                )

                if moved_away:
                    return_found = True
                    break

            if return_found:
                return_count += 1
                last_return_index = current_index

        return return_count

    def dwell_time(self, radius=80):
        if len(self.history) < 2:
            return 0

        latest_position = self.history[-1][
            "position"
        ]

        latest_time = self.history[-1]["time"]
        dwell_start_time = latest_time

        for item in reversed(self.history):
            position = item["position"]

            distance = self._distance(
                position,
                latest_position
            )

            if distance <= radius:
                dwell_start_time = item["time"]
            else:
                break

        return latest_time - dwell_start_time

    def closed_area(self, dense_area):
        if dense_area is None:
            return None

        if len(self.history) == 0:
            return None

        center = dense_area["center"]
        radius = dense_area["radius"]

        inside_count = 0

        for item in self.history:
            x, y = item["position"]

            distance = self._distance(
                (x, y),
                center
            )

            if distance <= radius:
                inside_count += 1

        occupancy = (
            inside_count
            / len(self.history)
        )

        current_position = self.history[-1][
            "position"
        ]

        current_distance = self._distance(
            current_position,
            center
        )

        inside = current_distance <= radius
        duration = self.dwell_time(radius)

        return {
            "inside": inside,
            "occupancy": occupancy,
            "duration": duration,
            "center": center,
            "radius": radius,
            "current_position": current_position,
            "current_distance": current_distance
        }

    def micro_motion(self, last_n=30):
        if len(self.history) < last_n:
            return None

        recent = self.history[-last_n:]
        total_motion = 0.0

        for index in range(1, len(recent)):
            previous = recent[
                index - 1
            ]["position"]

            current = recent[
                index
            ]["position"]

            total_motion += self._distance(
                previous,
                current
            )

        average_motion = (
            total_motion
            / (len(recent) - 1)
        )

        return {
            "average_motion": average_motion,
            "static": average_motion < 2.0
        }

    def _distance(self, point_a, point_b):
        return (
            (point_a[0] - point_b[0]) ** 2
            + (point_a[1] - point_b[1]) ** 2
        ) ** 0.5