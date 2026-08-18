"""
drawing.py

Drawing Grammar Engine v4.0

Rewrite is treated as a later intervention into an already
materialised commit line.
"""

import math
import random
import time

import cv2
import numpy as np


class Drawing:

    VALID_INTENT_TYPES = {
        "probe_intent",
        "commit_intent",
        "rewrite_intent"
    }

    STAGE_RANK = {
        "probe": 1,
        "commit": 2,
        "rewrite": 3
    }

    def __init__(
        self,
        axidraw_controller=None,
        physical_interval=2.0
    ):
        self.axidraw_controller = axidraw_controller
        self.physical_interval = physical_interval

        self.physical_enabled = False
        self.last_physical_draw_time = 0.0

        self.active_episode_id = None
        self.episode_count = 0
        self.mark_count = 0
        self.rewrite_count = 0
        self.max_rewrites_per_commit = 3
        self.expansion_count = 0
        self.max_expansions_per_cycle = 1
        self.expansion_minimum_seconds = 7.0
        self.expansion_evidence_gap = 70
        self.last_expansion_time = None

        self.episode_direction = None
        self.highest_materialised_stage_rank = 0
        self.last_materialised_intent = None
        self.episode_commit = None

        self.stroke_history = []
        self.physical_queue = []

        self.missing_intent_frames = 0
        self.max_missing_intent_frames = 15

        self.rewrite_evidence_thresholds = (60, 55, 65)
        self.rewrite_minimum_intervals = (6.0, 5.0, 6.0)
        self.last_rewrite_time = None

        # Spatial re-commit:
        # the same visitor may receive a new definition line after
        # moving into a clearly different area and sustaining attention
        # there again. This is not continuous path drawing.
        self.spatial_recommit_distance_threshold = 120.0
        self.spatial_recommit_dwell_threshold = 0.68
        self.spatial_recommit_attention_threshold = 0.74
        self.spatial_recommit_confidence_threshold = 0.76
        self.spatial_recommit_evidence_gap = 18

        # Spatial re-commit diagnostics.
        # Print at most twice per second to avoid flooding the terminal.
        self.spatial_recommit_debug_interval = 0.50
        self.last_spatial_recommit_debug_time = 0.0

        self.rewrite_variants = (
            {
                "name": "front_cut",
                "start_range": (0.10, 0.22),
                "crossings": 5,
                "coverage_range": (0.46, 0.62),
                "amplitude_range": (26.0, 46.0),
                "bias_range": (-0.08, 0.04),
                "tail_scale": (0.20, 0.34)
            },
            {
                "name": "middle_pressure",
                "start_range": (0.24, 0.38),
                "crossings": 6,
                "coverage_range": (0.44, 0.60),
                "amplitude_range": (24.0, 44.0),
                "bias_range": (-0.03, 0.07),
                "tail_scale": (0.18, 0.30)
            },
            {
                "name": "deep_overwrite",
                "start_range": (0.18, 0.34),
                "crossings": 7,
                "coverage_range": (0.50, 0.68),
                "amplitude_range": (28.0, 50.0),
                "bias_range": (-0.05, 0.05),
                "tail_scale": (0.22, 0.36)
            },
            {
                "name": "rear_cut",
                "start_range": (0.34, 0.48),
                "crossings": 5,
                "coverage_range": (0.40, 0.56),
                "amplitude_range": (25.0, 45.0),
                "bias_range": (-0.04, 0.09),
                "tail_scale": (0.18, 0.32)
            }
        )

    def set_physical_enabled(
        self,
        enabled
    ):
        self.physical_enabled = enabled

        print(
            "Physical drawing:",
            self.physical_enabled
        )

    def clear_physical_queue(self):
        """
        Removes physical actions that have not started yet.

        Already materialised preview strokes remain visible.
        """
        queued_count = len(self.physical_queue)
        self.physical_queue.clear()

        print(
            "Physical queue cleared |",
            "removed:",
            queued_count
        )

    def emergency_pause(self):
        """
        Stops new physical output and removes pending actions.

        The controller raises the pen, but does not return home.
        """
        self.set_physical_enabled(False)
        self.clear_physical_queue()

        if self.axidraw_controller is not None:
            self.axidraw_controller.pen_up()

        print("Emergency pause activated.")

    def reset_sheet(self):
        """
        Starts a new physical sheet.

        Previous physical marks remain on the removed sheet, while
        the digital preview, pending queue, and episode-local state
        are reset for the replacement sheet.
        """
        self.set_physical_enabled(False)
        self.clear_physical_queue()
        self.stroke_history.clear()
        self.end_episode()
        self.last_physical_draw_time = 0.0

        if self.axidraw_controller is not None:
            self.axidraw_controller.pen_up()

        print("New sheet state prepared.")

    def end_episode(self):
        if self.active_episode_id is not None:
            print(
                "Drawing episode reset:",
                self.active_episode_id
            )

        self.active_episode_id = None
        self.mark_count = 0
        self.rewrite_count = 0
        self.expansion_count = 0
        self.last_expansion_time = None
        self.episode_direction = None
        self.highest_materialised_stage_rank = 0
        self.last_materialised_intent = None
        self.episode_commit = None
        self.last_rewrite_time = None
        self.missing_intent_frames = 0

    def draw(
        self,
        frame,
        intent
    ):
        if intent is None:
            self._handle_missing_intent()
        else:
            self._handle_intent(intent)

        self._draw_preview(frame)
        self._flush_physical_queue(frame)

    def _handle_intent(
        self,
        intent
    ):
        intent_type = intent.get("type")

        if intent_type not in self.VALID_INTENT_TYPES:
            return

        stage = intent.get("stage")

        if stage not in self.STAGE_RANK:
            return

        self.missing_intent_frames = 0

        if self.active_episode_id is None:
            self._start_episode(intent)

        if self._should_expand(intent):
            mark = self._generate_expansion_mark(
                intent
            )
        elif self._should_spatial_recommit(intent):
            mark = self._generate_spatial_recommit_mark(
                intent
            )
        else:
            if not self._should_materialise(intent):
                return

            mark = self._generate_mark(intent)

        if not mark["actions"]:
            return

        for action in mark["actions"]:
            self.stroke_history.append(action)

            if self.physical_enabled:
                self.physical_queue.append(action)

        self.last_materialised_intent = intent.copy()

        materialised_stage = mark["stage"]
        stage_rank = self.STAGE_RANK[
            materialised_stage
        ]

        if stage_rank > self.highest_materialised_stage_rank:
            self.highest_materialised_stage_rank = stage_rank

        print(
            f"Interpretation materialised | "
            f"Episode: {mark['episode_id']} | "
            f"Mark: {mark['mark_count']} | "
            f"Stage: {mark['stage']} | "
            f"Grammar: {mark['grammar']} | "
            f"Actions: {len(mark['actions'])}"
        )

    def is_cycle_complete(self):
        """
        A cycle is complete after one commit and the configured
        maximum number of rewrites.
        """

        return (
            self.episode_commit is not None
            and self.rewrite_count
            >= self.max_rewrites_per_commit
            and self.expansion_count
            >= self.max_expansions_per_cycle
        )


    def _should_expand(
        self,
        intent
    ):
        """
        Allows one post-Rewrite expansion after three Rewrites.
        Both additional time and evidence are required.
        """

        if self.episode_commit is None:
            return False

        if self.rewrite_count < self.max_rewrites_per_commit:
            return False

        if self.expansion_count >= self.max_expansions_per_cycle:
            return False

        if intent.get("stage") != "rewrite":
            return False

        if self.last_materialised_intent is None:
            return False

        if self.last_rewrite_time is None:
            return False

        elapsed = time.monotonic() - self.last_rewrite_time

        evidence_gap = (
            intent.get("evidence_count", 0)
            - self.last_materialised_intent.get(
                "evidence_count",
                0
            )
        )

        return (
            elapsed >= self.expansion_minimum_seconds
            and evidence_gap >= self.expansion_evidence_gap
        )

    def _generate_expansion_mark(
        self,
        intent
    ):
        """
        Generates one expansion after the third Rewrite.

        Sustained-attention grammars create a directional extension.
        Repeated-engagement grammars create an angular bending path.
        """

        self.mark_count += 1
        self.expansion_count += 1
        self.last_expansion_time = time.monotonic()

        grammar = intent.get("grammar", "overwrite")

        if grammar in {"cross_out", "correction"}:
            expansion_kind = "bending"
            points = self._build_bending_expansion(intent)
        else:
            expansion_kind = "extension"
            points = self._build_directional_extension(intent)

        action = {
            "type": "polyline",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": "rewrite",
            "grammar": f"expansion_{expansion_kind}",
            "points": points,
            "confidence": intent["confidence"],
            "evidence_count": intent["evidence_count"],
            "segment_delay": 0.05
        }

        return {
            "type": "drawing_mark",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": "rewrite",
            "grammar": action["grammar"],
            "actions": [action]
        }

    def _build_directional_extension(
        self,
        intent
    ):
        """
        Extends the interpretation away from its retained Commit.
        """

        commit_start = self.episode_commit["start"]
        commit_end = self.episode_commit["end"]

        direction = self.episode_direction
        if direction is None:
            direction = 0.0

        start = (
            commit_end
            if math.cos(direction) >= 0
            else commit_start
        )

        confidence = intent.get("confidence", 0.8)
        total_length = 170.0 + confidence * 70.0
        segment_count = 8
        bend_sign = 1.0 if math.sin(direction) >= 0 else -1.0

        points = [start]

        for index in range(1, segment_count + 1):
            progress = index / segment_count
            local_angle = (
                direction
                + bend_sign * (0.10 + 0.34 * progress)
            )
            distance = total_length * progress
            lateral = 18.0 * math.sin(progress * math.pi)

            x = (
                start[0]
                + math.cos(local_angle) * distance
                - math.sin(direction) * lateral
            )
            y = (
                start[1]
                + math.sin(local_angle) * distance
                + math.cos(direction) * lateral
            )

            points.append(
                (int(round(x)), int(round(y)))
            )

        return points

    def _build_bending_expansion(
        self,
        intent
    ):
        """
        Creates an angular segmented curve for repeated engagement.
        """

        commit_start = self.episode_commit["start"]
        commit_end = self.episode_commit["end"]

        direction = self.episode_direction
        if direction is None:
            direction = 0.0

        start = (
            commit_end
            if math.cos(direction) >= 0
            else commit_start
        )

        confidence = intent.get("confidence", 0.8)
        segment_lengths = (36.0, 42.0, 48.0, 55.0, 62.0)
        turn_scale = 0.28 + confidence * 0.18
        turn_sign = 1.0
        current_angle = direction
        current = start
        points = [start]

        for index, length in enumerate(segment_lengths):
            current_angle += (
                turn_scale
                * turn_sign
                * (0.75 + 0.12 * index)
            )

            x = current[0] + math.cos(current_angle) * length
            y = current[1] + math.sin(current_angle) * length

            current = (int(round(x)), int(round(y)))
            points.append(current)
            turn_sign *= -1.0

        return points

    def _handle_missing_intent(
        self
    ):
        """
        Records a temporary absence of interpretive intent.

        A brief belief drop, tracking fluctuation, or temporary
        Selection failure must not end the Drawing episode.

        Episode closure is controlled externally by main.py when:
        - the owning visitor session is formally closed, or
        - drawing ownership transfers to another visitor.
        """

        if self.active_episode_id is None:
            return

        self.missing_intent_frames += 1

    def _start_episode(
        self,
        intent
    ):
        self.episode_count += 1
        self.active_episode_id = self.episode_count

        self.mark_count = 0
        self.rewrite_count = 0
        self.expansion_count = 0
        self.last_expansion_time = None
        self.highest_materialised_stage_rank = 0
        self.last_materialised_intent = None
        self.episode_commit = None
        self.last_rewrite_time = None

        initial_direction = intent.get(
            "direction"
        )

        if initial_direction is None:
            initial_direction = 0.0

        self.episode_direction = initial_direction

    def _should_spatial_recommit(
        self,
        intent
    ):
        """
        Allows a new commit within the same visitor session only when
        the visitor has moved into a clearly different spatial area
        and sustained attention there again.

        When blocked, prints the exact failed conditions at a limited rate.
        """

        if self.episode_commit is None:
            return False

        # The first retained Commit must receive at least one Rewrite
        # before movement can be interpreted as a new spatial Commit.
        # This prevents relocation from repeatedly replacing Rewrite.
        if self.rewrite_count < 1:
            return False

        stage = intent.get("stage")

        if stage not in ("probe", "commit", "rewrite"):
            return False

        confidence = intent.get(
            "confidence",
            0.0
        )

        components = intent.get(
            "components",
            {}
        )

        dwell_score = components.get(
            "dwell",
            0.0
        )

        sustained_attention = components.get(
            "sustained_attention",
            0.0
        )

        evidence_gap = (
            intent.get("evidence_count", 0)
            - self.episode_commit.get(
                "evidence_count",
                0
            )
        )

        position_difference = self._point_distance(
            intent["origin"],
            self.episode_commit["origin"]
        )

        checks = {
            "confidence": (
                confidence
                >= self.spatial_recommit_confidence_threshold
            ),
            "attention": (
                dwell_score
                >= self.spatial_recommit_dwell_threshold
                or sustained_attention
                >= self.spatial_recommit_attention_threshold
            ),
            "evidence_gap": (
                evidence_gap
                >= self.spatial_recommit_evidence_gap
            ),
            "distance": (
                position_difference
                >= self.spatial_recommit_distance_threshold
            )
        }

        passed = all(checks.values())

        if not passed:
            self._debug_spatial_recommit(
                stage=stage,
                confidence=confidence,
                dwell_score=dwell_score,
                sustained_attention=sustained_attention,
                evidence_gap=evidence_gap,
                position_difference=position_difference,
                checks=checks
            )

        return passed

    def _debug_spatial_recommit(
        self,
        stage,
        confidence,
        dwell_score,
        sustained_attention,
        evidence_gap,
        position_difference,
        checks
    ):
        now = time.monotonic()

        if (
            now
            - self.last_spatial_recommit_debug_time
            < self.spatial_recommit_debug_interval
        ):
            return

        self.last_spatial_recommit_debug_time = now

        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        print(
            "Spatial recommit blocked |",
            "failed:",
            ",".join(failed),
            "| stage:",
            stage,
            "| distance:",
            round(position_difference, 1),
            "/",
            self.spatial_recommit_distance_threshold,
            "| confidence:",
            round(confidence, 2),
            "/",
            self.spatial_recommit_confidence_threshold,
            "| dwell:",
            round(dwell_score, 2),
            "/",
            self.spatial_recommit_dwell_threshold,
            "| sustained:",
            round(sustained_attention, 2),
            "/",
            self.spatial_recommit_attention_threshold,
            "| evidence_gap:",
            evidence_gap,
            "/",
            self.spatial_recommit_evidence_gap
        )

    def _generate_spatial_recommit_mark(
        self,
        intent
    ):
        """
        Materialises a new definition line for a new area while
        preserving the same visitor session and all previous marks.
        """

        self.mark_count += 1

        recommit_direction = intent.get(
            "direction"
        )

        if recommit_direction is not None:
            self.episode_direction = (
                recommit_direction
            )

        recommit_intent = intent.copy()
        recommit_intent["grammar"] = (
            "spatial_recommit"
        )

        action = self._generate_line_action(
            intent=recommit_intent,
            stage="commit"
        )

        self.episode_commit = {
            "start": action["start"],
            "end": action["end"],
            "origin": intent["origin"],
            "direction": self.episode_direction,
            "length": self._point_distance(
                action["start"],
                action["end"]
            ),
            "evidence_count": intent[
                "evidence_count"
            ],
            "materialised_at": time.monotonic()
        }

        # A new definition line becomes the current object
        # of any later rewrite.
        self.rewrite_count = 0
        self.expansion_count = 0
        self.last_rewrite_time = None
        self.last_expansion_time = None

        return {
            "type": "drawing_mark",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": "commit",
            "grammar": "spatial_recommit",
            "actions": [action]
        }

    def _should_materialise(
        self,
        intent
    ):
        stage = intent["stage"]
        stage_rank = self.STAGE_RANK[stage]

        if stage_rank < self.highest_materialised_stage_rank:
            return False

        if stage_rank > self.highest_materialised_stage_rank:
            return True

        if stage in ("probe", "commit"):
            return False

        if (
            stage == "rewrite"
            and self.rewrite_count
            >= self.max_rewrites_per_commit
        ):
            return False

        if self.last_materialised_intent is None:
            return True

        rewrite_index = min(
            self.rewrite_count,
            len(self.rewrite_evidence_thresholds) - 1
        )

        required_evidence = self.rewrite_evidence_thresholds[
            rewrite_index
        ]
        required_interval = self.rewrite_minimum_intervals[
            rewrite_index
        ]

        evidence_difference = (
            intent["evidence_count"]
            - self.last_materialised_intent["evidence_count"]
        )

        reference_time = self.last_rewrite_time

        if reference_time is None:
            reference_time = (
                self.episode_commit or {}
            ).get("materialised_at")

        enough_time = (
            reference_time is not None
            and (
                time.monotonic()
                - reference_time
            ) >= required_interval
        )

        return (
            enough_time
            and evidence_difference >= required_evidence
        )

    def _generate_mark(
        self,
        intent
    ):
        self.mark_count += 1
        stage = intent["stage"]

        if stage == "rewrite":
            self.rewrite_count += 1
            self.last_rewrite_time = time.monotonic()

        if stage in ("probe", "commit"):
            if stage == "commit":
                commit_direction = intent.get(
                    "direction"
                )

                if commit_direction is not None:
                    self.episode_direction = (
                        commit_direction
                    )

            action = self._generate_line_action(
                intent=intent,
                stage=stage
            )

            if stage == "commit":
                self.episode_commit = {
                    "start": action["start"],
                    "end": action["end"],
                    "origin": intent["origin"],
                    "direction": self.episode_direction,
                    "length": self._point_distance(
                        action["start"],
                        action["end"]
                    ),
                    "evidence_count": intent[
                        "evidence_count"
                    ],
                    "materialised_at": time.monotonic()
                }

            actions = [action]

        else:
            actions = []

            # Rewrite must intervene in a materialised commit.
            # This branch protects against a rare state recovery in
            # which Judgement resumes at rewrite stage without retained
            # commit geometry.
            if self.episode_commit is None:
                commit_intent = intent.copy()
                commit_intent["grammar"] = (
                    "recovered_commit"
                )

                commit_action = self._generate_line_action(
                    intent=commit_intent,
                    stage="commit"
                )

                self.episode_commit = {
                    "start": commit_action["start"],
                    "end": commit_action["end"],
                    "origin": intent["origin"],
                    "direction": self.episode_direction,
                    "length": self._point_distance(
                        commit_action["start"],
                        commit_action["end"]
                    ),
                    "evidence_count": intent[
                        "evidence_count"
                    ],
                    "materialised_at": time.monotonic()
                }

                actions.append(
                    commit_action
                )

            rewrite_action = self._generate_rewrite_action(
                intent
            )

            if rewrite_action is not None:
                actions.append(
                    rewrite_action
                )

        return {
            "type": "drawing_mark",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": stage,
            "grammar": intent.get("grammar", stage),
            "actions": actions
        }

    def _generate_line_action(
        self,
        intent,
        stage
    ):
        length = self._length_for_stage(
            stage=stage,
            confidence=intent["confidence"]
        )

        start, end = self._make_line(
            centre=intent["origin"],
            angle=self.episode_direction,
            length=length
        )

        return {
            "type": "line",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": stage,
            "grammar": intent.get("grammar", stage),
            "start": start,
            "end": end,
            "confidence": intent["confidence"],
            "evidence_count": intent["evidence_count"]
        }

    def _generate_rewrite_action(
        self,
        intent
    ):
        if self.episode_commit is None:
            print(
                "Rewrite skipped: no commit geometry "
                "has been materialised."
            )
            return None

        variant_index = (
            self.rewrite_count - 1
        ) % len(self.rewrite_variants)

        variant = self.rewrite_variants[variant_index]

        points = self._build_rewrite_polyline(
            commit_start=self.episode_commit["start"],
            commit_end=self.episode_commit["end"],
            variant=variant
        )

        return {
            "type": "polyline",
            "episode_id": self.active_episode_id,
            "mark_count": self.mark_count,
            "stage": "rewrite",
            "grammar": f"rewrite_local_{variant['name']}",
            "points": points,
            "confidence": intent["confidence"],
            "evidence_count": intent["evidence_count"],
            "segment_delay": 0.04
        }

    def _build_rewrite_polyline(
        self,
        commit_start,
        commit_end,
        variant
    ):
        """
        Build an asymmetric intervention that repeatedly crosses
        the retained commit line.

        The path remains rule-based, but avoids evenly spaced,
        icon-like oscillation. It expands beyond the commit and
        returns through it several times, producing a more invasive
        physical overwrite.
        """

        line_length = self._point_distance(
            commit_start,
            commit_end
        )

        tangent, normal = self._unit_vectors(
            commit_start,
            commit_end
        )

        start_ratio = random.uniform(
            variant["start_range"][0],
            variant["start_range"][1]
        )

        coverage_ratio = random.uniform(
            variant["coverage_range"][0],
            variant["coverage_range"][1]
        )

        longitudinal_bias = random.uniform(
            variant["bias_range"][0],
            variant["bias_range"][1]
        )

        crossings = variant["crossings"]
        side = random.choice((-1.0, 1.0))
        points = []

        # Begin outside the commit line so the intervention approaches
        # the existing definition rather than appearing inside it.
        entry_ratio = max(
            0.04,
            min(
                start_ratio + longitudinal_bias,
                0.92
            )
        )

        entry_base = self._point_on_line(
            commit_start,
            commit_end,
            entry_ratio
        )

        entry_amplitude = random.uniform(
            variant["amplitude_range"][0],
            variant["amplitude_range"][1]
        ) * random.uniform(0.75, 1.05)

        points.append(
            (
                int(
                    entry_base[0]
                    + normal[0]
                    * entry_amplitude
                    * side
                    - tangent[0]
                    * line_length
                    * random.uniform(0.03, 0.09)
                ),
                int(
                    entry_base[1]
                    + normal[1]
                    * entry_amplitude
                    * side
                    - tangent[1]
                    * line_length
                    * random.uniform(0.03, 0.09)
                )
            )
        )

        for index in range(crossings):
            progress = (
                index + 1
            ) / max(1, crossings)

            # Non-linear spacing prevents a regular zig-zag icon.
            warped_progress = progress ** random.uniform(
                0.82,
                1.18
            )

            ratio = (
                start_ratio
                + coverage_ratio * warped_progress
                + longitudinal_bias
                + random.uniform(-0.045, 0.045)
            )

            ratio = max(
                0.04,
                min(ratio, 0.96)
            )

            base_point = self._point_on_line(
                commit_start,
                commit_end,
                ratio
            )

            amplitude = random.uniform(
                variant["amplitude_range"][0],
                variant["amplitude_range"][1]
            )

            # Middle crossings carry more force.
            centrality = 1.0 - abs(
                progress - 0.5
            ) * 2.0

            amplitude *= (
                0.85
                + centrality * random.uniform(0.20, 0.55)
            )

            # Small tangent drift introduces angular variation while
            # preserving repeated intersections with the commit.
            tangent_shift = (
                line_length
                * random.uniform(-0.07, 0.07)
            )

            point = (
                int(
                    base_point[0]
                    + normal[0]
                    * amplitude
                    * side
                    + tangent[0]
                    * tangent_shift
                ),
                int(
                    base_point[1]
                    + normal[1]
                    * amplitude
                    * side
                    + tangent[1]
                    * tangent_shift
                )
            )

            points.append(point)
            side *= -1.0

        # Exit with an asymmetric tail rather than returning neatly
        # to the commit line.
        tail_ratio = max(
            0.06,
            min(
                start_ratio
                + coverage_ratio
                + random.uniform(-0.02, 0.10),
                0.97
            )
        )

        tail_base = self._point_on_line(
            commit_start,
            commit_end,
            tail_ratio
        )

        tail_amplitude = random.uniform(
            variant["amplitude_range"][0],
            variant["amplitude_range"][1]
        ) * random.uniform(
            variant["tail_scale"][0],
            variant["tail_scale"][1]
        )

        tail_tangent_shift = (
            line_length
            * random.uniform(0.05, 0.14)
        )

        points.append(
            (
                int(
                    tail_base[0]
                    + normal[0]
                    * tail_amplitude
                    * side
                    + tangent[0]
                    * tail_tangent_shift
                ),
                int(
                    tail_base[1]
                    + normal[1]
                    * tail_amplitude
                    * side
                    + tangent[1]
                    * tail_tangent_shift
                )
            )
        )

        return points

    def _length_for_stage(
        self,
        stage,
        confidence
    ):
        confidence_adjustment = (
            confidence - 0.70
        ) * 20.0

        if stage == "probe":
            return max(
                45.0,
                58.0 + confidence_adjustment
            )

        if stage == "commit":
            return max(
                150.0,
                190.0 + confidence_adjustment * 1.5
            )

        return 30.0

    def _make_line(
        self,
        centre,
        angle,
        length
    ):
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2

        start = (
            int(centre[0] - dx),
            int(centre[1] - dy)
        )

        end = (
            int(centre[0] + dx),
            int(centre[1] + dy)
        )

        return start, end

    def _draw_preview(
        self,
        frame
    ):
        for action in self.stroke_history:
            thickness = self._preview_thickness(
                action["stage"]
            )

            if action["type"] == "line":
                cv2.line(
                    frame,
                    action["start"],
                    action["end"],
                    (0, 0, 0),
                    thickness
                )

            elif action["type"] == "polyline":
                points = action.get("points", [])

                if len(points) < 2:
                    continue

                point_array = np.array(
                    points,
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                cv2.polylines(
                    frame,
                    [point_array],
                    False,
                    (0, 0, 0),
                    thickness
                )

        if self.active_episode_id is None:
            status = "Drawing Episode: None"
        else:
            status = (
                f"Drawing Episode: "
                f"{self.active_episode_id} | "
                f"Marks: {self.mark_count}"
            )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )

    def _preview_thickness(
        self,
        stage
    ):
        if stage == "probe":
            return 1

        return 2

    def _flush_physical_queue(
        self,
        frame
    ):
        if not self.physical_enabled:
            return

        if self.axidraw_controller is None:
            return

        if len(self.physical_queue) == 0:
            return

        current_time = time.time()

        if (
            current_time
            - self.last_physical_draw_time
            < self.physical_interval
        ):
            return

        action = self.physical_queue.pop(0)

        if action["type"] == "line":
            physical_action = {
                "type": "definition_line",
                "start": action["start"],
                "end": action["end"],
                "stage": action["stage"],
                "grammar": action.get(
                    "grammar",
                    action["stage"]
                )
            }

        elif action["type"] == "polyline":
            physical_action = {
                "type": "definition_polyline",
                "points": action["points"],
                "stage": action["stage"],
                "grammar": action.get(
                    "grammar",
                    action["stage"]
                ),
                "segment_delay": action.get(
                    "segment_delay",
                    0.05
                )
            }

        else:
            return

        self.axidraw_controller.draw_action(
            physical_action,
            frame.shape
        )

        self.last_physical_draw_time = current_time

    def _point_on_line(
        self,
        start,
        end,
        ratio
    ):
        return (
            start[0]
            + (end[0] - start[0])
            * ratio,

            start[1]
            + (end[1] - start[1])
            * ratio
        )

    def _unit_vectors(
        self,
        start,
        end
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length = math.hypot(dx, dy)

        if length == 0:
            return (
                (1.0, 0.0),
                (0.0, 1.0)
            )

        tangent = (
            dx / length,
            dy / length
        )

        normal = (
            -tangent[1],
            tangent[0]
        )

        return tangent, normal

    def _point_distance(
        self,
        point_a,
        point_b
    ):
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]

        return math.sqrt(
            dx * dx
            + dy * dy
        )