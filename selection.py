"""
selection.py

Selects which accepted evidence the system chooses to believe.

Selection is not neutral.
It privileges behavioural evidence that can be interpreted as:

1. sustained attention
   - remaining in front of the work
   - slow movement
   - continued presence within a limited area

2. repeated engagement
   - revisiting previously occupied positions
   - changing direction
   - moving back and forth within the same session

Technical confidence only decides whether evidence is reliable enough
to enter Selection. It does not make behaviour more meaningful.
"""

from config import *


class Selection:

    def select(self, evidence):
        """
        Selects evidence according to the system's designed preference.

        The method preserves the existing Selection role:
        - it does not calculate Judgement belief
        - it does not find dense areas
        - it does not generate interpretation grammar
        - it only decides which accepted evidence is privileged

        Returns:
            list:
                Copies of selected evidence items with an added
                selection weight and diagnostic selection metadata.
        """

        selected = []

        history = evidence.get_history()

        if len(history) < SELECTION_MIN_EVIDENCE_COUNT:
            return selected

        dwell = evidence.dwell_time()
        avg_speed = evidence.average_speed()
        trajectory = evidence.trajectory()

        # ------------------------------------------
        # Sustained attention
        # ------------------------------------------

        dwell_score = self._normalise(
            dwell,
            SELECTION_DWELL_SECONDS
        )

        slow_movement_score = (
            1.0
            - self._normalise(
                avg_speed,
                SELECTION_SLOW_SPEED_MAX
            )
        )

        sustained_attention_score = self._clamp(
            (
                dwell_score
                * SELECTION_SUSTAINED_DWELL_WEIGHT
            )
            + (
                slow_movement_score
                * SELECTION_SUSTAINED_SLOW_WEIGHT
            )
        )

        # ------------------------------------------
        # Repeated engagement
        # ------------------------------------------

        revisit_score = self._normalise(
            trajectory.get("revisit", 0),
            SELECTION_REVISIT_COUNT
        )

        direction_change_score = self._normalise(
            trajectory.get("direction_changes", 0),
            SELECTION_DIRECTION_CHANGE_COUNT
        )

        repeated_engagement_score = self._clamp(
            (
                revisit_score
                * SELECTION_REPEATED_REVISIT_WEIGHT
            )
            + (
                direction_change_score
                * SELECTION_REPEATED_DIRECTION_WEIGHT
            )
        )

        # Either route may make the evidence meaningful.
        #
        # A visitor can appear highly engaged by standing still
        # and watching, or by repeatedly moving around and returning.
        interpretive_value = max(
            sustained_attention_score,
            repeated_engagement_score
        )

        recent_start = max(
            0,
            len(history) - SELECTION_RECENT_N
        )

        for index, item in enumerate(history):
            confidence = item["confidence"]

            # Technical reliability gate.
            # Confidence does not contribute to artistic meaning.
            if confidence < SELECTION_CONFIDENCE_THRESHOLD:
                continue

            weight = interpretive_value

            is_recent = index >= recent_start

            # Recency can strengthen meaningful evidence,
            # but can no longer make weak behaviour pass by itself.
            if is_recent:
                weight += SELECTION_RECENT_BONUS

            weight = self._clamp(weight)

            if weight < SELECTION_WEIGHT_THRESHOLD:
                continue

            new_item = item.copy()

            new_item["weight"] = weight
            new_item["selection"] = {
                "interpretive_value": interpretive_value,
                "sustained_attention":
                    sustained_attention_score,
                "repeated_engagement":
                    repeated_engagement_score,
                "dwell": dwell_score,
                "slow_movement":
                    slow_movement_score,
                "revisit": revisit_score,
                "direction_changes":
                    direction_change_score,
                "recent": is_recent
            }

            selected.append(new_item)

        return selected

    def _normalise(self, value, maximum):
        if maximum <= 0:
            return 0.0

        return self._clamp(
            float(value)
            / float(maximum)
        )

    def _clamp(self, value):
        return max(
            0.0,
            min(float(value), 1.0)
        )