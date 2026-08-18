"""
judgement_debug.py

Development-only diagnostics for Judgement.

This module only displays and prints values that Judgement
has already calculated. It does not change Selection,
Judgement, Interpretation, Drawing, or visitor arbitration.
"""

import time
import cv2


class JudgementDebug:

    def __init__(
        self,
        enabled=True,
        print_interval=0.5
    ):
        self.enabled = enabled
        self.print_interval = print_interval
        self.last_print_time = 0.0

    def draw(
        self,
        frame,
        active_session_ids,
        visitor_states,
        selected_session_id
    ):
        if not self.enabled:
            return

        cv2.putText(
            frame,
            (
                "Drawing visitor session: "
                f"{selected_session_id}"
            ),
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2
        )

        for row_index, session_id in enumerate(
            sorted(active_session_ids)
        ):
            state = visitor_states.get(
                session_id
            )

            if state is None:
                continue

            judgement = state["judgement"]
            components = judgement.get_components()
            dominant = judgement.get_dominant_component()

            dominant_name = (
                dominant.get("name", "none")
                if dominant is not None
                else "none"
            )

            stage = judgement.get_stage() or "none"
            belief = judgement.get_belief()

            sustained = components.get(
                "sustained_attention",
                0.0
            )

            repeated = components.get(
                "repeated_engagement",
                0.0
            )

            evidence_count = int(
                components.get(
                    "evidence_count",
                    0
                )
            )

            first_line_y = (
                105
                + row_index * 44
            )

            cv2.putText(
                frame,
                (
                    f"S{session_id} "
                    f"belief {belief:.2f} "
                    f"stage {stage} "
                    f"dominant {dominant_name}"
                ),
                (20, first_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    f"sustained {sustained:.2f} | "
                    f"repeated {repeated:.2f} | "
                    f"selected {evidence_count}"
                ),
                (20, first_line_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 0),
                2
            )

    def print(
        self,
        active_session_ids,
        visitor_states
    ):
        if not self.enabled:
            return

        now = time.monotonic()

        if (
            now - self.last_print_time
            < self.print_interval
        ):
            return

        self.last_print_time = now

        for session_id in sorted(
            active_session_ids
        ):
            state = visitor_states.get(
                session_id
            )

            if state is None:
                continue

            judgement = state["judgement"]
            components = judgement.get_components()
            dominant = judgement.get_dominant_component()

            dominant_name = (
                dominant.get("name", "none")
                if dominant is not None
                else "none"
            )

            print(
                "Judgement debug | "
                f"S{session_id} | "
                f"sustained: "
                f"{components.get('sustained_attention', 0.0):.2f} | "
                f"repeated: "
                f"{components.get('repeated_engagement', 0.0):.2f} | "
                f"belief: {judgement.get_belief():.2f} | "
                f"stage: {judgement.get_stage() or 'none'} | "
                f"dominant: {dominant_name} | "
                f"selected: "
                f"{int(components.get('evidence_count', 0))}"
            )