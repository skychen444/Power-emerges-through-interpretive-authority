"""
main.py

Power Emerges Through Interpretive Authority

Main integration layer.

Observation -> Reduction -> Per-session Evidence
-> Per-session Selection -> Spatial Analysis
-> Per-session Judgement -> Per-session Interpretation
-> Visitor Arbitration -> Drawing -> AxiDraw
"""

import sys
import time

print(
    "Python version:",
    sys.version
)

print(
    "Python executable:",
    sys.executable
)

import cv2

from observation import Observation
from reduction import BodyCenterReduction
from evidence import Evidence
from selection import Selection
from judgement import Judgement
from spatial import SpatialAnalysis
from interpretation import Interpretation
from drawing import Drawing
from axidraw_controller import AxiDrawController
from judgement_debug import JudgementDebug
from engagement import Engagement
from drawing_placement import DrawingPlacement

from config import (
    WINDOW_NAME,
    POINT_RADIUS,
    POINT_COLOR,
    AXIDRAW_PORT
)


observer = Observation()
reduction_engine = BodyCenterReduction()
engagement_engine = Engagement()
drawing_placement = DrawingPlacement()

visitor_states = {}
selected_session_id = None

# Session currently owning the global Drawing episode.
# Stroke history remains global and is never cleared here.
drawing_session_id = None

# A completed visitor may become eligible again when they remain
# engaged after a short pause. This prevents the installation from
# appearing frozen while people are still facing the work.
CONTINUATION_COOLDOWN_SECONDS = 10.0
CONTINUATION_MINIMUM_BELIEF = 0.78
CONTINUATION_MINIMUM_NEW_EVIDENCE = 20
OWNER_INTENT_GRACE_SECONDS = 3.0


axidraw_controller = AxiDrawController(
    port=AXIDRAW_PORT
)

axidraw_connected = (
    axidraw_controller.connect()
)

judgement_debug = JudgementDebug(
    enabled=True,
    print_interval=0.5
)


drawing = Drawing(
    axidraw_controller=(
        axidraw_controller
        if axidraw_connected
        else None
    ),
    physical_interval=2.0
)


def create_visitor_state(
    session_id
):
    print(
        "Creating visitor state |",
        "session_id:",
        session_id
    )

    return {
        "evidence": Evidence(),
        "selection": Selection(),
        "judgement": Judgement(),
        "spatial": SpatialAnalysis(),
        "interpretation": Interpretation(),
        "current_track_id": None,
        "last_seen": None,
        "last_evidence_time": None,
        "intent": None,
        "last_valid_intent_at": None,
        "engagement": None,
        "effective_belief": 0.0,
        "cycle_completed": False,
        "cycle_completed_at": None,
        "completion_evidence_count": 0,
        "completed_cycle_count": 0
    }


def close_visitor_state(
    session_id
):
    global selected_session_id
    global drawing_session_id

    state = visitor_states.pop(
        session_id,
        None
    )

    if state is None:
        return

    print(
        "Closing visitor state |",
        "session_id:",
        session_id
    )

    state[
        "judgement"
    ].clear()

    state[
        "evidence"
    ].reset_episode()

    if (
        selected_session_id
        == session_id
    ):
        selected_session_id = None

    # End the Drawing episode only when the confirmed
    # closed visitor currently owns that episode.
    if (
        drawing_session_id
        == session_id
    ):
        drawing.end_episode()
        drawing_session_id = None



def update_drawing_session_owner(
    new_session_id
):
    """
    Transfers or releases the global Drawing episode.

    Pending physical actions belong to the previous owner, so they are
    removed before ownership changes. An action already executing inside
    AxiDrawController is allowed to finish safely.
    """

    global drawing_session_id

    if drawing_session_id == new_session_id:
        return

    previous_session_id = drawing_session_id

    if drawing_session_id is not None:
        drawing.clear_physical_queue()
        drawing.end_episode()

    drawing_session_id = new_session_id

    print(
        "Drawing session owner changed |",
        "from:",
        previous_session_id,
        "| to:",
        drawing_session_id
    )


def select_drawing_session(
    active_session_ids
):
    """
    Selects among active, eligible visitors.

    Foreground always outranks background. Within the same level,
    the highest effective belief is selected.
    """

    candidates = []

    for session_id in active_session_ids:
        state = visitor_states.get(
            session_id
        )

        if state is None:
            continue

        engagement = state.get(
            "engagement"
        )

        if (
            engagement is None
            or not engagement.get(
                "eligible",
                False
            )
            or state.get(
                "intent"
            ) is None
            or state.get(
                "cycle_completed",
                False
            )
        ):
            continue

        level = engagement.get(
            "level",
            "background"
        )

        level_rank = (
            1
            if level == "foreground"
            else 0
        )

        candidates.append(
            (
                level_rank,
                state.get(
                    "effective_belief",
                    0.0
                ),
                state.get(
                    "last_evidence_time"
                )
                or 0.0,
                -session_id,
                session_id
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    return candidates[0][4]


def emergency_pause():
    drawing.emergency_pause()


def clear_pending_actions():
    drawing.clear_physical_queue()


def prepare_new_sheet():
    """
    Pauses drawing and resets all interpretation state for a new sheet.

    Observation continues running. Active people will be recreated as
    fresh visitor states on the following frame.
    """

    global selected_session_id
    global drawing_session_id

    print()
    print("================================")
    print("PREPARING NEW SHEET")
    print("================================")

    drawing.reset_sheet()
    drawing_placement.reset_sheet()

    visitor_states.clear()
    selected_session_id = None
    drawing_session_id = None

    print(
        "New sheet ready. Replace the paper, then press P "
        "to resume physical drawing."
    )


def return_axidraw_home():
    if not axidraw_connected:
        print(
            "Cannot return home: "
            "AxiDraw is not connected."
        )
        return

    print()
    print(
        "================================"
    )
    print(
        "RETURNING AXIDRAW HOME"
    )
    print(
        "================================"
    )

    drawing.set_physical_enabled(
        False
    )

    drawing.clear_physical_queue()

    axidraw_controller.pen_up()
    axidraw_controller.go_home()

    print(
        "AxiDraw returned home."
    )


try:
    while True:
        (
            frame,
            observations,
            session_events
        ) = observer.read()

        if frame is None:
            print(
                "Camera not found."
            )
            break

        active_session_ids = {
            observation[
                "session_id"
            ]
            for observation
            in observations
        }

        # Register one missing frame for sessions that are
        # temporarily absent but not yet formally closed.
        for session_id, state in (
            visitor_states.items()
        ):
            if (
                session_id
                not in active_session_ids
            ):
                state[
                    "evidence"
                ].register_missing_frame()

        # Each validated session receives its own complete
        # Evidence -> Judgement -> Interpretation pipeline.
        for observation in observations:
            session_id = observation[
                "session_id"
            ]

            state = visitor_states.get(
                session_id
            )

            if state is None:
                state = create_visitor_state(
                    session_id
                )

                visitor_states[
                    session_id
                ] = state

            state[
                "current_track_id"
            ] = observation[
                "track_id"
            ]

            state[
                "last_seen"
            ] = observation[
                "time"
            ]

            representation = (
                reduction_engine.get_position(
                    observation,
                    frame.shape
                )
            )

            selected_evidence = []

            if representation is not None:
                engagement = engagement_engine.classify(
                    observation=observation,
                    representation=representation,
                    frame_shape=frame.shape
                )

                state["engagement"] = engagement

                representation = representation.copy()
                representation["engagement_level"] = (
                    engagement["level"]
                )
                representation["engagement_weight"] = (
                    engagement["weight"]
                )

                evidence_added = state[
                    "evidence"
                ].add(
                    representation
                )

                if evidence_added:
                    state[
                        "last_evidence_time"
                    ] = representation[
                        "time"
                    ]

                    selected_evidence = state[
                        "selection"
                    ].select(
                        state[
                            "evidence"
                        ]
                    )

                    dense_area = state[
                        "spatial"
                    ].find_dense_area(
                        selected_evidence
                    )

                    closed_area = state[
                        "evidence"
                    ].closed_area(
                        dense_area
                    )

                    state[
                        "judgement"
                    ].update(
                        evidence=state[
                            "evidence"
                        ],
                        selected_evidence=(
                            selected_evidence
                        ),
                        dense_area=dense_area,
                        closed_area=closed_area
                    )

                    state["effective_belief"] = (
                        state["judgement"].get_belief()
                        * engagement["weight"]
                    )

                    # A visitor who has completed a full cycle may be
                    # interpreted again after remaining engaged for ten
                    # additional seconds and producing new evidence.
                    if state.get("cycle_completed", False):
                        completed_at = state.get(
                            "cycle_completed_at"
                        )

                        current_evidence_count = (
                            state["judgement"]
                            .get_components()
                            .get("evidence_count", 0)
                        )

                        new_evidence_count = (
                            current_evidence_count
                            - state.get(
                                "completion_evidence_count",
                                0
                            )
                        )

                        cooldown_elapsed = (
                            completed_at is not None
                            and (
                                time.monotonic()
                                - completed_at
                            )
                            >= CONTINUATION_COOLDOWN_SECONDS
                        )

                        continuation_ready = (
                            cooldown_elapsed
                            and engagement["level"]
                            == "foreground"
                            and state["effective_belief"]
                            >= CONTINUATION_MINIMUM_BELIEF
                            and new_evidence_count
                            >= CONTINUATION_MINIMUM_NEW_EVIDENCE
                        )

                        if continuation_ready:
                            state["cycle_completed"] = False
                            state["cycle_completed_at"] = None
                            state["completion_evidence_count"] = (
                                current_evidence_count
                            )
                            state["completed_cycle_count"] += 1

                            # Preserve the current calculated belief and
                            # components, but begin the new drawing cycle
                            # from Commit rather than recovering directly
                            # into Rewrite.
                            state["judgement"].stage = "commit"
                            state[
                                "judgement"
                            ].highest_stage_reached = "commit"
                            state[
                                "judgement"
                            ].commit_evidence_count = (
                                current_evidence_count
                            )
                            state[
                                "judgement"
                            ].commit_reached_at = (
                                time.monotonic()
                            )

                            print(
                                "Visitor reactivated after cooldown |",
                                "session_id:",
                                session_id,
                                "| continuation cycle:",
                                state["completed_cycle_count"],
                                "| new evidence:",
                                new_evidence_count,
                                "| effective belief:",
                                round(
                                    state["effective_belief"],
                                    2
                                )
                            )

                cv2.circle(
                    frame,
                    representation[
                        "position"
                    ],
                    POINT_RADIUS,
                    POINT_COLOR,
                    -1
                )

                cv2.putText(
                    frame,
                    (
                        f"S{session_id} "
                        f"T{observation['track_id']}"
                    ),
                    (
                        representation[
                            "position"
                        ][0] + 10,

                        representation[
                            "position"
                        ][1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 0),
                    2
                )

            raw_intent = state[
                "interpretation"
            ].update(
                selected_evidence,
                state[
                    "judgement"
                ]
            )

            state["intent"] = drawing_placement.place(
                intent=raw_intent,
                session_id=session_id,
                frame_shape=frame.shape,
                cycle_index=state.get(
                    "completed_cycle_count",
                    0
                )
            )

            if state["intent"] is not None:
                state["last_valid_intent_at"] = time.monotonic()
                state["intent"]["engagement_level"] = (
                    state["engagement"]["level"]
                )
                state["intent"]["effective_belief"] = (
                    state["effective_belief"]
                )

        # Confirmed exits end only the matching visitor state.
        for event in session_events:
            if (
                event.get(
                    "event"
                )
                == "session_closed"
            ):
                close_visitor_state(
                    event[
                        "session_id"
                    ]
                )

        best_session_id = select_drawing_session(
            active_session_ids
        )

        current_owner_state = visitor_states.get(
            drawing_session_id
        )

        owner_last_valid_intent_at = None

        if current_owner_state is not None:
            owner_last_valid_intent_at = current_owner_state.get(
                "last_valid_intent_at"
            )

        owner_in_intent_grace = (
            owner_last_valid_intent_at is not None
            and (
                time.monotonic()
                - owner_last_valid_intent_at
            ) <= OWNER_INTENT_GRACE_SECONDS
        )

        owner_remains_valid = (
            drawing_session_id is not None
            and current_owner_state is not None
            and not current_owner_state.get(
                "cycle_completed",
                False
            )
            and (
                (
                    drawing_session_id in active_session_ids
                    and current_owner_state.get("intent") is not None
                )
                or owner_in_intent_grace
            )
        )

        # A foreground visitor immediately suppresses a background owner.
        foreground_takeover = False

        if (
            owner_remains_valid
            and best_session_id is not None
            and best_session_id != drawing_session_id
        ):
            owner_level = current_owner_state[
                "engagement"
            ]["level"]

            challenger_level = visitor_states[
                best_session_id
            ]["engagement"]["level"]

            foreground_takeover = (
                owner_level == "background"
                and challenger_level == "foreground"
            )

        if (
            not owner_remains_valid
            or foreground_takeover
        ):
            selected_session_id = best_session_id
        else:
            selected_session_id = drawing_session_id

        update_drawing_session_owner(
            selected_session_id
        )

        selected_intent = None

        if selected_session_id is not None:
            selected_intent = visitor_states[
                selected_session_id
            ][
                "intent"
            ]

        drawing.draw(
            frame,
            selected_intent
        )

        if (
            drawing_session_id is not None
            and drawing.is_cycle_complete()
        ):
            owner_state = visitor_states.get(
                drawing_session_id
            )

            if owner_state is not None:
                owner_state["cycle_completed"] = True
                owner_state["cycle_completed_at"] = (
                    time.monotonic()
                )
                owner_state[
                    "completion_evidence_count"
                ] = (
                    owner_state["judgement"]
                    .get_components()
                    .get("evidence_count", 0)
                )

                print(
                    "Interpretation cycle completed after expansion |",
                    "session_id:",
                    drawing_session_id,
                    "| cooldown:",
                    CONTINUATION_COOLDOWN_SECONDS,
                    "seconds"
                )

        frame_height = frame.shape[
            0
        ]

        cv2.putText(
            frame,
            (
                f"Active sessions: "
                f"{len(active_session_ids)}"
            ),
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )

        judgement_debug.draw(
            frame=frame,
            active_session_ids=active_session_ids,
            visitor_states=visitor_states,
            selected_session_id=selected_session_id
        )

        judgement_debug.print(
            active_session_ids=active_session_ids,
            visitor_states=visitor_states
        )

        controls = [
            "P: Toggle physical drawing",
            "SPACE: Emergency pause",
            "C: Clear pending queue",
            "N: Prepare new sheet",
            "H: Return AxiDraw home",
            "Q: Return home and quit"
        ]

        for control_index, control_text in enumerate(controls):
            cv2.putText(
                frame,
                control_text,
                (
                    20,
                    frame_height
                    - 145
                    + control_index * 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord(
            "p"
        ):
            drawing.set_physical_enabled(
                not drawing.physical_enabled
            )

        elif key == 32:
            emergency_pause()

        elif key == ord(
            "c"
        ):
            clear_pending_actions()

        elif key == ord(
            "n"
        ):
            prepare_new_sheet()

        elif key == ord(
            "h"
        ):
            return_axidraw_home()

        elif key == ord(
            "q"
        ):
            print()
            print(
                "Quit requested."
            )

            drawing.set_physical_enabled(
                False
            )

            drawing.clear_physical_queue()

            if axidraw_connected:
                axidraw_controller.pen_up()
                axidraw_controller.go_home()

            break

finally:
    drawing.set_physical_enabled(
        False
    )

    drawing.clear_physical_queue()

    if axidraw_connected:
        axidraw_controller.disconnect(
            return_home=False
        )

    observer.release()

    cv2.destroyAllWindows()

    print(
        "System shutdown complete."
    )