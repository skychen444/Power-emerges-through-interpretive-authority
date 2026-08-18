"""
visitor_session.py

Candidate-aware, one-to-one short-term reconciliation for visitor sessions.

A visitor session represents one continuous appearance.

Rules
-----
- One active track can belong to at most one session.
- One session can belong to at most one active track.
- One lost session can be reserved by at most one probation candidate.
- If the original track returns, any competing pending reservation is cancelled.
- A recently missing active track may hand over to a plausible replacement.
- A confirmed exit closes the session; later re-entry creates a new session.

This is short-term continuity, not permanent identity recognition.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class VisitorSession:
    session_id: int
    current_track_id: Optional[int]
    created_at: float
    last_seen: float
    last_position: Optional[Point] = None
    last_shoulder_width: Optional[float] = None
    status: str = "active"
    pending_track_id: Optional[int] = None
    pending_since: Optional[float] = None
    pending_last_seen: Optional[float] = None


class VisitorSessionRegistry:
    """Maps temporary tracker IDs to continuous visitor sessions."""

    def __init__(
        self,
        reconnect_window_seconds=1.90,
        confirmed_exit_seconds=2.20,
        maximum_reconnect_distance=180.0,
        maximum_shoulder_ratio_difference=0.60,
        pending_candidate_timeout_seconds=1.50,
        active_handover_minimum_missing_seconds=0.12,
        active_handover_window_seconds=0.90,
        maximum_active_handover_distance=220.0,
    ):
        self.reconnect_window_seconds = float(
            reconnect_window_seconds
        )
        self.confirmed_exit_seconds = float(
            confirmed_exit_seconds
        )
        self.maximum_reconnect_distance = float(
            maximum_reconnect_distance
        )
        self.maximum_shoulder_ratio_difference = float(
            maximum_shoulder_ratio_difference
        )
        self.pending_candidate_timeout_seconds = float(
            pending_candidate_timeout_seconds
        )
        self.active_handover_minimum_missing_seconds = float(
            active_handover_minimum_missing_seconds
        )
        self.active_handover_window_seconds = float(
            active_handover_window_seconds
        )
        self.maximum_active_handover_distance = float(
            maximum_active_handover_distance
        )

        self.next_session_id = 1

        self.sessions: Dict[
            int,
            VisitorSession
        ] = {}

        self.track_to_session: Dict[
            int,
            int
        ] = {}

        self.pending_track_to_session: Dict[
            int,
            int
        ] = {}

    def observe_candidate(
        self,
        track_id,
        representation,
        timestamp=None
    ):
        """
        Observe every detected candidate, including probation tracks.

        A candidate may reserve one lost session. A session that is already
        active or reserved cannot be offered to another candidate.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        track_id = int(track_id)

        # An already assigned track must never reserve another session.
        if track_id in self.track_to_session:
            return None

        # Keep an existing valid reservation alive.
        pending_session_id = (
            self.pending_track_to_session.get(
                track_id
            )
        )

        if pending_session_id is not None:
            session = self.sessions.get(
                pending_session_id
            )

            if (
                session is not None
                and session.status != "closed"
                and session.pending_track_id == track_id
            ):
                session.pending_last_seen = timestamp
                return pending_session_id

            self.pending_track_to_session.pop(
                track_id,
                None
            )

        matched_session = (
            self._find_reconnect_candidate(
                representation=representation,
                timestamp=timestamp
            )
        )

        if matched_session is None:
            return None

        # Reserve this session in both directions.
        matched_session.status = "pending"
        matched_session.pending_track_id = track_id
        matched_session.pending_since = timestamp
        matched_session.pending_last_seen = timestamp

        self.pending_track_to_session[
            track_id
        ] = matched_session.session_id

        print(
            "Pending reconnect |",
            "candidate_track_id:",
            track_id,
            "| session_id:",
            matched_session.session_id
        )

        return matched_session.session_id

    def assign(
        self,
        track_id,
        representation,
        timestamp=None
    ):
        """Assign one validated active track to exactly one session."""
        if timestamp is None:
            timestamp = time.monotonic()

        track_id = int(track_id)

        existing_session_id = (
            self.track_to_session.get(
                track_id
            )
        )

        if existing_session_id is not None:
            session = self.sessions[
                existing_session_id
            ]

            # The original track has returned. It keeps its session, so any
            # competing probation candidate loses its reservation.
            if (
                session.pending_track_id is not None
                and session.pending_track_id != track_id
            ):
                cancelled_track_id = (
                    session.pending_track_id
                )

                self._release_pending(
                    session
                )

                print(
                    "Pending reconnect cancelled |",
                    "candidate_track_id:",
                    cancelled_track_id,
                    "| session_id:",
                    session.session_id,
                    "| reason: original_track_returned"
                )

            self._update_session(
                session,
                track_id,
                representation,
                timestamp
            )

            return {
                "session_id": session.session_id,
                "new_session": False,
                "reconnected": False,
                "session": session
            }

        pending_session_id = (
            self.pending_track_to_session.get(
                track_id
            )
        )

        if pending_session_id is not None:
            session = self.sessions.get(
                pending_session_id
            )

            reservation_is_valid = (
                session is not None
                and session.status == "pending"
                and session.pending_track_id == track_id
            )

            if reservation_is_valid:
                self.pending_track_to_session.pop(
                    track_id,
                    None
                )

                self._clear_pending(
                    session
                )

                self._connect_track(
                    session,
                    track_id,
                    representation,
                    timestamp
                )

                print(
                    "Track reconnected |",
                    "new_track_id:",
                    track_id,
                    "| session_id:",
                    session.session_id,
                    "| source: pending_candidate"
                )

                return {
                    "session_id": session.session_id,
                    "new_session": False,
                    "reconnected": True,
                    "session": session
                }

            # Remove stale or cancelled reservation.
            self.pending_track_to_session.pop(
                track_id,
                None
            )

        matched_session = (
            self._find_reconnect_candidate(
                representation=representation,
                timestamp=timestamp
            )
        )

        if matched_session is not None:
            self._connect_track(
                matched_session,
                track_id,
                representation,
                timestamp
            )

            print(
                "Track reconnected |",
                "new_track_id:",
                track_id,
                "| session_id:",
                matched_session.session_id,
                "| source: active_match"
            )

            return {
                "session_id": matched_session.session_id,
                "new_session": False,
                "reconnected": True,
                "session": matched_session
            }

        handover_session = (
            self._find_active_handover_candidate(
                representation=representation,
                timestamp=timestamp
            )
        )

        if handover_session is not None:
            previous_track_id = (
                handover_session.current_track_id
            )

            self._connect_track(
                handover_session,
                track_id,
                representation,
                timestamp
            )

            print(
                "Active track handover |",
                "old_track_id:",
                previous_track_id,
                "| new_track_id:",
                track_id,
                "| session_id:",
                handover_session.session_id
            )

            return {
                "session_id": handover_session.session_id,
                "new_session": False,
                "reconnected": True,
                "session": handover_session
            }

        return self._create_session(
            track_id,
            representation,
            timestamp
        )

    def mark_frame_complete(
        self,
        seen_active_track_ids,
        seen_candidate_track_ids=None,
        timestamp=None
    ):
        """Update active, lost, pending, and closed session states."""
        if timestamp is None:
            timestamp = time.monotonic()

        active_seen = {
            int(track_id)
            for track_id in seen_active_track_ids
        }

        candidate_seen = {
            int(track_id)
            for track_id in (
                seen_candidate_track_ids
                or set()
            )
        }

        events = []

        for session in self.sessions.values():
            if session.status == "closed":
                continue

            current_track_id = (
                session.current_track_id
            )

            # The assigned track is still present. It has priority over every
            # pending candidate reservation.
            if (
                current_track_id is not None
                and current_track_id in active_seen
            ):
                if (
                    session.pending_track_id
                    is not None
                    and session.pending_track_id
                    != current_track_id
                ):
                    cancelled_track_id = (
                        session.pending_track_id
                    )

                    self._release_pending(
                        session
                    )

                    print(
                        "Pending reconnect cancelled |",
                        "candidate_track_id:",
                        cancelled_track_id,
                        "| session_id:",
                        session.session_id,
                        "| reason: assigned_track_visible"
                    )

                session.status = "active"
                continue

            pending_track_id = (
                session.pending_track_id
            )

            if pending_track_id is not None:
                if pending_track_id in candidate_seen:
                    session.status = "pending"
                    session.pending_last_seen = (
                        timestamp
                    )
                    continue

                pending_reference = (
                    session.pending_last_seen
                    if session.pending_last_seen
                    is not None
                    else session.pending_since
                )

                if pending_reference is not None:
                    pending_missing = (
                        timestamp
                        - pending_reference
                    )

                    if (
                        pending_missing
                        <= self.pending_candidate_timeout_seconds
                    ):
                        session.status = "pending"
                        continue

                released_track_id = (
                    session.pending_track_id
                )

                self._release_pending(
                    session
                )

                print(
                    "Pending reconnect released |",
                    "candidate_track_id:",
                    released_track_id,
                    "| session_id:",
                    session.session_id,
                    "| reason: candidate_disappeared"
                )

            missing_duration = (
                timestamp
                - session.last_seen
            )

            if (
                missing_duration
                < self.confirmed_exit_seconds
            ):
                session.status = "lost"
                continue

            self._close_session(
                session,
                missing_duration,
                events
            )

        return events

    def _find_active_handover_candidate(
        self,
        representation,
        timestamp
    ):
        """
        Find an active session whose assigned track has only just
        disappeared.

        This handles ByteTrack ID replacement during continuous presence.
        A session whose original track was updated very recently is excluded,
        which protects genuinely simultaneous visitors from being merged.
        """

        new_position = representation.get(
            "position"
        )

        new_shoulder_width = representation.get(
            "shoulder_width"
        )

        best_session = None
        best_score = None

        for session in self.sessions.values():
            if session.status != "active":
                continue

            if session.pending_track_id is not None:
                continue

            missing_duration = (
                timestamp
                - session.last_seen
            )

            if (
                missing_duration
                < self.active_handover_minimum_missing_seconds
                or missing_duration
                > self.active_handover_window_seconds
            ):
                continue

            distance = self._distance(
                session.last_position,
                new_position
            )

            if (
                distance is None
                or distance
                > self.maximum_active_handover_distance
            ):
                continue

            shoulder_difference = (
                self._shoulder_ratio_difference(
                    session.last_shoulder_width,
                    new_shoulder_width
                )
            )

            if (
                shoulder_difference is not None
                and shoulder_difference
                > self.maximum_shoulder_ratio_difference
            ):
                continue

            score = (
                distance
                + missing_duration * 30.0
            )

            if shoulder_difference is not None:
                score += (
                    shoulder_difference
                    * 100.0
                )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score
                best_session = session

        return best_session

    def _find_reconnect_candidate(
        self,
        representation,
        timestamp
    ):
        """
        Return one unreserved lost session.

        Active and pending sessions are excluded. This enforces one-to-one
        ownership before a reservation is created.
        """
        new_position = representation.get(
            "position"
        )

        new_shoulder_width = (
            representation.get(
                "shoulder_width"
            )
        )

        best_session = None
        best_score = None

        for session in self.sessions.values():
            if session.status != "lost":
                continue

            if session.pending_track_id is not None:
                continue

            missing_duration = (
                timestamp
                - session.last_seen
            )

            if (
                missing_duration
                > (
                    self.reconnect_window_seconds
                    + self.pending_candidate_timeout_seconds
                )
            ):
                continue

            distance = self._distance(
                session.last_position,
                new_position
            )

            if (
                distance is None
                or distance
                > self.maximum_reconnect_distance
            ):
                continue

            shoulder_difference = (
                self._shoulder_ratio_difference(
                    session.last_shoulder_width,
                    new_shoulder_width
                )
            )

            if (
                shoulder_difference is not None
                and shoulder_difference
                > self.maximum_shoulder_ratio_difference
            ):
                continue

            score = (
                distance
                + missing_duration * 20.0
            )

            if shoulder_difference is not None:
                score += (
                    shoulder_difference
                    * 100.0
                )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score
                best_session = session

        return best_session

    def _connect_track(
        self,
        session,
        track_id,
        representation,
        timestamp
    ):
        track_id = int(
            track_id
        )

        # Defensive check: one track cannot own two sessions.
        other_session_id = (
            self.track_to_session.get(
                track_id
            )
        )

        if (
            other_session_id is not None
            and other_session_id
            != session.session_id
        ):
            other_session = self.sessions.get(
                other_session_id
            )

            if other_session is not None:
                other_session.current_track_id = (
                    None
                )

            self.track_to_session.pop(
                track_id,
                None
            )

        old_track_id = (
            session.current_track_id
        )

        if (
            old_track_id is not None
            and old_track_id != track_id
        ):
            self.track_to_session.pop(
                old_track_id,
                None
            )

        self._release_pending(
            session
        )

        self.track_to_session[
            track_id
        ] = session.session_id

        self._update_session(
            session,
            track_id,
            representation,
            timestamp
        )

    def _create_session(
        self,
        track_id,
        representation,
        timestamp
    ):
        session = VisitorSession(
            session_id=self.next_session_id,
            current_track_id=int(
                track_id
            ),
            created_at=timestamp,
            last_seen=timestamp,
            last_position=self._normalise_point(
                representation.get(
                    "position"
                )
            ),
            last_shoulder_width=self._normalise_width(
                representation.get(
                    "shoulder_width"
                )
            ),
            status="active"
        )

        self.sessions[
            session.session_id
        ] = session

        self.track_to_session[
            int(track_id)
        ] = session.session_id

        self.next_session_id += 1

        print(
            "New visitor session |",
            "session_id:",
            session.session_id,
            "| track_id:",
            track_id
        )

        return {
            "session_id": session.session_id,
            "new_session": True,
            "reconnected": False,
            "session": session
        }

    def _update_session(
        self,
        session,
        track_id,
        representation,
        timestamp
    ):
        session.current_track_id = int(
            track_id
        )

        session.last_seen = timestamp
        session.status = "active"

        position = self._normalise_point(
            representation.get(
                "position"
            )
        )

        shoulder_width = self._normalise_width(
            representation.get(
                "shoulder_width"
            )
        )

        if position is not None:
            session.last_position = position

        if shoulder_width is not None:
            session.last_shoulder_width = (
                shoulder_width
            )

    def _close_session(
        self,
        session,
        missing_duration,
        events
    ):
        if session.current_track_id is not None:
            self.track_to_session.pop(
                session.current_track_id,
                None
            )

        self._release_pending(
            session
        )

        session.current_track_id = None
        session.status = "closed"

        event = {
            "event": "session_closed",
            "session_id": session.session_id,
            "missing_duration": round(
                missing_duration,
                3
            )
        }

        events.append(
            event
        )

        print(
            "Visitor session closed |",
            event
        )

    def _release_pending(
        self,
        session
    ):
        pending_track_id = (
            session.pending_track_id
        )

        if pending_track_id is not None:
            self.pending_track_to_session.pop(
                pending_track_id,
                None
            )

        self._clear_pending(
            session
        )

    @staticmethod
    def _clear_pending(
        session
    ):
        session.pending_track_id = None
        session.pending_since = None
        session.pending_last_seen = None

    @staticmethod
    def _normalise_point(
        point
    ):
        if point is None:
            return None

        return (
            float(
                point[0]
            ),
            float(
                point[1]
            )
        )

    @staticmethod
    def _normalise_width(
        width
    ):
        if width is None:
            return None

        width = float(
            width
        )

        return (
            width
            if width > 0
            else None
        )

    @staticmethod
    def _distance(
        point_a,
        point_b
    ):
        if (
            point_a is None
            or point_b is None
        ):
            return None

        return math.hypot(
            float(
                point_b[0]
            )
            - float(
                point_a[0]
            ),
            float(
                point_b[1]
            )
            - float(
                point_a[1]
            )
        )

    @staticmethod
    def _shoulder_ratio_difference(
        width_a,
        width_b
    ):
        if (
            width_a is None
            or width_b is None
            or width_a <= 0
            or width_b <= 0
        ):
            return None

        return abs(
            float(
                width_a
            )
            - float(
                width_b
            )
        ) / max(
            float(
                width_a
            ),
            float(
                width_b
            )
        )