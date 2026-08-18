"""
judgement.py

Belief Engine.

Judgement does not classify the viewer.
It calculates interpretive confidence from selected
behavioural evidence.

Belief is recalculated from evidence on every frame.
It does not increase simply because more frames have passed.

Judgement recognises two equally valid routes through which
behaviour may be interpreted as heightened interest:

1. sustained attention
   - remaining in front of the work
   - moving slowly
   - accumulating evidence within a limited area

2. repeated engagement
   - revisiting previous positions
   - changing direction
   - moving back and forth within the same session

Judgement also determines whether the accumulated evidence
has enough authority to enter one of three interpretation stages:

- probe
- commit
- rewrite

Interpretation only translates this stage into an intent.

Judgement also identifies the dominant behavioural component
that currently shapes the interpretation grammar.
"""


class Judgement:

    def __init__(self):
        self.belief = 0.0
        self.stage = None
        self.dominant_component = None

        # The machine must observe enough selected evidence
        # before it is allowed to form an interpretation.
        self.minimum_evidence_count = 60

        # Belief thresholds for interpretive authority.
        self.probe_threshold = 0.70
        self.commit_threshold = 0.82
        self.rewrite_threshold = 0.92

        # Stage progression memory.
        #
        # Rewrite should not depend only on reaching an unusually high
        # instantaneous belief. Once Commit has been reached, additional
        # admitted evidence can justify later reinterpretation.
        self.highest_stage_reached = None
        self.commit_evidence_count = None
        self.rewrite_evidence_gap = 35
        self.rewrite_minimum_belief = 0.78

        # Recent micro movement acts only as a credibility modifier.
        # It does not define whether the visitor appears interested.
        self.minimum_motion_multiplier = 0.80

        # Designed normalisation thresholds.
        self.thresholds = {
            "dwell_seconds": 3.0,
            "slow_speed_pixels_per_second": 120.0,
            "revisit_count": 120.0,
            "direction_changes": 30.0,
            "micro_motion_pixels": 2.5
        }

        # Designed route weights.
        #
        # Either route may independently create strong belief.
        self.sustained_attention_weights = {
            "dwell": 0.45,
            "slow_movement": 0.25,
            "density": 0.20,
            "occupancy": 0.10
        }

        self.repeated_engagement_weights = {
            "revisit": 0.45,
            "trajectory_instability": 0.35,
            "density": 0.20
        }

        self.components = {}

    def update(
        self,
        evidence,
        selected_evidence,
        dense_area=None,
        closed_area=None
    ):
        """
        Recalculates belief, interpretation stage, and dominant
        behavioural component from current evidence.

        Selection has already decided which evidence is admitted.
        Judgement uses the selected evidence count as part of its
        authority condition.
        """

        if not selected_evidence:
            self.clear()
            return

        evidence_count = len(selected_evidence)

        # ------------------------------------------
        # Behavioural evidence
        # ------------------------------------------

        dwell_score = self._normalise(
            evidence.dwell_time(),
            self.thresholds["dwell_seconds"]
        )

        average_speed = evidence.average_speed()

        slow_movement_score = (
            1.0
            - self._normalise(
                average_speed,
                self.thresholds[
                    "slow_speed_pixels_per_second"
                ]
            )
        )

        trajectory = evidence.trajectory()

        revisit_score = self._normalise(
            trajectory.get("revisit", 0),
            self.thresholds["revisit_count"]
        )

        direction_changes = trajectory.get(
            "direction_changes",
            0
        )

        trajectory_instability_score = (
            self._normalise(
                direction_changes,
                self.thresholds["direction_changes"]
            )
        )

        density_score = 0.0

        if dense_area is not None:
            density_score = self._clamp(
                dense_area.get("density", 0.0)
            )

        occupancy_score = 0.0

        if closed_area is not None:
            occupancy_score = self._clamp(
                closed_area.get("occupancy", 0.0)
            )

        # ------------------------------------------
        # Sustained attention route
        # ------------------------------------------

        sustained_attention_components = {
            "dwell": dwell_score,
            "slow_movement": slow_movement_score,
            "density": density_score,
            "occupancy": occupancy_score
        }

        sustained_attention_score = sum(
            sustained_attention_components[name]
            * weight
            for name, weight
            in self.sustained_attention_weights.items()
        )

        sustained_attention_score = self._clamp(
            sustained_attention_score
        )

        # ------------------------------------------
        # Repeated engagement route
        # ------------------------------------------

        repeated_engagement_components = {
            "revisit": revisit_score,
            "trajectory_instability":
                trajectory_instability_score,
            "density": density_score
        }

        repeated_engagement_score = sum(
            repeated_engagement_components[name]
            * weight
            for name, weight
            in self.repeated_engagement_weights.items()
        )

        repeated_engagement_score = self._clamp(
            repeated_engagement_score
        )

        # Either sustained attention or repeated engagement
        # can independently justify strong interpretive belief.
        behavioural_belief = max(
            sustained_attention_score,
            repeated_engagement_score
        )

        # ------------------------------------------
        # Temporal credibility
        # ------------------------------------------

        micro_motion = evidence.micro_motion()

        if micro_motion is None:
            average_micro_motion = None
            micro_motion_score = 0.0
        else:
            average_micro_motion = micro_motion[
                "average_motion"
            ]

            micro_motion_score = self._normalise(
                average_micro_motion,
                self.thresholds["micro_motion_pixels"]
            )

        motion_gate = (
            self.minimum_motion_multiplier
            + (
                1.0
                - self.minimum_motion_multiplier
            )
            * micro_motion_score
        )

        self.belief = (
            behavioural_belief
            * motion_gate
        )

        self.belief = self._clamp(
            self.belief
        )

        self.components = {
            "dwell": dwell_score,
            "slow_movement": slow_movement_score,
            "revisit": revisit_score,
            "density": density_score,
            "occupancy": occupancy_score,
            "trajectory_instability":
                trajectory_instability_score,
            "sustained_attention":
                sustained_attention_score,
            "repeated_engagement":
                repeated_engagement_score,
            "micro_motion": micro_motion_score,
            "motion_gate": motion_gate,
            "behavioural_belief": behavioural_belief,
            "evidence_count": evidence_count,
            "average_speed": average_speed
        }

        if average_micro_motion is not None:
            self.components[
                "average_micro_motion"
            ] = average_micro_motion

        # ------------------------------------------
        # Dominant behavioural component
        # ------------------------------------------

        self.dominant_component = (
            self._select_dominant_component()
        )

        # ------------------------------------------
        # Interpretation stage
        # ------------------------------------------

        self.stage = self._determine_stage(
            evidence_count=evidence_count,
            belief=self.belief
        )

    def _determine_stage(
        self,
        evidence_count,
        belief
    ):
        """
        Determines interpretive stage with monotonic progression.

        Probe and Commit still require their belief thresholds.
        After Commit, Rewrite may be reached through either:
        - very high current belief, or
        - enough additional admitted evidence while belief remains credible.

        A temporary belief drop does not demote an already reached stage.
        Session closure and Evidence reset call clear(), which resets this
        progression memory.
        """

        if evidence_count < self.minimum_evidence_count:
            return None

        candidate_stage = None

        if belief >= self.rewrite_threshold:
            candidate_stage = "rewrite"

        elif belief >= self.commit_threshold:
            candidate_stage = "commit"

        elif belief >= self.probe_threshold:
            candidate_stage = "probe"

        # Record the first Commit evidence count.
        if (
            candidate_stage in ("commit", "rewrite")
            and self.commit_evidence_count is None
        ):
            self.commit_evidence_count = evidence_count

        # Additional evidence after Commit can produce Rewrite even when
        # belief does not reach the old 0.92 threshold.
        if (
            self.commit_evidence_count is not None
            and belief >= self.rewrite_minimum_belief
            and (
                evidence_count
                - self.commit_evidence_count
            ) >= self.rewrite_evidence_gap
        ):
            candidate_stage = "rewrite"

        stage_rank = {
            None: 0,
            "probe": 1,
            "commit": 2,
            "rewrite": 3
        }

        if (
            stage_rank[candidate_stage]
            > stage_rank[self.highest_stage_reached]
        ):
            self.highest_stage_reached = candidate_stage

        return self.highest_stage_reached

    def _select_dominant_component(self):
        """
        Selects the strongest granular behavioural signal from
        the components already calculated by Judgement.

        Route scores such as sustained_attention and
        repeated_engagement determine belief, but the granular
        signal determines the interpretation grammar.
        """

        if not self.components:
            return None

        signals = {
            "dwell": self.components.get(
                "dwell",
                0.0
            ),

            "revisit": self.components.get(
                "revisit",
                0.0
            ),

            "density": self.components.get(
                "density",
                0.0
            ),

            "occupancy": self.components.get(
                "occupancy",
                0.0
            ),

            "trajectory_instability":
                self.components.get(
                    "trajectory_instability",
                    0.0
                )
        }

        dominant_name = max(
            signals,
            key=signals.get
        )

        return {
            "name": dominant_name,
            "score": self._clamp(
                signals[dominant_name]
            ),
            "signals": {
                name: self._clamp(value)
                for name, value
                in signals.items()
            }
        }

    def clear(self):
        """
        Removes belief and interpretive authority.
        """

        self.belief = 0.0
        self.stage = None
        self.components = {}
        self.dominant_component = None
        self.highest_stage_reached = None
        self.commit_evidence_count = None

    def has_interpretive_authority(self):
        """
        Authority exists only when Judgement has produced
        an active interpretation stage.
        """

        return self.stage is not None

    def get_stage(self):
        return self.stage

    def get_belief(self):
        return self.belief

    def get_components(self):
        return self.components.copy()

    def get_dominant_component(self):
        """
        Returns the strongest behavioural signal currently
        shaping the machine's judgement.
        """

        if self.dominant_component is None:
            return None

        return {
            "name": self.dominant_component["name"],
            "score": self.dominant_component["score"],
            "signals": self.dominant_component[
                "signals"
            ].copy()
        }

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