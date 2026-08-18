"""
config.py

Global settings for Power of Interpretation.
"""

# Camera
CAMERA_ID = 0

# AxiDraw
# Hardware port for the current exhibition computer.
AXIDRAW_PORT = "COM5"

# Trajectory
MAX_TRAJECTORY_LENGTH = 1000
MIN_DISTANCE_BETWEEN_POINTS = 3

# Visual test settings
POINT_COLOR = (0, 0, 255)
POINT_RADIUS = 6

TRAJECTORY_COLOR = (0, 0, 0)
TRAJECTORY_THICKNESS = 2

WINDOW_NAME = "Power of Interpretation v0.2 - Trajectory"

# -----------------------
# Evidence Parameters
# -----------------------

DWELL_RADIUS = 80

SPEED_AVERAGE_LAST_N = 30

TRAJECTORY_LAST_N = 120
TRAJECTORY_REVISIT_RADIUS = 50
TRAJECTORY_LOOP_RADIUS = 80
TRAJECTORY_LOOP_MIN_DISTANCE = 300

CLOSED_AREA_OCCUPANCY_THRESHOLD = 0.4

# -----------------------
# Selection Parameters
# -----------------------

# Technical reliability gate.
SELECTION_CONFIDENCE_THRESHOLD = 0.5

# Selection waits for a small behavioural sample before
# assigning interpretive value.
SELECTION_MIN_EVIDENCE_COUNT = 12

# Recent evidence remains privileged, but recency alone
# cannot make evidence pass Selection.
SELECTION_RECENT_N = 80
SELECTION_RECENT_BONUS = 0.10

# Final minimum weight required for evidence to be selected.
SELECTION_WEIGHT_THRESHOLD = 0.45

# Sustained attention:
# remaining in front of the work, moving slowly,
# and maintaining presence within a limited area.
SELECTION_DWELL_SECONDS = 3.0
SELECTION_SLOW_SPEED_MAX = 120.0

SELECTION_SUSTAINED_DWELL_WEIGHT = 0.65
SELECTION_SUSTAINED_SLOW_WEIGHT = 0.35

# Repeated engagement:
# revisiting positions and changing direction within
# the same visitor session.
SELECTION_REVISIT_COUNT = 60.0
SELECTION_DIRECTION_CHANGE_COUNT = 18.0

SELECTION_REPEATED_REVISIT_WEIGHT = 0.65
SELECTION_REPEATED_DIRECTION_WEIGHT = 0.35