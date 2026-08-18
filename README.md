# Power Emerges Through Interpretive Authority

An interactive drawing installation exploring how interpretation gains authority through observation, selection and judgement.

The work observes visitors through a camera-based tracking system and reduces each validated body to a limited spatial representation. Behavioural evidence is accumulated over time and selectively weighted according to a rule-based framework designed by the artist.

The system privileges behaviors that may be interpreted as suggesting sustained attention or repeated engagement. From this selected evidence, it forms an interpretive confidence and translates that judgement into a drawing intent.

The resulting interpretation is materialized physically through an AxiDraw plotter.

The system does not attempt to objectively identify or describe a person. Instead, the work examines how selected evidence, designed thresholds and interpretive rules can produce a judgement that comes to stand in for someone.

> What gives an interpretation the confidence to stand in for a person?

## System

The installation follows this general process:

`Observation → Reduction → Evidence → Selection → Judgement → Interpretation → Visitor Arbitration → Drawing → AxiDraw`

### Observation
Detects and tracks visitors using YOLO Pose and ByteTrack, while maintaining temporary visitor sessions.

### Reduction
Reduces a validated body observation to a selected spatial point.

### Evidence
Accumulates behavioral evidence including dwell time, movement, trajectory and spatial repetition.

### Selection
Selects which evidence is given interpretive value. Selection is intentionally non-neutral and follows artist-defined rules.

### Judgement
Calculates interpretive confidence from selected behavioral evidence and determines when the system has enough authority to form an interpretation.

### Interpretation
Translates judgement into an interpretive intent, including its stage, position, direction and drawing grammar.

### Visitor Arbitration
Determines which visitor's interpretation is allowed to control the shared drawing process when multiple visitors are present.

### Drawing
Materializes interpretation through rule-based drawing stages including commit, rewrite and later intervention.

### AxiDraw
Maps digital drawing actions onto the physical drawing surface and executes them using the AxiDraw plotter.

## Main Files

- `main.py` — integrates the complete installation system
- `observation.py` — camera input, pose detection and tracking
- `visitor_validation.py` — validates credible visitor tracks
- `visitor_session.py` — maintains temporary visitor continuity
- `reduction.py` — reduces the detected body to a spatial representation
- `evidence.py` — accumulates behavioral evidence
- `selection.py` — selects evidence according to designed interpretive priorities
- `spatial.py` — analyses spatial density
- `engagement.py` — distinguishes foreground and background engagement
- `judgement.py` — calculates interpretive confidence and interpretation stages
- `interpretation.py` — produces interpretive drawing intent
- `drawing_placement.py` — places drawing actions within the drawing surface
- `drawing.py` — generates and manages the drawing grammar
- `axidraw_controller.py` — controls physical AxiDraw output
- `judgement_debug.py` — development diagnostics

## Requirements

Python 3.10 is recommended.

Core dependencies include:

- OpenCV
- NumPy
- Ultralytics YOLO
- pySerial
- AxiDraw Python API

Install the standard Python dependencies with:

```bash
pip install -r requirements.txt
