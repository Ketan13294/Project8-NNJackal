# NLP Trajectory Generator

This module converts natural-language commands into center-of-mass (COM) trajectories for the Clearpath Jackal robot, using a zero-shot LLM for intent classification.

## Installation

1. Clone or enter your project’s root directory.  
2. Install dependencies:  
   ```bash
   pip install numpy torch transformers datasets scikit-learn pandas matplotlib gymnasium mujoco scipy
   ```

## Directory Structure

```
.
├── improved_jackal_trajectory.py    # Core trajectory generator class
├── jackal_nlp_interface.py          # Simple wrapper API for trajectory functions
├── example_integration.py           # Demo: NL → trajectory → wheel commands
├── train_intent_model.py            # Fine‑tune HF NLI model for 7‑way intents
├── plot_intent_metrics.py           # Plot CSV logs of training metrics
├── train.csv                        # Labeled Traning Set
├── validation.csv                   # Validation set
└── README.md                        # This file
```

## Data Files

- **train.csv**: Labeled training set for intent classification, with columns `text` and `label` (0–6).  
- **validation.csv**: Validation set for evaluation, same format.

## Overview

The NLP Trajectory Generator takes commands like:

- `"move forward 2 meters"`  
- `"turn left 90 degrees"`  
- `"go to position (1.5, 2.0)"`  

and produces a smooth, discretized list of waypoints `[[x, y, θ], …]` suitable for downstream control.

## Usage

### Intent Model Training

```bash
python train_intent_model.py \
  --train_csv train.csv \
  --val_csv validation.csv \
  --output_dir intent_model \
  --epochs 16 \
  --batch_size 64
```

### Plotting Training Metrics

```bash
python plot_intent_metrics.py
```

### Simple Integration

```python
from jackal_nlp_interface import get_trajectory

trajectory = get_trajectory("move forward 1 meter")
```

### Command-Line Demo

```bash
python example_integration.py
```

### Custom Zero-Shot Model

To use a custom model, edit the initialization in `jackal_nlp_interface.py`:

```diff
- _nlp_generator = JackalTrajectoryGenerator()
+ _nlp_generator = JackalTrajectoryGenerator(model_path="path/to/your-model")
```

## Supported Commands

- **Move**  
  - `move forward 2 meters`  
  - `go back 50 cm`  
- **Turn**  
  - `turn left 90 degrees`  
  - `rotate right 45 degrees`  
- **Goto**  
  - `go to position (1.5, 2.0)`  
- **Stop**  
  - `stop`

## Notes

- Numerical parameters (distances, angles) are extracted via regex.  
- Customize parameters need to match the robot, like `control_freq`, `max_linear_speed`, etc., at the top of `improved_jackal_trajectory.py`.  

---

*Last updated: May 2025*
