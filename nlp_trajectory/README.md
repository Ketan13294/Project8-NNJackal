# NLP Trajectory Generator

This module converts natural-language commands into center-of-mass (COM) trajectories for the Clearpath Jackal robot, using a zero-shot LLM for intent classification.

## Installation

1. Clone or enter your project’s root directory.  
2. Install dependencies:  
  
   python==3.12.0
   gymnasium==0.28.1
   mujoco==3.3.1
   scipy==1.15.2
   numpy==1.26.4
   torch>=1.13.1
   transformers>=4.30.0

## Overview

The NLP Trajectory Generator takes commands like:

- `"move forward 2 meters"`  
- `"turn left 90 degrees"`  
- `"go to position (1.5, 2.0)"`  

and produces a smooth, discretized list of waypoints `[[x, y, θ], …]` suitable for downstream control.

## Usage

### Simple integration

```python
from nlp_trajectory.jackal_nlp_interface import get_trajectory

# Get a trajectory for “move forward”:
trajectory = get_trajectory("move forward 1 meter")
# → [[0.0,0.0,0.0], [0.05,0.0,0.0], …, [1.0,0.0,0.0]]
```

### Command-line demo

```bash
python nlp_trajectory/example_integration.py
```

### Swapping in a custom model

By default, the generator uses Hugging Face’s **distilbert-base-uncased** zero-shot classifier. To use your own model (local or remote), edit the initialization in `jackal_nlp_interface.py`:

```diff
- _nlp_generator = JackalTrajectoryGenerator()
+ _nlp_generator = JackalTrajectoryGenerator(model_path="path/to/your-model")
```

## Supported Commands

- **Move**:  
  - `move forward 2 meters`  
  - `go back 50 cm`  
- **Turn**:  
  - `turn left 90 degrees`  
  - `rotate right 45 degrees`  
- **Goto**:  
  - `go to position (1.5, 2.0)`  
- **Stop**:  
  - `stop`

## Notes

- Numerical parameters (distance in m/cm, angles in °/rad) are extracted via regex currently.  
- Intent classification is performed by the zero-shot LLM, with a substring-match fallback if loading fails.  
- Customize `control_freq`, `max_linear_speed`, etc., by editing `improved_jackal_trajectory.py`.  

---

*Last updated: May 2025*
