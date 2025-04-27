# NLP Trajectory Generator

This module converts natural language commands into center of mass (COM) trajectories for the Jackal robot.

## Overview

The NLP Trajectory Generator takes commands like "move forward one meter" or "turn left 90 degrees" and produces a list of waypoints that define a smooth trajectory for the robot. This component fulfills the plannerLLM role described in our project proposal.

## Files

- `improved_jackal_trajectory.py` - Core trajectory generator logic
- `jackal_nlp_interface.py` - Simple interface for integration
- `example_integration.py` - Example code showing how to use the system

## Integration

To use this with your wheel command model:

```python
from nlp_trajectory.jackal_nlp_interface import get_trajectory

# Get trajectory from natural language command
trajectory = get_trajectory("move forward 1 meter")

# trajectory is a list of [x, y, theta] waypoints
# Example: [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], ..., [1.0, 0.0, 0.0]]

# Use the trajectory with your wheel command model
for point in trajectory:
    # Process each waypoint
    x, y, theta = point
```

## Supported Commands

- **Forward/Backward Movement**: 
  - "move forward 2 meters"
  - "go back 50 cm"
  
- **Turning**:
  - "turn left 90 degrees"
  - "rotate right 45 degrees"
  
- **Going to a Position**:
  - "go to position (1.5, 2.0)"
  
- **Stopping**:
  - "stop"

## Examples

Run the example integration script to test how it works:

```bash
python ./example_integration.py
```

## Notes

- All trajectories have smooth acceleration/deceleration profiles
- For just the final position: final_position = trajectory[-1]
- Turns are implemented as rotations around the center of mass (fixed position, changing orientation)
