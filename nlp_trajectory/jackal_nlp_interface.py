#!/usr/bin/env python3

"""
Jackal NLP Interface for Integration

This module provides a clean, simple interface for converting natural language commands
into COM trajectories for the Jackal robot. It's designed to be easily integrated with
the second model that will handle the timing and wheel commands.

Usage:
  from jackal_nlp_interface import get_trajectory
  
  # Get trajectory points from natural language command
  trajectory = get_trajectory("move forward 1 meter")
  
  # Pass to second model
  # trajectory format: [[x1, y1, theta1], [x2, y2, theta2], ...]
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any

# Ensure path includes the current directory
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Import the trajectory generator
from improved_jackal_trajectory import JackalTrajectoryGenerator

# Avoids reloading the model
_nlp_generator = None

def get_trajectory(command: str) -> List[List[float]]:
    """
    Simple function wrapper that takes a natural language command and 
    returns a list of COM trajectory waypoints.
    
    Args:
        command: Natural language command (e.g., "move forward 1 meter")
    
    Returns:
        List of position lists [x, y, theta]
    """
    global _nlp_generator
    
    # Initialize generator if not already done
    if _nlp_generator is None:
        _nlp_generator = JackalTrajectoryGenerator()
    
    # Generate the trajectory
    trajectory = _nlp_generator.generate_trajectory(command)
    
    # Convert to list format
    return [point.tolist() for point in trajectory]

def reset_position() -> None:
    """Reset the current position to the origin [0, 0, 0]."""
    global _nlp_generator
    
    # Initialize generator if not already done
    if _nlp_generator is None:
        _nlp_generator = JackalTrajectoryGenerator()
    
    _nlp_generator.reset_position()

def set_position(x: float, y: float, theta: float) -> None:
    """
    Set the current position of the robot.
    
    Args:
        x: X coordinate
        y: Y coordinate
        theta: Orientation in radians
    """
    global _nlp_generator
    
    # Initialize generator if not already done
    if _nlp_generator is None:
        _nlp_generator = JackalTrajectoryGenerator()
    
    _nlp_generator.set_position(np.array([x, y, theta]))

# Example of how the function can be used
if __name__ == "__main__":
    # Simple test to demonstrate usage
    test_commands = [
        "move forward 1 meter",
        "turn left 90 degrees",
        "go to position (1, 1)",
        "turn right 45 degrees",
        "stop"
    ]
    
    # Test each command
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        trajectory = get_trajectory(command)
        print(f"Trajectory has {len(trajectory)} points")
        print(f"First point: {trajectory[0]}")
        print(f"Last point: {trajectory[-1]}")
        
        # Detailed output for a few points
        if len(trajectory) > 2:
            print("\nSample points:")
            step = max(1, len(trajectory) // 5)  # Show about 5 points
            for i in range(0, len(trajectory), step):
                print(f"Point {i}: {trajectory[i]}")
