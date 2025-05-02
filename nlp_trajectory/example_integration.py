#!/usr/bin/env python3

import time
import numpy as np
import os
import sys
from typing import List, Tuple

# Ensure path includes the current directory
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Import the NLP interface
from jackal_nlp_interface import get_trajectory, set_position, reset_position

# Example wheel command model (to be replaced with the real model when not testing this script)
class WheelCommandModel:
    """
    Example model that converts trajectory points to wheel commands.
    """
    
    def __init__(self, wheel_base: float = 0.4, control_freq: float = 50.0):
        self.wheel_base = wheel_base
        self.control_freq = control_freq
        self.time_step = 1.0 / control_freq
        
        # Internal state
        self.trajectory = []
        self.current_index = 0
        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        
        print(f"Wheel command model initialized (control freq: {control_freq} Hz)")
    
    def set_trajectory(self, trajectory: List[List[float]]) -> None:
        self.trajectory = trajectory
        self.current_index = 0
        print(f"New trajectory set with {len(trajectory)} points")
    
    def compute_wheel_commands(self, current_point: List[float], target_point: List[float]) -> Tuple[float, float]:
        # Extract positions
        x1, y1, theta1 = current_point
        x2, y2, theta2 = target_point
        
        # Compute required linear and angular velocities
        dx = x2 - x1
        dy = y2 - y1
        
        # Simple approach: compute straight-line distance and angle difference
        linear_distance = np.sqrt(dx*dx + dy*dy)
        linear_velocity = linear_distance / self.time_step
        
        # Angular velocity (handle angle wrapping)
        angular_diff = theta2 - theta1
        while angular_diff > np.pi:
            angular_diff -= 2 * np.pi
        while angular_diff < -np.pi:
            angular_diff += 2 * np.pi
        
        angular_velocity = angular_diff / self.time_step
        
        # Convert to wheel velocities (differential drive kinematics)
        left_wheel = (linear_velocity - angular_velocity * self.wheel_base / 2)
        right_wheel = (linear_velocity + angular_velocity * self.wheel_base / 2)
        
        return left_wheel, right_wheel
    
    def update(self) -> Tuple[float, float]:
        # Check if we have a trajectory and haven't reached the end
        if not self.trajectory or self.current_index >= len(self.trajectory) - 1:
            return 0.0, 0.0  # No movement if no trajectory or at end
        
        # Get current and target points
        current_point = self.trajectory[self.current_index]
        target_point = self.trajectory[self.current_index + 1]
        
        # Compute wheel commands
        left_wheel, right_wheel = self.compute_wheel_commands(current_point, target_point)
        
        # Move to next point in trajectory
        self.current_index += 1
        
        # Update current position
        self.current_position = self.trajectory[self.current_index]
        
        return left_wheel, right_wheel


def run_command_demo(command: str, duration: float = 5.0) -> None:
    print("\n" + "="*50)
    print(f"Running command: '{command}'")
    print("="*50)

    reset_position()
    
    # Get trajectory from NLP
    trajectory = get_trajectory(command)
    print(f"Generated trajectory with {len(trajectory)} points")
    
    # Create wheel command model
    model = WheelCommandModel(control_freq=50.0)  # 50 Hz control loop
    
    # Set the trajectory
    model.set_trajectory(trajectory)
    
    # Run fixed-frequency control loop
    start_time = time.time()
    elapsed_time = 0.0
    
    print("\nStarting control loop...")
    while elapsed_time < duration:
        loop_start = time.time()
        
        # Get wheel commands
        left_wheel, right_wheel = model.update()
        
        # In a real system, these would be sent to the robot, here I just have them print occasionally
        if model.current_index % 10 == 0 or model.current_index >= len(trajectory) - 1:
            print(f"Time: {elapsed_time:.2f}s, Position: {model.current_position}, " 
                  f"Wheel commands: L={left_wheel:.2f}, R={right_wheel:.2f}")
        
        # Check if we've reached the end of the trajectory
        if model.current_index >= len(trajectory) - 1:
            print("Reached end of trajectory")
            break
        
        # Sleep to maintain fixed frequency
        elapsed = time.time() - loop_start
        if elapsed < model.time_step:
            time.sleep(model.time_step - elapsed)
        
        elapsed_time = time.time() - start_time
    
    print("Demo completed")


def main():

    print("Jackal NLP + Wheel Command Integration Demo")
    print("This demonstrates the pipeline from natural language to wheel commands")
    
    # Test a variety of commands
    commands = [
        "move forward 1 meter",
        "turn left 90 degrees",
        "turn right 45 degrees",
        "go to position (1, 1)"
    ]
    
    # Run each command for the demo
    for command in commands:
        run_command_demo(command)
    
    print("\nAll demonstrations completed")


if __name__ == "__main__":
    main()
