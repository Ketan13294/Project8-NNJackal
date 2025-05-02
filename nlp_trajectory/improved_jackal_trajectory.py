#!/usr/bin/env python3

"""
Improved Jackal NLP Trajectory Generator

This module converts natural language commands into center of mass (COM) trajectories
for a Jackal robot using a language model. It takes commands like "move forward one meter"
and outputs a sequence of position waypoints that define the robot's trajectory.
"""

import numpy as np
import re
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import pipeline
from pathlib import Path

class JackalTrajectoryGenerator:
    """
    A class that generates COM trajectories from natural language commands.
    """

    def __init__(self, model_path='distilbert-base-uncased'):
        """
        Initialize the Jackal Trajectory Generator.
        
        Args:
            model_path: Path to the language model or model identifier from Hugging Face
        """
        # Robot parameters - these can be tuned based on actual robot capabilities
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 0.5  # rad/s
        self.control_freq = 100  # Hz - for trajectory discretization only
        self.time_step = 1.0 / self.control_freq
        
        # NLP setup
        print(f"Loading language model from {model_path}...")
        self.model_path = model_path
        
        # Use a text classification pipeline
        try:
            self.nlp = pipeline(
                "text-classification", 
                model=model_path, 
                return_all_scores=True
            )
        except Exception as e:
            print(f"Warning: Error initializing NLP model, using pattern matching only: {e}")
            self.nlp = None
        
        # Command patterns and parsing
        self._setup_command_patterns()
        
        # Initial state
        self.current_position = np.array([0.0, 0.0, 0.0])  # x, y, theta (in world frame)
        
        print("Jackal Trajectory Generator initialized and ready for commands")
    
    def _setup_command_patterns(self):
        """Set up regex patterns and command templates for parsing."""
        # Command templates with their corresponding trajectory generation functions
        self.command_templates = {
            "move forward": self._generate_forward_trajectory,
            "go forward": self._generate_forward_trajectory,
            "move ahead": self._generate_forward_trajectory,
            "move backward": self._generate_backward_trajectory,
            "go backward": self._generate_backward_trajectory,
            "move back": self._generate_backward_trajectory,
            "turn left": self._generate_left_turn_trajectory,
            "rotate left": self._generate_left_turn_trajectory,
            "turn right": self._generate_right_turn_trajectory,
            "rotate right": self._generate_right_turn_trajectory,
            "rotate": self._generate_rotation_trajectory,
            "stop": self._generate_stop_trajectory,
            "go to": self._generate_goto_trajectory,
        }
        
        # Regex patterns for extracting values
        self.distance_pattern = r"(\d+\.?\d*)\s*(m|meters|meter|cm|centimeters|centimeter)"
        self.angle_pattern = r"(\d+\.?\d*)\s*(degrees|degree|deg|°|rad|radians|radian)"
        self.position_pattern = r"position\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"
    
    def parse_command(self, command_text: str) -> Tuple[callable, Dict[str, Any]]:
        """
        Parse a natural language command and return the corresponding trajectory
        generation function and parameters.
        
        Args:
            command_text: The natural language command text
            
        Returns:
            Function to generate trajectory and parameters dictionary
        """
        command_text = command_text.lower().strip()
        
        # Find the matching command template
        matched_template = None
        for template in self.command_templates:
            if template in command_text:
                matched_template = template
                break
        
        if matched_template is None:
            print(f"Unknown command: {command_text}")
            return self._generate_stop_trajectory, {}
        
        # Get the corresponding trajectory generation function
        trajectory_func = self.command_templates[matched_template]
        
        # Extract parameters based on the command type
        params = {}
        
        # Extract distance if present
        distance_match = re.search(self.distance_pattern, command_text)
        if distance_match:
            value = float(distance_match.group(1))
            unit = distance_match.group(2)
            
            # Convert to meters if necessary
            if unit.startswith('c'):  # centimeters
                value /= 100.0
                
            params['distance'] = value
        
        # Extract angle if present
        angle_match = re.search(self.angle_pattern, command_text)
        if angle_match:
            value = float(angle_match.group(1))
            unit = angle_match.group(2)
            
            # Convert to radians if necessary
            if unit in ['degrees', 'degree', 'deg', '°']:
                value = value * math.pi / 180.0
                
            params['angle'] = value
        
        # Extract position if present
        position_match = re.search(self.position_pattern, command_text)
        if position_match:
            x = float(position_match.group(1))
            y = float(position_match.group(2))
            params['position'] = (x, y)
        
        return trajectory_func, params
    
    def interpret_nlp_command(self, command_text: str) -> Tuple[callable, Dict[str, Any]]:
        """
        Use NLP to interpret natural language commands and extract parameters.
        
        Args:
            command_text: The natural language command
            
        Returns:
            Function to generate trajectory and parameters dictionary
        """
        # First try direct pattern matching
        trajectory_func, params = self.parse_command(command_text)
        
        # If no parameters were extracted but we matched a command, try NLP
        if not params and trajectory_func != self._generate_stop_trajectory:
            print("Using alternative parsing for command parameters...")
            
            # For demonstration, inferring parameters from the command text
            if "forward" in command_text or "ahead" in command_text:
                # Check for numeric values without explicit units
                number_match = re.search(r'(\d+\.?\d*)', command_text)
                if number_match:
                    params["distance"] = float(number_match.group(1))
                else:
                    # Default to 1 meter if no specific distance
                    params["distance"] = 1.0
            elif "back" in command_text or "backward" in command_text:
                # Check for numeric values without explicit units
                number_match = re.search(r'(\d+\.?\d*)', command_text)
                if number_match:
                    params["distance"] = float(number_match.group(1))
                else:
                    params["distance"] = 1.0
            elif "left" in command_text:
                # Check for numeric values without explicit units
                number_match = re.search(r'(\d+\.?\d*)', command_text)
                if number_match:
                    params["angle"] = float(number_match.group(1)) * math.pi / 180.0
                else:
                    params["angle"] = math.pi/2  # 90 degrees
            elif "right" in command_text:
                # Check for numeric values without explicit units
                number_match = re.search(r'(\d+\.?\d*)', command_text)
                if number_match:
                    params["angle"] = float(number_match.group(1)) * math.pi / 180.0
                else:
                    params["angle"] = math.pi/2  # 90 degrees
        
        return trajectory_func, params
    
    def _generate_forward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a forward trajectory for the robot's COM.
        
        Args:
            params: Dictionary with 'distance' key
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        distance = params.get('distance', 1.0)  # Default to 1m if not specified
        print(f"Generating forward trajectory for {distance} meters")
        
        # Calculate how many steps needed
        duration = distance / (0.75 * self.max_linear_speed)  # Adjusted for trapezoidal profile
        steps = int(duration * self.control_freq)
        steps = max(steps, 2)  # Ensure at least start and end points
        
        # Create a smooth trajectory
        trajectory = []
        current_pos = self.current_position.copy()
        
        # Initial orientation determines the direction of movement
        heading = current_pos[2]  # Current orientation in radians
        direction_vector = np.array([math.cos(heading), math.sin(heading)])
        
        for i in range(steps + 1):  # +1 to include the endpoint
            # Position along the path using a smooth acceleration/deceleration profile
            normalized_time = i / steps
            
            # Trapezoidal position profile (derived from velocity profile)
            if normalized_time < 0.25:
                # Accelerating: quadratic position increase
                progress = 2 * normalized_time**2
            elif normalized_time < 0.75:
                # Constant velocity: linear position increase
                progress = 0.125 + (normalized_time - 0.25) * 0.75
            else:
                # Decelerating: quadratic position increase slowing down
                progress = 0.5 + 0.5 * normalized_time - 0.125 * (4 * (1 - normalized_time))**2
            
            # Calculate position
            position = current_pos.copy()
            position[0] += direction_vector[0] * distance * progress
            position[1] += direction_vector[1] * distance * progress
            
            trajectory.append(position)
        
        # Update current position to the end of the trajectory
        self.current_position = trajectory[-1].copy()
        
        return trajectory
    
    def _generate_backward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a backward trajectory for the robot's COM.
        
        Args:
            params: Dictionary with 'distance' key
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        # For backward motion, we simply negate the distance
        backward_params = params.copy()
        backward_params['distance'] = -params.get('distance', 1.0)
        
        return self._generate_forward_trajectory(backward_params)
    
    def _generate_left_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a left turn trajectory for the robot's COM.
        
        Args:
            params: Dictionary with 'angle' key (in radians)
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        angle = params.get('angle', math.pi/2)  # Default to 90 degrees if not specified
        print(f"Generating left turn trajectory for {angle * 180/math.pi} degrees")
        
        # Calculate how many steps needed
        duration = angle / self.max_angular_speed
        steps = int(duration * self.control_freq)
        steps = max(steps, 2)  # Ensure at least start and end points
        
        # Create a smooth trajectory
        trajectory = []
        current_pos = self.current_position.copy()
        
        for i in range(steps + 1):  # +1 to include the endpoint
            # Normalized time from 0 to 1
            normalized_time = i / steps
            
            # Smooth angular profile (ease in, ease out)
            if normalized_time < 0.3:
                # Accelerating: quadratic angle increase
                progress = normalized_time**2 / 0.09  # (0.3^2 = 0.09)
            elif normalized_time < 0.7:
                # Constant angular velocity
                progress = (normalized_time - 0.3) / 0.4 * 0.7 + 0.3
            else:
                # Decelerating: quadratic angle increase slowing down
                progress = 1.0 - ((1.0 - normalized_time)**2 / 0.09)
                progress = min(progress, 1.0)  # Ensure we don't exceed 1.0
            
            # Calculate position (only theta changes for a turn in place)
            position = current_pos.copy()
            position[2] = current_pos[2] + angle * progress
            
            # Normalize theta to [-pi, pi]
            position[2] = (position[2] + math.pi) % (2 * math.pi) - math.pi
            
            trajectory.append(position)
        
        # Update current position to the end of the trajectory
        self.current_position = trajectory[-1].copy()
        
        return trajectory
    
    def _generate_right_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a right turn trajectory for the robot's COM.
        
        Args:
            params: Dictionary with 'angle' key (in radians)
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        # For right turn, we simply negate the angle
        right_params = params.copy()
        right_params['angle'] = -params.get('angle', math.pi/2)
        
        return self._generate_left_turn_trajectory(right_params)
    
    def _generate_rotation_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a rotation trajectory for the robot's COM.
        
        Args:
            params: Dictionary with 'angle' key (in radians)
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        angle = params.get('angle', 0.0)
        
        if angle >= 0:
            return self._generate_left_turn_trajectory(params)
        else:
            return self._generate_right_turn_trajectory({'angle': abs(angle)})
    
    def _generate_stop_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a stop trajectory (just stay in place).
            
        Returns:
            List containing just the current position
        """
        print("Generating stop trajectory")
        return [self.current_position.copy()]
    
    def _generate_goto_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Generate a trajectory to go to a specific position.
        
        Args:
            params: Dictionary with 'position' tuple (x, y)
            
        Returns:
            List of position arrays [x, y, theta] for the trajectory
        """
        if 'position' not in params:
            print("No position specified, staying in place")
            return self._generate_stop_trajectory({})
        
        target_x, target_y = params['position']
        print(f"Generating trajectory to position ({target_x}, {target_y})")
        
        # Current position
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        current_theta = self.current_position[2]
        
        # 1. Calculate vector to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Early exit if we're already at the target
        if distance < 0.01:  # Within 1cm
            return [self.current_position.copy()]
        
        # 2. Calculate angle to target
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - current_theta
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2*math.pi
        while angle_diff < -math.pi:
            angle_diff += 2*math.pi
        
        # 3. Generate trajectory: first rotate to face target
        rotation_params = {'angle': angle_diff}
        trajectory = self._generate_rotation_trajectory(rotation_params)
        
        # 4. Then move forward to target
        forward_params = {'distance': distance}
        
        # We need to exclude the first point of the forward trajectory since
        # it's the same as the last point of the rotation trajectory
        forward_trajectory = self._generate_forward_trajectory(forward_params)[1:]
        
        # Combine the trajectories
        trajectory.extend(forward_trajectory)
        
        return trajectory
    
    def generate_trajectory(self, command_text: str) -> List[np.ndarray]:
        """
        Generate a COM trajectory from a natural language command.
        
        Args:
            command_text: The natural language command text
            
        Returns:
            List of position arrays [x, y, theta] representing the trajectory
        """
        # Parse the command
        trajectory_func, params = self.interpret_nlp_command(command_text)
        
        # Generate trajectory
        return trajectory_func(params)
    
    def set_position(self, position: np.ndarray) -> None:
        """
        Set the current position of the robot.
        
        Args:
            position: Array [x, y, theta] representing position and orientation
        """
        self.current_position = position.copy()
    
    def reset_position(self) -> None:
        """Reset the current position to the origin."""
        self.current_position = np.array([0.0, 0.0, 0.0])


def trajectory_to_list(trajectory: List[np.ndarray]) -> List[List[float]]:
    """
    Convert a trajectory of numpy arrays to a list of lists.
    
    Args:
        trajectory: List of position arrays [x, y, theta]
        
    Returns:
        List of position lists [x, y, theta]
    """
    return [point.tolist() for point in trajectory]


def get_trajectory(command: str) -> List[List[float]]:
    """
    Generate a trajectory from a natural language command.
    
    Args:
        command: Natural language command (e.g., "move forward 1 meter")
        
    Returns:
        List of position lists [x, y, theta]
    """
    generator = JackalTrajectoryGenerator()
    trajectory = generator.generate_trajectory(command)
    return trajectory_to_list(trajectory)


if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Jackal Trajectory Generator')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                      help='Path to language model or model identifier')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization (requires matplotlib)')
    parser.add_argument('--output-dir', type=str, default=os.path.expanduser('~'),
                      help='Directory to save visualization images')
    args = parser.parse_args()
    
    # Initialize the trajectory generator
    generator = JackalTrajectoryGenerator(model_path=args.model)
    
    # Setup visualization if enabled
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            def visualize_trajectory(trajectory, command_text=None):

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                x = [point[0] for point in trajectory]
                y = [point[1] for point in trajectory]
                
                x_min, x_max = min(x) - 0.5, max(x) + 0.5
                y_min, y_max = min(y) - 0.5, max(y) + 0.5
                
                grid_size = max(x_max - x_min, y_max - y_min, 2.0)
                x_min, x_max = min(x_min, -grid_size/2), max(x_max, grid_size/2)
                y_min, y_max = min(y_min, -grid_size/2), max(y_max, grid_size/2)

                ax.set_aspect('equal')

                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
                major_ticks = np.arange(math.floor(min(x_min, y_min)), math.ceil(max(x_max, y_max)) + 1, 0.5)
                minor_ticks = np.arange(math.floor(min(x_min, y_min)), math.ceil(max(x_max, y_max)) + 1, 0.1)
                
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)
                ax.set_yticks(major_ticks)
                ax.set_yticks(minor_ticks, minor=True)

                ax.grid(which='major', color='gray', linestyle='-', alpha=0.7)
                ax.grid(which='minor', color='lightgray', linestyle=':', alpha=0.4)
                
                # Plot trajectory with different segments for better visualization
                segments = []
                current_segment = [trajectory[0]]
                
                for i in range(1, len(trajectory)):
                    prev_pos = trajectory[i-1]
                    curr_pos = trajectory[i]
                    
                    # Check if this is a significant change in heading (turning) or position is same (in-place rotation)
                    angle_change = abs(prev_pos[2] - curr_pos[2])
                    position_same = (abs(prev_pos[0] - curr_pos[0]) < 0.01) and (abs(prev_pos[1] - curr_pos[1]) < 0.01)
                    
                    if angle_change > 0.1 or position_same:  # More than ~5.7 degrees change or in-place rotation
                        # End current segment and start a new one
                        if id(current_segment[-1]) != id(curr_pos):
                            current_segment.append(curr_pos)
                        segments.append(current_segment)
                        current_segment = [curr_pos]
                    else:
                        current_segment.append(curr_pos)
                
                # Add the last segment
                if current_segment:
                    segments.append(current_segment)
                
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
                
                for i, segment in enumerate(segments):
                    seg_x = [point[0] for point in segment]
                    seg_y = [point[1] for point in segment]
                    color = colors[i % len(colors)]
                    ax.plot(seg_x, seg_y, marker='o', markersize=3, 
                            color=color, label=f'Segment {i+1}' if i < 7 else None)
                    
                    if len(seg_x) > 1:
                        total_dist = 0
                        for j in range(1, len(segment)):
                            dx = segment[j][0] - segment[j-1][0]
                            dy = segment[j][1] - segment[j-1][1]
                            total_dist += math.sqrt(dx**2 + dy**2)

                        mid_idx = len(segment) // 2
                        mid_x, mid_y = segment[mid_idx][0], segment[mid_idx][1]
                        
                        if total_dist > 0.05:  # Only show distance for non-trivial segments
                            ax.text(mid_x, mid_y, f"{total_dist:.2f}m", 
                                    color=color, fontweight='bold', 
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Plot robot orientations with arrows at key points
                num_arrows = min(len(trajectory), 20)
                indices = [int(i * len(trajectory) / num_arrows) for i in range(num_arrows)]
                
                # Add arrows at the start and end of each segment for clarity
                for segment in segments:
                    if len(segment) > 1:
                        # Can't use direct list.index() for numpy arrays, need to find indices manually
                        for i, point in enumerate(trajectory):
                            if np.array_equal(point, segment[0]):
                                indices.append(i)
                                break
                        for i, point in enumerate(trajectory):
                            if np.array_equal(point, segment[-1]):
                                indices.append(i)
                                break
                
                # Remove duplicates and sort
                indices = sorted(list(set(indices)))
                
                for i in indices:
                    point = trajectory[i]
                    ax.arrow(point[0], point[1], 
                             0.2 * math.cos(point[2]), 0.2 * math.sin(point[2]),  # Larger arrows
                             head_width=0.08, head_length=0.12, fc='black', ec='black', alpha=0.7)

                ax.plot(x[0], y[0], 'go', markersize=12, label='Start')
                ax.plot(x[-1], y[-1], 'ro', markersize=12, label='End')

                ax.set_xlabel('X (meters)', fontsize=12)
                ax.set_ylabel('Y (meters)', fontsize=12)
                
                if command_text:
                    ax.set_title(f'Robot Trajectory: "{command_text}"', fontsize=14)
                else:
                    ax.set_title('Robot Trajectory', fontsize=14)
                
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best')

                start_pos = trajectory[0]
                end_pos = trajectory[-1]
                ax.annotate(f'Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}rad)', 
                           (start_pos[0], start_pos[1]), xytext=(-20, 20),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.8))
                
                ax.annotate(f'End: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}rad)', 
                           (end_pos[0], end_pos[1]), xytext=(20, -20),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.8))
                
                if command_text:
                    safe_command = command_text.replace(' ', '_').replace('/', '_')
                    safe_command = ''.join(c for c in safe_command if c.isalnum() or c == '_')
                    filename = f"trajectory_{safe_command}_{timestamp}.png"
                else:
                    filename = f"trajectory_{timestamp}.png"
                
                filepath = os.path.join(args.output_dir, filename)
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Visualization saved to: {filepath}")
                return filepath
                
        except ImportError:
            print("Warning: Visualization requires matplotlib. Run 'pip install matplotlib'")
            args.visualize = False
    
    # Interactive loop
    print("\nJackal Trajectory Generator")
    print("Type 'exit' or 'quit' to end the program")
    print("Type 'reset' to reset the current position to the origin")
    print("Type 'position x y theta' to set the current position")
    if args.visualize:
        print(f"Visualizations will be saved to: {args.output_dir}")
    print("\nExample commands:")
    print("  move forward 2 meters")
    print("  turn left 90 degrees")
    print("  go to position (1.5, 2.0)")
    
    while True:
        try:
            command = input("\nEnter command: ")
            if command.lower() in ['exit', 'quit']:
                break
            
            # Special commands
            if command.lower() == 'reset':
                generator.reset_position()
                print("Position reset to origin")
                continue
            
            if command.lower().startswith('position '):
                try:
                    parts = command.split()[1:]
                    if len(parts) >= 3:
                        x = float(parts[0])
                        y = float(parts[1])
                        theta = float(parts[2])
                        generator.set_position(np.array([x, y, theta]))
                        print(f"Position set to ({x}, {y}, {theta})")
                    else:
                        print("Invalid position format. Use 'position x y theta'")
                except ValueError:
                    print("Invalid position values. Use numeric values for x, y, and theta")
                continue
            
            # Generate trajectory for the command
            trajectory = generator.generate_trajectory(command)
            
            # Print summary of the generated trajectory
            print(f"Generated trajectory with {len(trajectory)} points")
            print(f"Start: {trajectory[0]}")
            print(f"End: {trajectory[-1]}")
            
            # Visualize if enabled
            if args.visualize:
                visualize_trajectory(trajectory, command)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting Jackal Trajectory Generator")
