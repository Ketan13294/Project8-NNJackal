#!/usr/bin/env python3

"""
Improved Jackal NLP Trajectory Generator

This module converts natural language commands into center of mass (COM) trajectories
for a Jackal robot using a zero‐shot classification LLM for intent detection.
"""

import re
import math
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from transformers import pipeline
from pathlib import Path

class JackalTrajectoryGenerator:
    """
    A class that generates COM trajectories from natural language commands.
    """

    def __init__(self, model_path: str = 'distilbert-base-uncased'):
        """
        Initialize the Jackal Trajectory Generator.

        Args:
            model_path: Hugging Face model identifier or local path for zero‐shot classification
        """
        # Robot parameters
        self.max_linear_speed = 0.5      # m/s
        self.max_angular_speed = 0.5     # rad/s
        self.control_freq = 100          # Hz (for discretization)
        self.time_step = 1.0 / self.control_freq

        # Load zero‐shot NLP pipeline
        print(f"Loading zero‐shot classifier from '{model_path}'...")
        try:
            self.nlp = pipeline(
                "zero-shot-classification",
                model=model_path,
                device=0 if torch.cuda.is_available() else -1,
            )
            print("NLP model loaded successfully.")
        except Exception as e:
            print(f"Failed to load NLP model ({e}); falling back to substring matching.")
            self.nlp = None

        # Command templates & regex patterns
        self._setup_command_patterns()

        # Intents list for zero‐shot
        self.intents = list(self.command_templates.keys()) if self.nlp else None

        # Robot’s current COM
        self.current_position = np.array([0.0, 0.0, 0.0])

        print("JackalTrajectoryGenerator initialized and ready.")

    def _setup_command_patterns(self):
        # Set up mapping from textual templates to trajectory functions.
        self.command_templates: Dict[str, Any] = {
            "move forward":     self._generate_forward_trajectory,
            "go forward":       self._generate_forward_trajectory,
            "move ahead":       self._generate_forward_trajectory,
            "move backward":    self._generate_backward_trajectory,
            "go backward":      self._generate_backward_trajectory,
            "move back":        self._generate_backward_trajectory,
            "turn left":        self._generate_left_turn_trajectory,
            "rotate left":      self._generate_left_turn_trajectory,
            "turn right":       self._generate_right_turn_trajectory,
            "rotate right":     self._generate_right_turn_trajectory,
            "rotate":           self._generate_rotation_trajectory,
            "go to":            self._generate_goto_trajectory,
            "stop":             self._generate_stop_trajectory,
        }

        # Regex for extracting numeric parameters
        self.distance_pattern = r"(\d+\.?\d*)\s*(m|meters|meter|cm|centimeters|centimeter)"
        self.angle_pattern    = r"(\d+\.?\d*)\s*(degrees|degree|deg|°|rad|radians|radian)"
        self.position_pattern = r"position\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"

    def parse_command(self, command_text: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Classify intent with the LLM (or substring fallback), then extract parameters.

        Returns:
            trajectory_func: callable
            params: dict of extracted {distance, angle, position}
        """
        text = command_text.lower().strip()

        # 1) Intent detection via zero‐shot model
        if self.nlp:
            result = self.nlp(text, candidate_labels=self.intents)
            intent = result["labels"][0]
            score  = result["scores"][0]
            print(f"Intent: '{intent}' (score {score:.2f})")
            trajectory_func = self.command_templates[intent]
        else:
            # Fallback: first substring match
            for tmpl, fn in self.command_templates.items():
                if tmpl in text:
                    intent, trajectory_func = tmpl, fn
                    break
            else:
                print(f"Unknown command '{text}', defaulting to STOP.")
                intent, trajectory_func = "stop", self._generate_stop_trajectory

        # 2) Parameter extraction via regex
        params: Dict[str, Any] = {}

        # Distance in meters
        dm = re.search(self.distance_pattern, text)
        if dm:
            val, unit = float(dm.group(1)), dm.group(2)
            if unit.startswith('c'):
                val /= 100.0
            params["distance"] = val

        # Angle in radians
        am = re.search(self.angle_pattern, text)
        if am:
            val, unit = float(am.group(1)), am.group(2)
            if unit in ['degrees', 'degree', 'deg', '°']:
                val = val * math.pi / 180.0
            params["angle"] = val

        # Absolute position
        pm = re.search(self.position_pattern, text)
        if pm:
            params["position"] = (float(pm.group(1)), float(pm.group(2)))

        return trajectory_func, params

    def generate_trajectory(self, command_text: str) -> List[np.ndarray]:
        """
        Generate COM trajectory based on parsed intent + params.
        """
        func, params = self.parse_command(command_text)
        return func(params)

    def set_position(self, position: np.ndarray) -> None:
        """Set the robot’s COM for subsequent trajectory building."""
        self.current_position = position.copy()

    def reset_position(self) -> None:
        """Reset COM to [0,0,0]."""
        self.current_position = np.array([0.0, 0.0, 0.0])

    def _generate_forward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        distance = params.get("distance", 1.0)
        steps = max(int(distance / (self.max_linear_speed * self.time_step)), 1)
        trajectory, pos = [], self.current_position.copy()
        for i in range(steps + 1):
            frac = i / steps
            delta = frac * distance
            new_pos = pos.copy()
            new_pos[0] += delta * math.cos(pos[2])
            new_pos[1] += delta * math.sin(pos[2])
            trajectory.append(new_pos)
        self.current_position = trajectory[-1].copy()
        return trajectory

    def _generate_backward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        bp = params.copy()
        bp["distance"] = -params.get("distance", 1.0)
        return self._generate_forward_trajectory(bp)

    def _generate_left_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        angle = params.get("angle", math.pi / 2)
        steps = max(int(abs(angle) / (self.max_angular_speed * self.time_step)), 1)
        trajectory = []
        start_theta = self.current_position[2]
        for i in range(steps + 1):
            frac = i / steps
            theta = start_theta + angle * frac
            pos = self.current_position.copy()
            pos[2] = (theta + math.pi) % (2 * math.pi) - math.pi
            trajectory.append(pos)
        self.current_position = trajectory[-1].copy()
        return trajectory

    def _generate_right_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        rp = params.copy()
        rp["angle"] = -params.get("angle", math.pi / 2)
        return self._generate_left_turn_trajectory(rp)

    def _generate_rotation_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        angle = params.get("angle", 0.0)
        if angle >= 0:
            return self._generate_left_turn_trajectory(params)
        else:
            return self._generate_right_turn_trajectory({"angle": abs(angle)})

    def _generate_stop_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        print("Generating stop trajectory")
        return [self.current_position.copy()]

    def _generate_goto_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        if "position" not in params:
            return self._generate_stop_trajectory({})
        tx, ty = params["position"]
        # rotate then forward
        cx, cy, ct = self.current_position
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        targ_angle = math.atan2(dy, dx)
        ang_diff = (targ_angle - ct + math.pi) % (2 * math.pi) - math.pi

        traj = self._generate_rotation_trajectory({"angle": ang_diff})
        forward_traj = self._generate_forward_trajectory({"distance": dist})[1:]
        traj.extend(forward_traj)
        return traj

# Helper to convert numpy arrays to plain lists
def trajectory_to_list(trajectory: List[np.ndarray]) -> List[List[float]]:
    return [pt.tolist() for pt in trajectory]

# If run as a script, simple CLI demo
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Jackal NLP Trajectory Generator")
    parser.add_argument("--model", default="distilbert-base-uncased",
                        help="Hugging Face model ID or local path")
    args = parser.parse_args()

    gen = JackalTrajectoryGenerator(model_path=args.model)
    examples = ["move forward 1 meter", "turn left 90 degrees", "go to position (1,1)", "stop"]
    for cmd in examples:
        print(f"\nCommand: '{cmd}'")
        traj = gen.generate_trajectory(cmd)
        print(f" → Trajectory ({len(traj)} points), start={traj[0]}, end={traj[-1]}")

