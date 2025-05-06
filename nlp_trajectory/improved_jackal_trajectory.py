#!/usr/bin/env python3

"""
Improved Jackal NLP Trajectory Generator v2

This module converts natural language commands into center-of-mass (COM)
trajectories for a Jackal robot by using a zero‑shot classification LLM
(fine‑tuned on MNLI) with exactly one canonical intent label per trajectory function.
"""

import re
import math
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from transformers import pipeline

class JackalTrajectoryGenerator:
    """
    Generates COM trajectories from natural language commands
    using a zero‑shot LLM with one intent per function.
    """

    def __init__(self, model_path: str = "./intent_model"):
        # Robot motion parameters
        self.max_linear_speed = 0.5    # m/s
        self.max_angular_speed = 0.5   # rad/s
        self.control_freq     = 100    # Hz
        self.time_step        = 1.0 / self.control_freq

        # Canonical intents (one per trajectory function)
        self.intents = [
            "move forward",
            "move backward",
            "turn left",
            "turn right",
            "rotate",
            "go to",
            "stop"
        ]

        # Load zero‑shot classifier (must be an NLI model)
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.nlp = pipeline(
                task="zero-shot-classification",
                model=model_path,
                device=device,
            )
            print(f"Loaded zero‑shot model '{model_path}' on device {device}")
        except Exception as e:
            print(f"Failed to load zero‑shot model '{model_path}': {e}")
            self.nlp = None

        # Map each intent to its trajectory builder
        self.command_templates: Dict[str, Any] = {
            "move forward":  self._generate_forward_trajectory,
            "move backward": self._generate_backward_trajectory,
            "turn left":     self._generate_left_turn_trajectory,
            "turn right":    self._generate_right_turn_trajectory,
            "rotate":        self._generate_rotation_trajectory,
            "go to":         self._generate_goto_trajectory,
            "stop":          self._generate_stop_trajectory,
        }

        # Parameter‑extraction regex patterns
        self.distance_pattern = r"(-?\d+\.?\d*)\s*(m|meter|meters|cm|centimeter|centimeters)\b"
        self.angle_pattern    = r"(-?\d+\.?\d*)\s*(°|deg|degree|degrees|rad|radian|radians)\b"
        self.position_pattern = r"position\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"

        # Initial robot COM
        self.current_position = np.array([0.0, 0.0, 0.0])

    def parse_command(self, command_text: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Classify the intent via zero‑shot (MNLI), then extract numeric parameters.
        Returns: (trajectory_function, params_dict).
        """
        text = command_text.lower().strip()

        # Intent classification
        if self.nlp:
            result = self.nlp(text, candidate_labels=self.intents)
            intent = result["labels"][0]
            score  = result["scores"][0]
            print(f"Intent: '{intent}'  (score={score:.2f})")
            func = self.command_templates[intent]
        else:
            # Fallback: substring match on canonical keywords
            func = self._generate_stop_trajectory
            for key, fn in self.command_templates.items():
                if key in text:
                    func = fn
                    break

        # Parameter extraction
        params: Dict[str, Any] = {}

        # Distance in meters
        dm = re.search(self.distance_pattern, text)
        if dm:
            val, unit = float(dm.group(1)), dm.group(2)
            if unit.startswith("c"):
                val /= 100.0
            params["distance"] = val

        # Angle in radians
        am = re.search(self.angle_pattern, text)
        if am:
            val, unit = float(am.group(1)), am.group(2)
            if unit in ("°","deg","degree","degrees"):
                val = val * math.pi / 180.0
            params["angle"] = val

        # Absolute position
        pm = re.search(self.position_pattern, text)
        if pm:
            x, y = float(pm.group(1)), float(pm.group(2))
            params["position"] = (x, y)

        return func, params

    def generate_trajectory(self, command_text: str) -> List[np.ndarray]:
        """
        High‑level entrypoint: parse the command, then run the
        matched trajectory function with extracted params.
        """
        func, params = self.parse_command(command_text)
        return func(params)

    def set_position(self, position: np.ndarray) -> None:
        """Set the robot’s current COM."""
        self.current_position = position.copy()

    def reset_position(self) -> None:
        """Reset the robot’s COM to the origin."""
        self.current_position = np.array([0.0, 0.0, 0.0])

    # Trajectory builders

    def _generate_forward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        distance = params.get("distance", 1.0)
        steps    = max(int(distance / (self.max_linear_speed * self.time_step)), 1)
        traj, pos0 = [], self.current_position.copy()
        for i in range(steps + 1):
            frac = i / steps
            delta = frac * distance
            p = pos0.copy()
            p[0] += delta * math.cos(pos0[2])
            p[1] += delta * math.sin(pos0[2])
            traj.append(p)
        self.current_position = traj[-1].copy()
        return traj

    def _generate_backward_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        back = params.copy()
        back["distance"] = -params.get("distance", 1.0)
        return self._generate_forward_trajectory(back)

    def _generate_left_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        angle = params.get("angle", math.pi/2)
        steps = max(int(abs(angle) / (self.max_angular_speed * self.time_step)), 1)
        traj = []
        theta0 = self.current_position[2]
        for i in range(steps + 1):
            frac = i / steps
            theta = theta0 + angle * frac
            p = self.current_position.copy()
            p[2] = (theta + math.pi) % (2*math.pi) - math.pi
            traj.append(p)
        self.current_position = traj[-1].copy()
        return traj

    def _generate_right_turn_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        right = params.copy()
        right["angle"] = -params.get("angle", math.pi/2)
        return self._generate_left_turn_trajectory(right)

    def _generate_rotation_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        angle = params.get("angle", 0.0)
        if angle >= 0:
            return self._generate_left_turn_trajectory(params)
        else:
            return self._generate_right_turn_trajectory({"angle": abs(angle)})

    def _generate_goto_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        if "position" not in params:
            return self._generate_stop_trajectory({})
        tx, ty = params["position"]
        cx, cy, ct = self.current_position
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        target_ang = math.atan2(dy, dx)
        diff = (target_ang - ct + math.pi) % (2*math.pi) - math.pi

        traj = self._generate_rotation_trajectory({"angle": diff})
        fwd = self._generate_forward_trajectory({"distance": dist})[1:]
        traj.extend(fwd)
        return traj

    def _generate_stop_trajectory(self, params: Dict[str, Any]) -> List[np.ndarray]:
        print("Generating stop trajectory")
        return [self.current_position.copy()]

# Helper to convert numpy arrays to lists
def trajectory_to_list(traj: List[np.ndarray]) -> List[List[float]]:
    return [p.tolist() for p in traj]

# Demo if run from CLI
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Jackal NLP Trajectory Generator (zero‑shot demo)"
    )
    parser.add_argument("--model", default="facebook/bart-large-mnli",
                        help="HF model ID or local path for zero‑shot")
    args = parser.parse_args()

    gen = JackalTrajectoryGenerator(model_path=args.model)
    examples = [
        "move forward 1 meter",
        "turn left 90 degrees",
        "go to position (1,1)",
        "please stop"
    ]
    for cmd in examples:
        print(f"\n> Command: {cmd!r}")
        traj = gen.generate_trajectory(cmd)
        print(f"  → {len(traj)} points, start={traj[0]}, end={traj[-1]}")

