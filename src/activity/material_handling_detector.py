from collections import deque

import numpy as np


class MaterialHandlingDetector:
    """Detect pose-inferred material handling over a temporal window."""

    def __init__(
        self,
        window_size=33,
        min_frames=20,
        min_wrist_motion=0.10,
        min_bending_ratio=0.12,
        min_active_state_ratio=0.35,
        max_walking_ratio=0.75,
        stationary_max_velocity=0.05,
        stationary_min_wrist_motion=0.05,
        stationary_min_wrist_hip_distance=0.50,
        stationary_min_standing_idle_ratio=0.60,
        stationary_max_bending_ratio=0.10,
    ):
        self.window_size = window_size
        self.min_frames = min_frames

        self.min_wrist_motion = min_wrist_motion
        self.min_bending_ratio = min_bending_ratio
        self.min_active_state_ratio = min_active_state_ratio
        self.max_walking_ratio = max_walking_ratio
        
        self.stationary_max_velocity = stationary_max_velocity
        self.stationary_min_wrist_motion = (
            stationary_min_wrist_motion
        )
        self.stationary_min_wrist_hip_distance = (
            stationary_min_wrist_hip_distance
        )
        self.stationary_min_standing_idle_ratio = (
            stationary_min_standing_idle_ratio
        )
        self.stationary_max_bending_ratio = (
            stationary_max_bending_ratio
        )

        self.history = {}

    def _get_history(self, track_id):
        if track_id not in self.history:
            self.history[track_id] = deque(
                maxlen=self.window_size
            )

        return self.history[track_id]

    def remove(self, track_id):
        self.history.pop(track_id, None)

    @staticmethod
    def _mean_valid(values):
        valid_values = [
            float(value)
            for value in values
            if value is not None
            and np.isfinite(value)
        ]

        if not valid_values:
            return 0.0

        return float(np.mean(valid_values))

    def update(
        self,
        track_id,
        activity,
        wrist_motion,
        wrist_hip_distance,
        torso_angle,
        velocity,
    ):
        window = self._get_history(track_id)

        window.append({
            "activity": activity,
            "wrist_motion": wrist_motion,
            "wrist_hip_distance": wrist_hip_distance,
            "torso_angle": torso_angle,
            "velocity": velocity,
        })

        if len(window) < self.min_frames:
            return {
                "detected": False,
                "pickup_detected": False,
                "stationary_detected": False,
                "confidence": 0.0,
                "bending_ratio": 0.0,
                "walking_ratio": 0.0,
                "carrying_ratio": 0.0,
                "active_state_ratio": 0.0,
                "standing_idle_ratio": 0.0,
                "mean_wrist_motion": 0.0,
                "mean_wrist_hip_distance": 0.0,
                "mean_velocity": 0.0,
                "distinct_states": 0,
            }

        activities = [
            item["activity"]
            for item in window
        ]

        bending_ratio = (
            activities.count("bending")
            / len(activities)
        )

        walking_ratio = (
            activities.count("walking")
            / len(activities)
        )

        carrying_ratio = (
            activities.count("carrying")
            / len(activities)
        )

        standing_count = activities.count("standing")
        idle_count = activities.count("idle")

        standing_idle_ratio = (
            (standing_count + idle_count)
            / len(activities)
        )

        active_states = {
            "bending",
            "carrying",
            "walking",
        }

        active_state_ratio = (
            sum(
                activity in active_states
                for activity in activities
            )
            / len(activities)
        )

        mean_wrist_motion = self._mean_valid(
            item["wrist_motion"]
            for item in window
        )

        mean_wrist_hip_distance = self._mean_valid(
            item["wrist_hip_distance"]
            for item in window
        )

        mean_velocity = self._mean_valid(
            item["velocity"]
            for item in window
        )

        meaningful_states = {
            "bending",
            "carrying",
            "walking",
            "standing",
            "idle",
        }

        distinct_states = len({
            activity
            for activity in activities
            if activity in meaningful_states
        })

        pickup_handling = (
            mean_wrist_motion
            >= self.min_wrist_motion
            and bending_ratio
            >= self.min_bending_ratio
            and active_state_ratio
            >= self.min_active_state_ratio
            and walking_ratio
            <= self.max_walking_ratio
            and distinct_states >= 2
        )

        stationary_handling = (
            mean_velocity
            <= self.stationary_max_velocity
            and mean_wrist_motion
            >= self.stationary_min_wrist_motion
            and mean_wrist_hip_distance
            >= self.stationary_min_wrist_hip_distance
            and standing_idle_ratio
            >= self.stationary_min_standing_idle_ratio
            and bending_ratio
            <= self.stationary_max_bending_ratio
        )

        material_handling = (
            pickup_handling
            or stationary_handling
        )

        if pickup_handling:
            bending_score = min(
                1.0,
                bending_ratio
                / max(
                    self.min_bending_ratio,
                    1e-6,
                ),
            )

            wrist_score = min(
                1.0,
                mean_wrist_motion
                / max(
                    self.min_wrist_motion,
                    1e-6,
                ),
            )

            activity_score = min(
                1.0,
                active_state_ratio
                / max(
                    self.min_active_state_ratio,
                    1e-6,
                ),
            )

            confidence = float(
                np.mean([
                    bending_score,
                    wrist_score,
                    activity_score,
                ])
            )

        elif stationary_handling:
            velocity_score = min(
                1.0,
                self.stationary_max_velocity
                / max(
                    mean_velocity,
                    1e-6,
                ),
            )

            wrist_score = min(
                1.0,
                mean_wrist_motion
                / max(
                    self.stationary_min_wrist_motion,
                    1e-6,
                ),
            )

            wrist_position_score = min(
                1.0,
                mean_wrist_hip_distance
                / max(
                    self.stationary_min_wrist_hip_distance,
                    1e-6,
                ),
            )

            posture_score = min(
                1.0,
                standing_idle_ratio
                / max(
                    self.stationary_min_standing_idle_ratio,
                    1e-6,
                ),
            )

            confidence = float(
                np.mean([
                    velocity_score,
                    wrist_score,
                    wrist_position_score,
                    posture_score,
                ])
            )
        else:
            confidence = 0.0

        return {
            "detected": material_handling,
            "pickup_detected": pickup_handling,
            "stationary_detected": stationary_handling,
            "confidence": confidence,
            "bending_ratio": bending_ratio,
            "walking_ratio": walking_ratio,
            "carrying_ratio": carrying_ratio,
            "active_state_ratio": active_state_ratio,
            "standing_idle_ratio": standing_idle_ratio,
            "mean_wrist_motion": mean_wrist_motion,
            "mean_wrist_hip_distance": mean_wrist_hip_distance,
            "mean_velocity": mean_velocity,
            "distinct_states": distinct_states,
        }
