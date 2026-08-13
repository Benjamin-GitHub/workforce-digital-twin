import numpy as np


class ActivityClassifier:
    def __init__(
        self,
        bending_angle_threshold=35.0,
        walking_velocity_threshold=0.08,
        walking_ankle_threshold=0.06,
        carrying_wrist_hip_threshold=0.50,
        carrying_max_torso_angle=20.0,
        carrying_min_velocity=0.05,
        carrying_max_wrist_motion=0.18,
        idle_velocity_threshold=0.015,
        idle_ankle_threshold=0.02,
        idle_min_frames=15,
        recent_window_size=10,
    ):
        self.bending_angle_threshold = bending_angle_threshold

        self.walking_velocity_threshold = (
            walking_velocity_threshold
        )

        self.walking_ankle_threshold = (
            walking_ankle_threshold
        )

        self.carrying_wrist_hip_threshold = (
            carrying_wrist_hip_threshold
        )

        self.carrying_max_torso_angle = (
            carrying_max_torso_angle
        )

        self.carrying_min_velocity = (
            carrying_min_velocity
        )

        self.carrying_max_wrist_motion = (
            carrying_max_wrist_motion
        )

        self.idle_velocity_threshold = (
            idle_velocity_threshold
        )

        self.idle_ankle_threshold = (
            idle_ankle_threshold
        )

        self.idle_min_frames = idle_min_frames

        self.recent_window_size = (
            recent_window_size
        )

    def _normalised_body_velocity(self, window):
        """
        Calculate average body-centre movement
        between consecutive frames.

        Movement is normalised using torso length
        so that distance from the camera has less
        effect on the measurement.
        """

        if len(window) < 2:
            return 0.0

        velocities = []

        for previous, current in zip(
            window[:-1],
            window[1:]
        ):
            dx = (
                current["body_center_x"]
                - previous["body_center_x"]
            )

            dy = (
                current["body_center_y"]
                - previous["body_center_y"]
            )

            movement = np.sqrt(
                dx ** 2 + dy ** 2
            )

            torso = current.get(
                "torso_length",
                1.0
            )

            if torso > 0:
                movement /= torso

            velocities.append(
                movement
            )

        if not velocities:
            return 0.0

        return float(
            np.mean(velocities)
        )

    def _ankle_motion(self, window):
        """
        Calculate average movement of the
        left and right ankles between frames.

        This helps detect walking even when
        the person's whole-body position changes
        only slightly.
        """

        if len(window) < 2:
            return 0.0

        movements = []

        for previous, current in zip(
            window[:-1],
            window[1:]
        ):
            torso = current.get(
                "torso_length",
                1.0
            )

            if torso is None or torso <= 0:
                continue

            for side in (
                "left",
                "right",
            ):
                previous_x = previous.get(
                    f"{side}_ankle_x"
                )

                previous_y = previous.get(
                    f"{side}_ankle_y"
                )

                current_x = current.get(
                    f"{side}_ankle_x"
                )

                current_y = current.get(
                    f"{side}_ankle_y"
                )

                if any(
                    value is None
                    for value in (
                        previous_x,
                        previous_y,
                        current_x,
                        current_y,
                    )
                ):
                    continue

                dx = (
                    current_x
                    - previous_x
                )

                dy = (
                    current_y
                    - previous_y
                )

                movement = np.sqrt(
                    dx ** 2 + dy ** 2
                )

                movement /= torso

                movements.append(
                    movement
                )

        if not movements:
            return 0.0

        return float(
            np.mean(movements)
        )

    def _wrist_motion(self, window):
        """
        Calculate average wrist movement between
        consecutive frames, normalised by torso length.

        High values indicate active upper-body/hand
        movement, useful for material-handling inference.
        """

        if len(window) < 2:
            return 0.0

        movements = []

        for previous, current in zip(
            window[:-1],
            window[1:]
        ):
            torso = current.get(
                "torso_length",
                1.0
            )

            if torso is None or torso <= 0:
                continue

            for side in (
                "left",
                "right",
            ):
                previous_x = previous.get(
                    f"{side}_wrist_x"
                )

                previous_y = previous.get(
                    f"{side}_wrist_y"
                )

                current_x = current.get(
                    f"{side}_wrist_x"
                )

                current_y = current.get(
                    f"{side}_wrist_y"
                )

                if any(
                    value is None
                    for value in (
                        previous_x,
                        previous_y,
                        current_x,
                        current_y,
                    )
                ):
                    continue

                dx = current_x - previous_x
                dy = current_y - previous_y

                movement = np.sqrt(
                    dx ** 2 + dy ** 2
                )

                movement /= torso

                movements.append(
                    movement
                )

        if not movements:
            return 0.0

        return float(
            np.mean(movements)
        )

    def _mean_wrist_hip_distance(
        self,
        window,
    ):
        """
        Mean normalised distance of visible wrists
        from the hip centre.
        """

        values = []

        for frame in window:
            for side in (
                "left",
                "right",
            ):
                value = frame.get(
                    f"{side}_wrist_hip_distance"
                )

                if value is not None:
                    values.append(value)

        if not values:
            return None

        return float(
            np.mean(values)
        )

    def _mean_torso_angle(self, window):
        values = [
            frame["torso_angle"]
            for frame in window
            if frame.get(
                "torso_angle"
            ) is not None
        ]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def _mean_knee_angle(self, window):
        knee_angles = []

        for frame in window:
            left_angle = frame.get(
                "left_knee_angle"
            )

            right_angle = frame.get(
                "right_knee_angle"
            )

            if left_angle is not None:
                knee_angles.append(
                    left_angle
                )

            if right_angle is not None:
                knee_angles.append(
                    right_angle
                )

        if not knee_angles:
            return None

        return float(
            np.mean(knee_angles)
        )

    def classify(self, window):
        """
        Classify activity using recent
        temporal pose features.
        """
        
        #window = window[-10:]
        
        if not window:
            return {
                "activity": "unknown",
                "confidence": 0.0,
                "velocity": 0.0,
                "ankle_motion": 0.0,
                "wrist_motion": 0.0,
                "wrist_hip_distance": None,
                "torso_angle": 0.0,
                "knee_angle": None,
            }

        # Only use the most recent frames
        # for current activity classification.
        recent_window = window[
            -self.recent_window_size:
        ]

        torso_angle = (
            self._mean_torso_angle(
                recent_window
            )
        )

        velocity = (
            self._normalised_body_velocity(
                recent_window
            )
        )

        ankle_motion = (
            self._ankle_motion(
                recent_window
            )
        )

        knee_angle = (
            self._mean_knee_angle(
                recent_window
            )
        )

        wrist_motion = (
            self._wrist_motion(
                recent_window
            )
        )

        wrist_hip_distance = (
            self._mean_wrist_hip_distance(
                recent_window
            )
        )

        # ---------------------------------
        # 1. BENDING
        # ---------------------------------

        if (
            torso_angle
            >= self.bending_angle_threshold
        ):
            confidence = min(
                1.0,
                torso_angle / 60.0
            )

            return {
                "activity": "bending",
                "confidence": confidence,
                "torso_angle": torso_angle,
                "velocity": velocity,
                "ankle_motion": ankle_motion,
                "wrist_motion": wrist_motion,
                "wrist_hip_distance": wrist_hip_distance,
                "knee_angle": knee_angle,
            }

        # ---------------------------------
        # 2. CARRYING
        # ---------------------------------

        walking_by_body = (
            velocity
            >= self.walking_velocity_threshold
        )

        walking_by_ankles = (
            ankle_motion
            >= self.walking_ankle_threshold
        )

        locomotion_detected = (
            walking_by_body
            or walking_by_ankles
        )

        carrying_hand_position = (
            wrist_hip_distance is not None
            and wrist_hip_distance
            >= self.carrying_wrist_hip_threshold
        )

        carrying_body_motion = (
            velocity
            >= self.carrying_min_velocity
        )

        carrying_stable_hands = (
            wrist_motion
            <= self.carrying_max_wrist_motion
        )

        carrying_upright_posture = (
            torso_angle
            <= self.carrying_max_torso_angle
        )

        if (
            locomotion_detected
            and carrying_body_motion
            and carrying_hand_position
            and carrying_stable_hands
            and carrying_upright_posture
        ):
            distance_score = (
                wrist_hip_distance
                / max(
                    self.carrying_wrist_hip_threshold,
                    1e-6,
                )
            )

            confidence = min(
                0.90,
                0.60
                + max(
                    0.0,
                    distance_score - 1.0,
                ) * 0.30,
            )

            return {
                "activity": "carrying",
                "confidence": confidence,
                "torso_angle": torso_angle,
                "velocity": velocity,
                "ankle_motion": ankle_motion,
                "wrist_motion": wrist_motion,
                "wrist_hip_distance": wrist_hip_distance,
                "knee_angle": knee_angle,
            }

        # ---------------------------------
        # 3. WALKING
        # ---------------------------------

        if locomotion_detected:
            body_score = (
                velocity
                / max(
                    self.walking_velocity_threshold,
                    1e-6,
                )
            )

            ankle_score = (
                ankle_motion
                / max(
                    self.walking_ankle_threshold,
                    1e-6,
                )
            )

            confidence = min(
                1.0,
                max(
                    body_score,
                    ankle_score,
                ) / 3.0,
            )

            # Prevent extremely low
            # walking confidence.
            confidence = max(
                confidence,
                0.60
            )

            return {
                "activity": "walking",
                "confidence": confidence,
                "torso_angle": torso_angle,
                "velocity": velocity,
                "ankle_motion": ankle_motion,
                "wrist_motion": wrist_motion,
                "wrist_hip_distance": wrist_hip_distance,
                "knee_angle": knee_angle,
            }

        # ---------------------------------
        # 4. IDLE
        # ---------------------------------

        enough_frames = (
            len(window)
            >= self.idle_min_frames
        )

        low_body_movement = (
            velocity
            <= self.idle_velocity_threshold
        )

        low_ankle_movement = (
            ankle_motion
            <= self.idle_ankle_threshold
        )

        if (
            enough_frames
            and low_body_movement
            and low_ankle_movement
        ):
            return {
                "activity": "idle",
                "confidence": 0.85,
                "torso_angle": torso_angle,
                "velocity": velocity,
                "ankle_motion": ankle_motion,
                "wrist_motion": wrist_motion,
                "wrist_hip_distance": wrist_hip_distance,
                "knee_angle": knee_angle,
            }

        # ---------------------------------
        # 5. STANDING
        # ---------------------------------

        if (
            knee_angle is None
            or knee_angle > 145
        ):
            return {
                "activity": "standing",
                "confidence": 0.75,
                "torso_angle": torso_angle,
                "velocity": velocity,
                "ankle_motion": ankle_motion,
                "wrist_motion": wrist_motion,
                "wrist_hip_distance": wrist_hip_distance,
                "knee_angle": knee_angle,
            }

        # ---------------------------------
        # FALLBACK
        # ---------------------------------

        return {
            "activity": "unknown",
            "confidence": 0.30,
            "torso_angle": torso_angle,
            "velocity": velocity,
            "ankle_motion": ankle_motion,
            "wrist_motion": wrist_motion,
            "wrist_hip_distance": wrist_hip_distance,
            "knee_angle": knee_angle,
        }
