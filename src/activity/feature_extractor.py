import math
import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def midpoint(p1, p2):
    return np.array([
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0
    ])


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def angle_between_points(a, b, c):
    """
    Angle ABC in degrees.
    """
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def torso_angle_from_vertical(shoulder_mid, hip_mid):
    """
    Calculates torso lean away from vertical.

    0 degrees = vertical/upright.
    Larger value = more leaning/bending.
    """
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))

    return float(angle)


def safe_point(keypoints, index):
    """
    keypoints expected as shape:
    [17, 2]
    """
    if index >= len(keypoints):
        return None

    point = keypoints[index]

    if point is None:
        return None

    x, y = point

    if np.isnan(x) or np.isnan(y):
        return None

    return np.array([float(x), float(y)])


def extract_pose_features(keypoints):
    """
    Extract useful geometric features from one person's pose.

    Returns dictionary or None if essential keypoints are missing.
    """

    ls = safe_point(keypoints, LEFT_SHOULDER)
    rs = safe_point(keypoints, RIGHT_SHOULDER)

    lh = safe_point(keypoints, LEFT_HIP)
    rh = safe_point(keypoints, RIGHT_HIP)

    lk = safe_point(keypoints, LEFT_KNEE)
    rk = safe_point(keypoints, RIGHT_KNEE)

    la = safe_point(keypoints, LEFT_ANKLE)
    ra = safe_point(keypoints, RIGHT_ANKLE)

    lw = safe_point(keypoints, LEFT_WRIST)
    rw = safe_point(keypoints, RIGHT_WRIST)

    essential = [ls, rs, lh, rh]

    if any(point is None for point in essential):
        return None

    shoulder_mid = midpoint(ls, rs)
    hip_mid = midpoint(lh, rh)

    torso_length = distance(shoulder_mid, hip_mid)

    if torso_length < 1:
        return None

    body_center = midpoint(shoulder_mid, hip_mid)

    torso_angle = torso_angle_from_vertical(
        shoulder_mid,
        hip_mid
    )

    left_knee_angle = None
    right_knee_angle = None

    if lh is not None and lk is not None and la is not None:
        left_knee_angle = angle_between_points(
            lh,
            lk,
            la
        )

    if rh is not None and rk is not None and ra is not None:
        right_knee_angle = angle_between_points(
            rh,
            rk,
            ra
        )

    ankle_distance = None

    if la is not None and ra is not None:
        ankle_distance = distance(la, ra) / torso_length

    wrist_to_hip_left = None
    wrist_to_hip_right = None

    if lw is not None:
        wrist_to_hip_left = distance(lw, hip_mid) / torso_length

    if rw is not None:
        wrist_to_hip_right = distance(rw, hip_mid) / torso_length

    left_ankle_x = None
    left_ankle_y = None
    right_ankle_x = None
    right_ankle_y = None

    if la is not None:
    	left_ankle_x = float(la[0])
    	left_ankle_y = float(la[1])

    if ra is not None:
    	right_ankle_x = float(ra[0])
    	right_ankle_y = float(ra[1])

    return {
        "body_center_x": float(body_center[0]),
        "body_center_y": float(body_center[1]),

        "shoulder_mid_x": float(shoulder_mid[0]),
        "shoulder_mid_y": float(shoulder_mid[1]),

        "hip_mid_x": float(hip_mid[0]),
        "hip_mid_y": float(hip_mid[1]),

        "torso_length": float(torso_length),
        "torso_angle": torso_angle,

        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,

        "ankle_distance": ankle_distance,

        "left_wrist_hip_distance": wrist_to_hip_left,
        "right_wrist_hip_distance": wrist_to_hip_right,

        "left_ankle_x": left_ankle_x,
        "left_ankle_y": left_ankle_y,

        "right_ankle_x": right_ankle_x,
        "right_ankle_y": right_ankle_y,
    }
