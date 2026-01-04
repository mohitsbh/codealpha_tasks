"""Object Tracking module using SORT algorithm."""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_test: Array of shape (N, 4) with [x1, y1, x2, y2] format.
        bb_gt: Array of shape (M, 4) with [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - intersection
    return intersection / np.maximum(union, 1e-10)


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, s, r] where s=area and r=aspect ratio."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float = None) -> np.ndarray:
    """Convert [cx, cy, s, r] back to [x1, y1, x2, y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0
    bbox = np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).flatten()
    if score is not None:
        return np.append(bbox, score)
    return bbox


class KalmanBoxTracker:
    """Kalman filter-based tracker for a single object."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with bounding box.

        Args:
            bbox: Initial bounding box [x1, y1, x2, y2].
        """
        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Predict next state and return bounding box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)


class Sort:
    """SORT: Simple Online and Realtime Tracking."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize SORT tracker.

        Args:
            max_age: Maximum frames to keep alive without matching.
            min_hits: Minimum hits before track is confirmed.
            iou_threshold: IoU threshold for matching.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update trackers with new detections.

        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, score].

        Returns:
            Array of shape (M, 5) with [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        # Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4])
            self.trackers.append(trk)

        # Return active tracks
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _associate_detections_to_trackers(
        self, detections: np.ndarray, trackers: np.ndarray
    ) -> tuple[np.ndarray, list, list]:
        """Associate detections to tracked objects using IoU."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))

        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])

        # Hungarian algorithm
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out low IoU matches
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers

    def reset(self):
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
