"""Visualization module for drawing bounding boxes and labels."""

from typing import Optional
import cv2
import numpy as np


# Color palette for different track IDs
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
]


def get_color(track_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return COLORS[int(track_id) % len(COLORS)]


class Visualizer:
    """Visualizer for drawing detection and tracking results."""

    def __init__(
        self,
        show_fps: bool = True,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_track_id: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.6,
    ):
        """
        Initialize visualizer.

        Args:
            show_fps: Whether to display FPS.
            show_labels: Whether to display class labels.
            show_confidence: Whether to display confidence scores.
            show_track_id: Whether to display tracking IDs.
            bbox_thickness: Thickness of bounding box lines.
            font_scale: Scale of text font.
        """
        self.show_fps = show_fps
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: dict,
        track_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: BGR image to draw on.
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2].
            scores: Array of shape (N,) with confidence scores.
            class_ids: Array of shape (N,) with class indices.
            class_names: Dict mapping class IDs to names.
            track_ids: Optional array of shape (N,) with track IDs.

        Returns:
            Frame with drawings.
        """
        frame = frame.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            score = scores[i] if len(scores) > i else 0.0
            class_id = class_ids[i] if len(class_ids) > i else 0
            track_id = track_ids[i] if track_ids is not None and len(track_ids) > i else None

            # Get color based on track ID or class ID
            color = get_color(track_id if track_id is not None else class_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)

            # Build label string
            label_parts = []
            if self.show_track_id and track_id is not None:
                label_parts.append(f"ID:{int(track_id)}")
            if self.show_labels:
                class_name = class_names.get(class_id, f"cls_{class_id}")
                label_parts.append(class_name)
            if self.show_confidence:
                label_parts.append(f"{score:.2f}")

            if label_parts:
                label = " ".join(label_parts)
                self._draw_label(frame, label, (x1, y1), color)

        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
        class_names: dict,
    ) -> np.ndarray:
        """
        Draw tracking results on frame.

        Args:
            frame: BGR image to draw on.
            tracks: Array of shape (N, 5) with [x1, y1, x2, y2, track_id].
            class_ids: Array of shape (N,) with class indices.
            scores: Array of shape (N,) with confidence scores.
            class_names: Dict mapping class IDs to names.

        Returns:
            Frame with drawings.
        """
        if len(tracks) == 0:
            return frame

        boxes = tracks[:, :4]
        track_ids = tracks[:, 4]

        return self.draw_detections(
            frame, boxes, scores, class_ids, class_names, track_ids
        )

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame."""
        if not self.show_fps:
            return frame

        frame = frame.copy()
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, text, (10, 30),
            self.font, self.font_scale, (0, 255, 0), 2
        )
        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ):
        """Draw label with background."""
        x, y = position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width + 5, y),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame, label, (x + 2, y - 5),
            self.font, self.font_scale, (255, 255, 255), 1
        )
