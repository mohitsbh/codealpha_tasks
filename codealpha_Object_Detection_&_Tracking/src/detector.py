"""Object Detection module using YOLOv8."""

from typing import Optional
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """YOLOv8-based object detector."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[list] = None,
        device: str = "auto",
    ):
        """
        Initialize the object detector.

        Args:
            model_path: Path to YOLO model weights.
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            classes: List of class indices to detect (None for all).
            device: Device to run inference on (auto, cpu, cuda, mps).
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device if device != "auto" else None
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Tuple of (boxes, scores, class_ids):
                - boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format.
                - scores: Array of shape (N,) with confidence scores.
                - class_ids: Array of shape (N,) with class indices.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        # Extract detections
        boxes = []
        scores = []
        class_ids = []

        for result in results:
            if result.boxes is not None and len(result.boxes):
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)

                boxes.extend(boxes_xyxy)
                scores.extend(confs)
                class_ids.extend(cls)

        return (
            np.array(boxes) if boxes else np.empty((0, 4)),
            np.array(scores) if scores else np.empty(0),
            np.array(class_ids) if class_ids else np.empty(0, dtype=int),
        )

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return self.class_names.get(class_id, f"class_{class_id}")
