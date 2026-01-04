"""Tests for the SORT tracker."""

import numpy as np
import pytest
from src.tracker import Sort, iou_batch, convert_bbox_to_z, convert_x_to_bbox


class TestIoU:
    """Tests for IoU computation."""

    def test_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[0, 0, 10, 10]])
        iou = iou_batch(box1, box2)
        assert np.allclose(iou, 1.0)

    def test_no_overlap(self):
        """Test IoU with no overlap."""
        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[20, 20, 30, 30]])
        iou = iou_batch(box1, box2)
        assert np.allclose(iou, 0.0)

    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[5, 5, 15, 15]])
        iou = iou_batch(box1, box2)
        # Intersection: 5x5=25, Union: 100+100-25=175
        expected = 25 / 175
        assert np.allclose(iou, expected, atol=0.01)


class TestBboxConversion:
    """Tests for bounding box format conversion."""

    def test_bbox_to_z_and_back(self):
        """Test conversion from bbox to z and back."""
        bbox = np.array([10, 20, 50, 80])
        z = convert_bbox_to_z(bbox)
        bbox_recovered = convert_x_to_bbox(z)
        assert np.allclose(bbox, bbox_recovered, atol=0.1)


class TestSort:
    """Tests for SORT tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)
        assert tracker.max_age == 10
        assert tracker.min_hits == 2
        assert len(tracker.trackers) == 0

    def test_single_detection(self):
        """Test tracking with single detection."""
        tracker = Sort(max_age=5, min_hits=1)

        # First frame - detection should start a track
        det = np.array([[100, 100, 200, 200, 0.9]])
        tracks = tracker.update(det)

        # After min_hits frames, track should appear
        for _ in range(2):
            tracks = tracker.update(det)

        assert len(tracks) >= 1

    def test_empty_detection(self):
        """Test tracking with no detections."""
        tracker = Sort()
        det = np.empty((0, 5))
        tracks = tracker.update(det)
        assert len(tracks) == 0

    def test_reset(self):
        """Test tracker reset."""
        tracker = Sort()
        det = np.array([[100, 100, 200, 200, 0.9]])
        tracker.update(det)
        tracker.update(det)

        tracker.reset()
        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
