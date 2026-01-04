"""
Object Detection & Tracking - Main Application

Real-time object detection and tracking using YOLOv8 and SORT algorithm.
Supports webcam input or video file processing.

Usage:
    python main.py                      # Run with default config (webcam)
    python main.py --source video.mp4   # Run with video file
    python main.py --config config.yaml # Run with custom config
"""

import argparse
import os
import cv2
import numpy as np

from src.detector import ObjectDetector
from src.tracker import Sort
from src.video_stream import VideoStream, VideoWriter
from src.visualizer import Visualizer
from src.utils import load_config, ensure_dir, FPSCounter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Object Detection & Tracking with YOLOv8 and SORT"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source (0 for webcam, or path to video file). Overrides config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model. Overrides config.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Overrides config.",
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    video_source = args.source if args.source is not None else config.get("video_source", 0)
    if video_source != "0" and str(video_source).isdigit():
        video_source = int(video_source)

    det_config = config.get("detection", {})
    tracker_config = config.get("tracker", {})
    display_config = config.get("display", {})
    output_config = config.get("output", {})

    model_path = args.model if args.model else det_config.get("model", "yolov8n.pt")
    save_video = args.save or output_config.get("save_video", False)
    output_path = args.output if args.output else output_config.get("output_path", "outputs/output.mp4")

    # Initialize components
    print("Initializing Object Detector...")
    detector = ObjectDetector(
        model_path=model_path,
        confidence_threshold=det_config.get("confidence_threshold", 0.5),
        iou_threshold=det_config.get("iou_threshold", 0.45),
        classes=det_config.get("classes"),
        device=det_config.get("device", "auto"),
    )

    print("Initializing SORT Tracker...")
    tracker = Sort(
        max_age=tracker_config.get("max_age", 30),
        min_hits=tracker_config.get("min_hits", 3),
        iou_threshold=tracker_config.get("iou_threshold", 0.3),
    )

    visualizer = Visualizer(
        show_fps=display_config.get("show_fps", True),
        show_labels=display_config.get("show_labels", True),
        show_confidence=display_config.get("show_confidence", True),
        show_track_id=display_config.get("show_track_id", True),
        bbox_thickness=display_config.get("bbox_thickness", 2),
        font_scale=display_config.get("font_scale", 0.6),
    )

    fps_counter = FPSCounter()
    window_name = display_config.get("window_name", "Object Detection & Tracking")

    # Initialize video stream
    print(f"Opening video source: {video_source}")
    video_stream = VideoStream(video_source)
    if not video_stream.start():
        print("Failed to open video source. Exiting.")
        return

    # Initialize video writer if saving
    video_writer = None
    if save_video:
        ensure_dir(os.path.dirname(output_path))
        video_writer = VideoWriter(
            output_path=output_path,
            width=video_stream.frame_width,
            height=video_stream.frame_height,
            fps=output_config.get("fps", video_stream.fps),
        )
        video_writer.start()

    print("\nStarting detection and tracking...")
    print("Press 'q' to quit, 'r' to reset tracker, 's' to save screenshot\n")

    # Main processing loop
    try:
        for frame in video_stream.frames():
            # Detect objects
            boxes, scores, class_ids = detector.detect(frame)

            # Prepare detections for tracker [x1, y1, x2, y2, score]
            if len(boxes) > 0:
                detections = np.column_stack([boxes, scores])
            else:
                detections = np.empty((0, 5))

            # Update tracker
            tracks = tracker.update(detections)

            # Match tracks to detections for class information
            track_class_ids = []
            track_scores = []
            for track in tracks:
                track_box = track[:4]
                # Find best matching detection
                best_match_idx = -1
                best_iou = 0.0
                for i, det_box in enumerate(boxes):
                    iou = compute_iou(track_box, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i

                if best_match_idx >= 0:
                    track_class_ids.append(class_ids[best_match_idx])
                    track_scores.append(scores[best_match_idx])
                else:
                    track_class_ids.append(0)
                    track_scores.append(0.0)

            track_class_ids = np.array(track_class_ids)
            track_scores = np.array(track_scores)

            # Update FPS
            fps = fps_counter.update()

            # Visualize results
            frame = visualizer.draw_tracks(
                frame, tracks, track_class_ids, track_scores, detector.class_names
            )
            frame = visualizer.draw_fps(frame, fps)

            # Save frame if recording
            if video_writer:
                video_writer.write(frame)

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("r"):
                tracker.reset()
                print("Tracker reset.")
            elif key == ord("s"):
                screenshot_path = f"outputs/screenshot_{int(fps_counter.last_time)}.jpg"
                ensure_dir("outputs")
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Cleanup
        video_stream.stop()
        if video_writer:
            video_writer.stop()
        cv2.destroyAllWindows()
        print("Done.")


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    main()
