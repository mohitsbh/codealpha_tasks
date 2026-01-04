"""
Flask Web Application for Object Detection & Tracking

Provides a web-based UI for real-time object detection and tracking.
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import threading
import time

from src.detector import ObjectDetector
from src.tracker import Sort
from src.visualizer import Visualizer
from src.utils import load_config, FPSCounter


app = Flask(__name__)

# Global variables for streaming
class StreamManager:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.visualizer = None
        self.config = None
        self.video_source = 0
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.fps_counter = FPSCounter()
        self.stats = {
            "fps": 0,
            "detections": 0,
            "tracks": 0,
        }
        
    def initialize(self, config_path="config/config.yaml"):
        """Initialize detection and tracking components."""
        self.config = load_config(config_path)
        
        det_config = self.config.get("detection", {})
        tracker_config = self.config.get("tracker", {})
        display_config = self.config.get("display", {})
        
        # Initialize detector
        self.detector = ObjectDetector(
            model_path=det_config.get("model", "yolov8n.pt"),
            confidence_threshold=det_config.get("confidence_threshold", 0.5),
            iou_threshold=det_config.get("iou_threshold", 0.45),
            classes=det_config.get("classes"),
            device=det_config.get("device", "auto"),
        )
        
        # Initialize tracker
        self.tracker = Sort(
            max_age=tracker_config.get("max_age", 30),
            min_hits=tracker_config.get("min_hits", 3),
            iou_threshold=tracker_config.get("iou_threshold", 0.3),
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            show_fps=display_config.get("show_fps", True),
            show_labels=display_config.get("show_labels", True),
            show_confidence=display_config.get("show_confidence", True),
            show_track_id=display_config.get("show_track_id", True),
            bbox_thickness=display_config.get("bbox_thickness", 2),
            font_scale=display_config.get("font_scale", 0.6),
        )
        
        self.video_source = self.config.get("video_source", 0)
        if str(self.video_source).isdigit():
            self.video_source = int(self.video_source)

    def process_frame(self, frame):
        """Process a single frame through detection and tracking."""
        # Run detection
        boxes, scores, class_ids = self.detector.detect(frame)
        
        # Prepare detections for tracker (x1, y1, x2, y2, score)
        if len(boxes) > 0:
            detections = np.column_stack([boxes, scores])
        else:
            detections = np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Extract tracking results
        if len(tracks) > 0:
            track_boxes = tracks[:, :4]
            track_ids = tracks[:, 4].astype(int)
            # Match track boxes with detections to get class IDs
            track_scores = np.ones(len(tracks))
            track_class_ids = np.zeros(len(tracks), dtype=int)
            
            for i, track_box in enumerate(track_boxes):
                if len(boxes) > 0:
                    # Find closest detection
                    ious = self._compute_iou(track_box, boxes)
                    if len(ious) > 0:
                        best_idx = np.argmax(ious)
                        track_scores[i] = scores[best_idx]
                        track_class_ids[i] = class_ids[best_idx]
        else:
            track_boxes = np.empty((0, 4))
            track_ids = np.empty(0, dtype=int)
            track_scores = np.empty(0)
            track_class_ids = np.empty(0, dtype=int)
        
        # Update stats
        self.stats["detections"] = len(boxes)
        self.stats["tracks"] = len(track_ids)
        
        # Draw results
        frame = self.visualizer.draw_detections(
            frame,
            track_boxes,
            track_scores,
            track_class_ids,
            self.detector.class_names,
            track_ids,
        )
        
        # Draw FPS
        self.fps_counter.update()
        fps = self.fps_counter.get_fps()
        self.stats["fps"] = round(fps, 1)
        
        if self.visualizer.show_fps:
            frame = self.visualizer.draw_fps(frame, fps)
        
        return frame
    
    def _compute_iou(self, box1, boxes):
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        return iou


stream_manager = StreamManager()


def generate_frames():
    """Generator function for video streaming."""
    cap = cv2.VideoCapture(stream_manager.video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {stream_manager.video_source}")
        return
    
    stream_manager.is_running = True
    
    while stream_manager.is_running:
        ret, frame = cap.read()
        if not ret:
            # If video file ends, restart
            if isinstance(stream_manager.video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Process frame
        processed_frame = stream_manager.process_frame(frame)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    stream_manager.is_running = False


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def get_stats():
    """Get current detection/tracking statistics."""
    return jsonify(stream_manager.stats)


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration."""
    if request.method == 'GET':
        return jsonify(stream_manager.config)
    else:
        data = request.json
        # Update confidence threshold
        if 'confidence' in data:
            stream_manager.detector.confidence_threshold = float(data['confidence'])
        if 'iou_threshold' in data:
            stream_manager.detector.iou_threshold = float(data['iou_threshold'])
        return jsonify({"status": "ok"})


@app.route('/api/source', methods=['POST'])
def set_source():
    """Set video source."""
    data = request.json
    source = data.get('source', 0)
    if str(source).isdigit():
        source = int(source)
    stream_manager.video_source = source
    return jsonify({"status": "ok", "source": source})


def main():
    """Main entry point for Flask app."""
    print("Initializing Object Detection & Tracking...")
    stream_manager.initialize()
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
