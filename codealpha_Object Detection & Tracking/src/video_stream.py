"""Video Stream module for capturing frames from webcam or video file."""

from typing import Optional, Generator
import cv2
import numpy as np


class VideoStream:
    """Video stream handler for webcam or video file input."""

    def __init__(self, source: int | str = 0):
        """
        Initialize video stream.

        Args:
            source: Video source (0 for webcam, or path to video file).
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

    def start(self) -> bool:
        """
        Start the video stream.

        Returns:
            True if stream started successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return False

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"Video stream started: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the stream.

        Returns:
            Tuple of (success, frame).
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from the video stream.

        Yields:
            BGR frames as numpy arrays.
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def stop(self):
        """Stop the video stream and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Video stream stopped.")

    def get_properties(self) -> dict:
        """Get video stream properties."""
        return {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "source": self.source,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class VideoWriter:
    """Video writer for saving output video."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "mp4v",
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file.
            width: Frame width.
            height: Frame height.
            fps: Frames per second.
            codec: FourCC codec code.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None

    def start(self) -> bool:
        """Start the video writer."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            print(f"Error: Could not open video writer: {self.output_path}")
            return False
        print(f"Video writer started: {self.output_path}")
        return True

    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        """Stop the video writer and release resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved to: {self.output_path}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
