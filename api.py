import cv2
import time
import os
import csv
import threading
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import logging
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Geupsik Line Detection API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # If using a dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "file://",  # For opening HTML files directly
        "*",  # Allow all origins (less secure but convenient for development)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Pydantic models for API
class DetectionConfig(BaseModel):
    video_source: str = "0"  # "0" for webcam, or IP camera URL
    wait_duration: int = 10
    stillness_threshold: int = 30
    capture_frames: int = 10


class DetectionStatus(BaseModel):
    is_running: bool
    current_phase: str
    time_remaining: Optional[int] = None
    total_detections: int
    last_detection: Optional[str] = None


class DetectionResult(BaseModel):
    timestamp: str
    x_coordinate: int
    confidence: float
    image_path: str


class DetectionSystem:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.is_running = False
        self.current_phase = "idle"
        self.time_remaining = None
        self.total_detections = 0
        self.last_detection = None
        self.config = DetectionConfig()
        self.cap = None
        self.detection_thread = None
        self.stop_requested = False

        self.observation_x = []

        self.waiting_time = None

        # Setup directories
        if not os.path.exists("results"):
            os.makedirs("results")

        # Setup CSV
        self.csv_log_path = "results/target_log.csv"
        csv_header = ["timestamp", "x_coordinate", "confidence"]
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)

    def _initialize_camera(self):
        """Initialize camera based on config"""
        try:
            video_source = self.config.video_source
            if video_source == "0":
                video_source = 0

            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {video_source}")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def _capture_and_analyze(self):
        """Core detection logic with real-time display"""
        if not self.cap:
            return False

        frames = []
        display_frames = []

        for i in range(self.config.capture_frames):
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"Failed to capture frame {i+1}")
                return False
            frames.append(frame)
            display_frames.append(frame.copy())
            time.sleep(0.1)

        # Analysis phase
        rightmost_positions = []
        rightmost_boxes = []

        for idx, frame in enumerate(frames):
            resized_frame = cv2.resize(frame, (600, 400))
            results = self.model(resized_frame, verbose=False)

            current_rightmost_box = None
            max_center_x = -1
            display_frame = cv2.resize(display_frames[idx], (600, 400))

            # Draw all detected persons
            for result in results:
                for box in result.boxes:
                    if self.model.names[int(box.cls[0])] == "person":
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = (x1 + x2) / 2

                        # Draw detection box
                        cv2.rectangle(
                            display_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )

                        # Draw confidence
                        conf = float(box.conf[0])
                        cv2.putText(
                            display_frame,
                            f"Person {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

                        if center_x > max_center_x:
                            max_center_x = center_x
                            current_rightmost_box = box

            # Highlight rightmost person
            if current_rightmost_box is not None:
                x1, y1, x2, y2 = current_rightmost_box.xyxy[0]
                cv2.rectangle(
                    display_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    display_frame,
                    "RIGHTMOST TARGET",
                    (int(x1), int(y1) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                rightmost_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))
                rightmost_boxes.append(current_rightmost_box)
            else:
                rightmost_positions.append(None)
                rightmost_boxes.append(None)

            # Show frame with analysis info
            cv2.putText(
                display_frame,
                f"Frame {idx+1}/{self.config.capture_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "ANALYZING...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Detection Analysis", display_frame)
            cv2.waitKey(200)  # Show each frame for 200ms

        # Check for confirmed target
        confirmed_target = False
        if all(pos is not None for pos in rightmost_positions):
            start_pos = rightmost_positions[0]
            end_pos = rightmost_positions[-1]

            dist = (
                (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
            ) ** 0.5

            if dist < self.config.stillness_threshold:
                confirmed_target = True
                logger.info(
                    f"Confirmed target found. Total movement: {dist:.2f} pixels"
                )

                # Save results
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                final_box = rightmost_boxes[-1]
                x1, _, _, _ = [int(coord) for coord in final_box.xyxy[0]]
                confidence = float(final_box.conf[0])

                # Log to CSV
                with open(self.csv_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, x1, confidence])

                # Save image
                final_frame = cv2.resize(frames[-1], (600, 400))
                label = f"Confirmed Target {confidence:.2f}"
                color = (255, 0, 255)
                cv2.rectangle(
                    final_frame,
                    (int(final_box.xyxy[0][0]), int(final_box.xyxy[0][1])),
                    (int(final_box.xyxy[0][2]), int(final_box.xyxy[0][3])),
                    color,
                    2,
                )
                cv2.putText(
                    final_frame,
                    label,
                    (x1, int(final_box.xyxy[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                image_path2 = f"results/confirmed_target_image_{timestamp}.jpg"
                cv2.imwrite(image_path2, final_frame)

                self.total_detections += 1
                self.last_detection = timestamp

                # Show final confirmed target
                result_frame = cv2.resize(frames[-1], (600, 400))
                cv2.rectangle(
                    result_frame,
                    (int(final_box.xyxy[0][0]), int(final_box.xyxy[0][1])),
                    (int(final_box.xyxy[0][2]), int(final_box.xyxy[0][3])),
                    (255, 0, 255),
                    3,
                )
                cv2.putText(
                    result_frame,
                    "CONFIRMED TARGET!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    result_frame,
                    f"Confidence: {confidence:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    result_frame,
                    f"X Position: {x1}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("DETECTION RESULT", result_frame)
                cv2.waitKey(3000)  # Show result for 3 seconds

                return True, x1
            else:
                # Show failed analysis
                result_frame = cv2.resize(frames[-1], (600, 400))
                cv2.putText(
                    result_frame,
                    "TARGET NOT CONFIRMED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    result_frame,
                    f"Movement: {dist:.2f} pixels",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    result_frame,
                    f"Threshold: {self.config.stillness_threshold}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("DETECTION RESULT", result_frame)
                cv2.waitKey(2000)  # Show result for 2 seconds
        else:
            # Show no person detected message
            result_frame = cv2.resize(frames[-1], (600, 400))
            cv2.putText(
                result_frame,
                "NO PERSON IN ALL FRAMES",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                result_frame,
                "Person not visible throughout capture",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imshow("DETECTION RESULT", result_frame)
            cv2.waitKey(2000)  # Show result for 2 seconds

        return False, None

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        self.stop_requested = False
        wait_duration = self.config.wait_duration

        while not self.stop_requested:
            # Wait phase
            self.current_phase = "waiting"
            start_time = time.time()

            while time.time() - start_time < wait_duration and not self.stop_requested:
                self.time_remaining = int(wait_duration - (time.time() - start_time))

                # Show live camera feed during waiting
                ret, frame = self.cap.read()
                if ret:
                    display_frame = cv2.resize(frame, (600, 400))
                    cv2.putText(
                        display_frame,
                        f"WAITING... {self.time_remaining}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Total Detections: {self.total_detections}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if self.waiting_time:
                        cv2.putText(
                            display_frame,
                            f"Predicted Wait: {self.waiting_time}s",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    cv2.imshow("Live Camera Feed", display_frame)
                    cv2.waitKey(1)

                time.sleep(1)

            if self.stop_requested:
                break

            # Capture and analyze phase
            self.current_phase = "analyzing"
            self.time_remaining = None

            confirmed, x1 = self._capture_and_analyze()

            if confirmed:
                avg_x = self.get_avg_x(x1)
                if avg_x is not None:
                    waiting_time = self.get_time(avg_x)

                    self.waiting_time = waiting_time

                    print(f"Waiting time: {waiting_time} seconds")

            # Set next wait duration based on results
            wait_duration = 10 if confirmed else 5

        self.is_running = False
        self.current_phase = "idle"
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_avg_x(self, x):
        self.observation_x.append(x)

        if len(self.observation_x) > 5:
            for i in range(2):
                max_x = max(self.observation_x)

                del self.observation_x[self.observation_x.index(max_x)]
            avg_x = sum(self.observation_x) / len(self.observation_x)
            self.observation_x = []
            return avg_x

        return None

    def get_time(self, avg_x):
        a, b, c = self.load_parameters()
        if avg_x > 600:
            avg_x = 600
        return int(a * np.exp(b * avg_x) + c)

    def load_parameters(self, filename="exponential_params.json"):
        """Load fitted parameters from a JSON file"""
        if os.path.exists(filename):
            with open(filename, "r") as f:
                params = json.load(f)
            print(f"Parameters loaded from {filename}")
            print(f"Loaded equation: {params['equation']}")
            return params["a"], params["b"], params["c"]
        else:
            print(f"No saved parameters found in {filename}")
            return 1.0, 0.1, 10.0  # Default parameters if file not found

    def start_detection(self) -> bool:
        """Start the detection system"""
        if self.is_running:
            return False

        if not self._initialize_camera():
            return False

        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        return True

    def stop_detection(self) -> bool:
        """Stop the detection system"""
        if not self.is_running:
            return False

        self.stop_requested = True
        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.is_running = False
        self.current_phase = "idle"
        self.time_remaining = None
        return True

    def get_status(self) -> DetectionStatus:
        """Get current system status"""
        return DetectionStatus(
            is_running=self.is_running,
            current_phase=self.current_phase,
            time_remaining=self.time_remaining,
            total_detections=self.total_detections,
            last_detection=self.last_detection,
        )


# Global detection system instance
detection_system = DetectionSystem()


@app.get("/")
async def root():
    """Root endpoint"""
    waiting_time = detection_system.waiting_time
    print(waiting_time)
    return {"waiting_time": waiting_time}


if __name__ == "__main__":

    detection_system.start_detection()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
