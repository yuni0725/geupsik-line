import cv2
import time
import os
import csv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# --- User choice for video source ---
print("Choose video source:")
print("1: Local Webcam")
print("2: IP Webcam (Mobile Phone)")
choice = input("Enter your choice (1 or 2): ")

video_source = None
if choice == "1":
    video_source = 0
elif choice == "2":
    video_source = "IP_CAMERA_URL"
else:
    print("Invalid choice. Exiting.")
    exit()

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# --- Setup for CSV Logging ---
if not os.path.exists("results"):
    os.makedirs("results")

csv_log_path = "results/target_log.csv"
csv_header = ["timestamp", "x_coordinate"]
if not os.path.exists(csv_log_path):
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# --- Main Loop to repeat the process ---
wait_duration = 10  # Initial wait time

while True:
    # --- 1. Wait/Warm-up Phase ---
    print(
        f"\nStarting next {wait_duration}-second cycle. Press 'q' during countdown to quit."
    )
    start_time = time.time()
    while time.time() - start_time < wait_duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Lost connection to camera during wait cycle.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        remaining_time = int(wait_duration - (time.time() - start_time))
        countdown_text = f"Next capture in: {remaining_time}s"
        cv2.putText(
            frame, countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) == ord("q"):
            print("Program exit requested by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print("Wait complete.")

    # --- 2. Capture Phase ---
    print("Capturing 10 frames...")
    frames = []
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame {i+1}.")
            break
        frames.append(frame)
        time.sleep(0.1)

    if len(frames) < 10:
        print("Could not capture enough frames. Restarting with a 10-second wait.")
        wait_duration = 10
        continue

    print("Capture complete.")

    # --- 3. Analysis Phase ---
    print("Analyzing frames to find a confirmed target...")
    STILLNESS_THRESHOLD = 30
    rightmost_positions = []
    rightmost_boxes = []

    for frame in frames:
        resized_frame = cv2.resize(frame, (600, 400))
        results = model(resized_frame, verbose=False)

        current_rightmost_box = None
        max_center_x = -1
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls[0])] == "person":
                    x1, _, x2, _ = box.xyxy[0]
                    center_x = (x1 + x2) / 2
                    if center_x > max_center_x:
                        max_center_x = center_x
                        current_rightmost_box = box

        if current_rightmost_box is not None:
            x1, y1, x2, y2 = current_rightmost_box.xyxy[0]
            rightmost_positions.append(((x1 + x2) / 2, (y1 + y2) / 2))
            rightmost_boxes.append(current_rightmost_box)
        else:
            rightmost_positions.append(None)
            rightmost_boxes.append(None)

    confirmed_target = False
    if all(pos is not None for pos in rightmost_positions):
        start_pos = rightmost_positions[0]
        end_pos = rightmost_positions[-1]

        dist = (
            (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
        ) ** 0.5

        if dist < STILLNESS_THRESHOLD:
            confirmed_target = True
            print(f"Confirmed target found. Total movement: {dist:.2f} pixels.")
        else:
            print(f"Target not confirmed. Moved {dist:.2f} pixels.")
    else:
        print(
            "Target not confirmed. Rightmost person was not visible in all 10 frames."
        )

    # --- 4. & 5. Output Phase ---
    if confirmed_target:
        wait_duration = 30  # Set next wait to 60 seconds
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        final_box = rightmost_boxes[-1]
        x1, _, _, _ = [int(coord) for coord in final_box.xyxy[0]]

        # Log to CSV
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, x1])

        # Save image with timestamp
        final_frame = cv2.resize(frames[-1], (600, 400))
        label = f"Confirmed Target {final_box.conf[0]:.2f}"
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
        cv2.imwrite(f"results/confirmed_target_image_{timestamp}.jpg", final_frame)

        print(f"Successfully logged results to CSV with timestamp {timestamp}.")
    else:
        wait_duration = 5  # Set next wait to 10 seconds
        print("Could not confirm a stationary target. No files were saved.")

cap.release()
cv2.destroyAllWindows()
