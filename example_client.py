import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"


def check_status():
    """Check current detection status"""
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        status = response.json()
        print(f"Status: {json.dumps(status, indent=2)}")
        return status
    else:
        print(f"Error getting status: {response.status_code}")
        return None


def start_detection():
    """Start the detection system"""
    response = requests.post(f"{BASE_URL}/start")
    if response.status_code == 200:
        print("Detection started successfully!")
        return True
    else:
        print(f"Error starting detection: {response.text}")
        return False


def stop_detection():
    """Stop the detection system"""
    response = requests.post(f"{BASE_URL}/stop")
    if response.status_code == 200:
        print("Detection stopped successfully!")
        return True
    else:
        print(f"Error stopping detection: {response.text}")
        return False


def get_logs():
    """Get all detection logs"""
    response = requests.get(f"{BASE_URL}/logs")
    if response.status_code == 200:
        logs = response.json()
        print(f"Found {len(logs)} detection results:")
        for log in logs:
            print(
                f"  - {log['timestamp']}: x={log['x_coordinate']}, confidence={log['confidence']:.2f}"
            )
        return logs
    else:
        print(f"Error getting logs: {response.status_code}")
        return []


def get_images():
    """Get list of available images"""
    response = requests.get(f"{BASE_URL}/images")
    if response.status_code == 200:
        images = response.json()
        print(f"Available images: {len(images['images'])}")
        for img in images["images"]:
            print(f"  - {img['timestamp']}: {img['url']}")
        return images["images"]
    else:
        print(f"Error getting images: {response.status_code}")
        return []


def update_config(video_source="0", wait_duration=10, stillness_threshold=30):
    """Update detection configuration"""
    config = {
        "video_source": video_source,
        "wait_duration": wait_duration,
        "stillness_threshold": stillness_threshold,
        "capture_frames": 10,
    }

    response = requests.post(f"{BASE_URL}/config", json=config)
    if response.status_code == 200:
        print("Configuration updated successfully!")
        return True
    else:
        print(f"Error updating config: {response.text}")
        return False


def monitor_detection(duration=60):
    """Monitor detection for a specified duration"""
    print(f"Monitoring detection for {duration} seconds...")
    end_time = time.time() + duration

    while time.time() < end_time:
        status = check_status()
        if status:
            if status["is_running"]:
                phase = status["current_phase"]
                remaining = status.get("time_remaining")
                if remaining:
                    print(f"  Phase: {phase}, Time remaining: {remaining}s")
                else:
                    print(f"  Phase: {phase}")
            else:
                print("  Detection is not running")

        time.sleep(5)  # Check every 5 seconds


def main():
    """Example usage of the API"""
    print("=== Geupsik Line Detection API Client ===")

    # Check initial status
    print("\n1. Checking initial status...")
    check_status()

    # Update configuration for webcam
    print("\n2. Updating configuration...")
    update_config(video_source="0", wait_duration=5)  # Short wait for demo

    # Start detection
    print("\n3. Starting detection...")
    if start_detection():
        # Monitor for 30 seconds
        print("\n4. Monitoring detection...")
        monitor_detection(30)

        # Check logs
        print("\n5. Checking detection logs...")
        get_logs()

        # Get images
        print("\n6. Checking available images...")
        get_images()

        # Stop detection
        print("\n7. Stopping detection...")
        stop_detection()

    print("\nExample completed!")


if __name__ == "__main__":
    main()
