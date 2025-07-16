import os
import subprocess
import time

import cv2


def has_libcamera_jpeg():
    """Check if libcamera-jpeg is available"""
    try:
        subprocess.run(["which", "libcamera-jpeg"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def capture_image(config):
    """Capture image using available camera"""
    if has_libcamera_jpeg():
        img_path = f"frame_{int(time.time() * 1000)}.jpg"
        try:
            subprocess.run(["pkill", "-f", "libcamera-jpeg"], capture_output=True)
            time.sleep(0.1)
            resolution = config.get("camera", {})
            width = resolution.get("width", 640)
            height = resolution.get("height", 480)
            subprocess.run(
                ["libcamera-jpeg", "-o", img_path, "-n", "-t", "1000", "--width", str(width), "--height", str(height)],
                capture_output=True,
                check=True,
            )
            if not os.path.exists(img_path):
                raise Exception(f"Image file not created: {img_path}")
            image = cv2.imread(img_path)
            os.unlink(img_path)
            return image
        except Exception as e:
            raise Exception(f"Pi Camera (libcamera) capture failed: {e}")
    else:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Camera not found, trying alternative camera indices...")
                # Try different camera indices
                for i in range(1, 5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        print(f"✅ Camera found at index {i}")
                        break
                else:
                    raise Exception("No camera found at any index")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
            cap.set(cv2.CAP_PROP_FPS, config["camera"]["fps"])

            # Try to read frame
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                return frame
            else:
                raise Exception("Failed to capture frame")
        except Exception as e:
            raise Exception(f"OpenCV webcam capture failed: {e}")
