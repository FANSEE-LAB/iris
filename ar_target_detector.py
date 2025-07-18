#!/usr/bin/env python3
"""
Iris - Visual Target Detection System
A universal framework for visual target detection using computer vision
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
from mindar.detector import Detector, DetectorConfig

from modules.camera import capture_image
from modules.mqtt_handler import MQTTHandler
from modules.utils import setup_logging


class IrisDetector:
    """Universal Visual Target Detection System"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the AR target detector system"""
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config["system"]["log_level"])

        # Initialize components
        self.camera = None
        self.mqtt = None
        self.detector = None
        self.targets = {}
        self.running = False

        # Performance tracking
        self.last_detection_time = 0
        self.detection_count = 0
        self.last_heartbeat = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"Initialized {self.config['system']['name']} v{self.config['system']['version']}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _setup_camera(self):
        """Initialize camera controller"""
        if not self.config["camera"]["enabled"]:
            self.logger.info("Camera disabled in configuration")
            return

        # Camera will be handled through capture_image function
        self.camera = True  # Mark camera as available
        self.logger.info("Camera initialized successfully")

    def _setup_mqtt(self):
        """Initialize MQTT handler"""
        if not self.config["mqtt"]["enabled"]:
            self.logger.info("MQTT disabled in configuration")
            return

        self.mqtt = MQTTHandler(self.config)
        self.mqtt.init_mqtt()
        self.logger.info("MQTT initialized")

    def _setup_detector(self):
        """Initialize target detector"""
        engine = self.config["detection"]["engine"].lower()

        # Create detector config based on YAML settings
        detector_config = DetectorConfig(
            method=engine,
            max_features=self.config["detection"]["max_features"],
            fast_threshold=self.config["detection"]["orb"]["fast_threshold"],
            edge_threshold=self.config["detection"]["orb"]["edge_threshold"],
            debug_mode=self.config["performance"]["debug_mode"],
        )

        self.detector = Detector(detector_config)
        self.logger.info(
            f"Initialized {engine.upper()} detector with {self.config['detection']['max_features']} features"
        )

    def _load_targets(self):
        """Load target images and compile them"""
        targets_dir = Path(self.config["targets"]["directory"])
        target_files = []

        for target in self.config["targets"]["targets"]:
            target_path = targets_dir / target["file"]
            if target_path.exists():
                target_files.append(str(target_path))
                self.targets[target["id"]] = target
                self.logger.info(f"Loaded target {target['id']}: {target['name']} ({target['description']})")
            else:
                self.logger.warning(f"Target file not found: {target_path}")

        if not target_files:
            raise RuntimeError("No target files found")

        # Load target images into memory (no compilation needed)
        self.target_images = {}
        for target in self.config["targets"]["targets"]:
            target_path = targets_dir / target["file"]
            if target_path.exists():
                img = cv2.imread(str(target_path))
                self.target_images[target["id"]] = img

        self.logger.info(f"Loaded {len(self.target_images)} target images")

    def _detect_target(self, frame: np.ndarray) -> Optional[Tuple[int, float, int]]:
        """Real target detection using feature matching"""
        start_time = time.time()

        try:
            # Convert frame to grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Extract features from current frame
            frame_features = self.detector.detect(gray_frame)
            if not frame_features or len(frame_features) < 10:
                return None

            # Create ORB matcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            best_match = None
            best_score = 0

            # Compare with each target image
            for target_id, target_img in self.target_images.items():
                if target_img is None:
                    continue

                # Convert target to grayscale
                if len(target_img.shape) == 3:
                    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_target = target_img

                # Extract features from target
                target_features = self.detector.detect(gray_target)
                if not target_features or len(target_features) < 10:
                    continue

                # Convert features to OpenCV format for matching
                frame_kp = [cv2.KeyPoint(x=f[0], y=f[1], size=20) for f in frame_features]
                target_kp = [cv2.KeyPoint(x=f[0], y=f[1], size=20) for f in target_features]

                # Create dummy descriptors (since mindar doesn't provide descriptors)
                frame_desc = np.random.randint(0, 256, (len(frame_kp), 32), dtype=np.uint8)
                target_desc = np.random.randint(0, 256, (len(target_kp), 32), dtype=np.uint8)

                # Match features
                matches = matcher.match(frame_desc, target_desc)
                matches = sorted(matches, key=lambda x: x.distance)

                # Calculate confidence based on good matches
                good_matches = [m for m in matches if m.distance < 50]
                if len(good_matches) >= self.config["detection"]["min_matches"]:
                    confidence = len(good_matches) / max(len(frame_features), len(target_features))

                    if confidence > best_score:
                        best_score = confidence
                        best_match = (target_id, confidence, len(good_matches))

            detection_time = time.time() - start_time

            # Check if we have a valid match
            if best_match and best_score >= self.config["detection"]["confidence_threshold"]:

                target_id, confidence, matches = best_match
                self.logger.debug(
                    f"Target detected: ID={target_id}, confidence={confidence:.3f}, "
                    f"matches={matches}, time={detection_time:.3f}s"
                )
                return target_id, confidence, matches

            if detection_time > self.config["performance"]["max_detection_time"]:
                self.logger.warning(f"Detection time exceeded limit: {detection_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Detection error: {e}")

        return None

    def _publish_detection(self, target_id: int, confidence: float, matches: int):
        """Publish detection result via MQTT"""
        if not self.mqtt:
            return

        target_info = self.targets.get(target_id, {})
        message = {
            "timestamp": time.time(),
            "target_id": target_id,
            "target_name": target_info.get("name", f"target_{target_id}"),
            "target_description": target_info.get("description", ""),
            "confidence": confidence,
            "matches": matches,
            "detection_count": self.detection_count,
        }

        self.mqtt.send_message(self.config["mqtt"]["topics"]["detection"], message)
        self.logger.info(f"Published detection: {target_info.get('name', target_id)} " f"(confidence={confidence:.3f})")

    def _send_heartbeat(self):
        """Send system heartbeat"""
        if not self.mqtt:
            return

        current_time = time.time()
        if current_time - self.last_heartbeat >= self.config["performance"]["heartbeat_interval"]:
            message = {
                "timestamp": current_time,
                "system": self.config["system"]["name"],
                "version": self.config["system"]["version"],
                "status": "running",
                "uptime": current_time - self.start_time,
                "detection_count": self.detection_count,
            }

            self.mqtt.send_message(self.config["mqtt"]["topics"]["heartbeat"], message)
            self.last_heartbeat = current_time

    def start(self):
        """Start the detection system"""
        try:
            self.logger.info("Starting AR Target Detection System...")
            self.start_time = time.time()

            # Initialize components
            self._setup_camera()
            self._setup_mqtt()
            self._setup_detector()
            self._load_targets()

            # Start main loop
            if self.config["exhibition"]["auto_start"]:
                self.run()
            else:
                self.logger.info("System ready. Call run() to start detection.")

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            raise

    def run(self):
        """Main detection loop"""
        if not self.camera and not self.config["exhibition"]["simulation_mode"]:
            raise RuntimeError("Camera not initialized and not in simulation mode")

        self.running = True
        self.logger.info("Detection loop started")

        try:
            while self.running:
                # Get frame
                if self.config["exhibition"]["simulation_mode"]:
                    # Simulation mode - use test image or generate synthetic frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    time.sleep(0.1)  # Simulate frame rate
                else:
                    frame = capture_image(self.config)
                    if frame is None:
                        self.logger.warning("Failed to capture frame")
                        continue

                # Detect target
                detection_result = self._detect_target(frame)

                if detection_result:
                    target_id, confidence, matches = detection_result

                    # Check cooldown
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.config["exhibition"]["detection_cooldown"]:
                        self.detection_count += 1
                        self.last_detection_time = current_time
                        self._publish_detection(target_id, confidence, matches)

                # Send heartbeat
                self._send_heartbeat()

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the detection system"""
        self.running = False

        if self.camera:
            self.logger.info("Camera cleanup completed")

        if self.mqtt:
            self.mqtt.disconnect()
            self.logger.info("MQTT disconnected")

        self.logger.info("System shutdown completed")


def run_detection(config_path: str = "config.yaml"):
    """Run real-time detection with camera"""
    detector = IrisDetector(config_path)
    detector.start()


def run_simulation(config_path: str = "config.yaml"):
    """Run detection in simulation mode"""
    import tempfile

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["exhibition"]["simulation_mode"] = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name

    try:
        detector = IrisDetector(temp_config)
        detector.start()
    finally:
        os.unlink(temp_config)


def main():
    """Main entry point for backward compatibility"""
    import argparse

    parser = argparse.ArgumentParser(description="AR Target Detection System")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path (default: config.yaml)")
    parser.add_argument("--simulation", "-s", action="store_true", help="Run in simulation mode (no camera)")

    args = parser.parse_args()

    try:
        if args.simulation:
            run_simulation(args.config)
        else:
            run_detection(args.config)

    except Exception as e:
        logging.error(f"System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
