import json
import os
from pathlib import Path


def load_config():
    """Load configuration from config.json file"""
    config_path = Path(__file__).parent.parent / "config.json"

    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print("✅ Loaded configuration from config.json")
        else:
            print("⚠️ config.json not found, using default configuration")
            config = get_default_config()
    except Exception as error:
        print(f"❌ Failed to load config.json: {error}")
        print("Using default configuration")
        config = get_default_config()

    # Override with environment variables
    return override_with_env_vars(config)


def get_default_config():
    """Returns default configuration"""
    return {
        "mqtt": {
            "enabled": True,
            "broker": "mqtt://localhost:1883",
            "topic": "card-detected",
            "username": "",
            "password": "",
            "client_id": "iris-device-001",
            "keepalive": 60,
        },
        "camera": {
            "device": "auto",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        "detection": {"confidence_threshold": 0.7, "cooldown_ms": 2000},
    }


def override_with_env_vars(config):
    """Overrides configuration with environment variables"""
    env_overrides = {
        "MQTT_BROKER": "mqtt.broker",
        "MQTT_TOPIC": "mqtt.topic",
        "MQTT_USERNAME": "mqtt.username",
        "MQTT_PASSWORD": "mqtt.password",
        "MQTT_CLIENT_ID": "mqtt.client_id",
        "CAMERA_WIDTH": "camera.width",
        "CAMERA_HEIGHT": "camera.height",
        "CAMERA_FPS": "camera.fps",
    }

    for env_var, config_path in env_overrides.items():
        if os.getenv(env_var):
            keys = config_path.split(".")
            current = config
            for i in range(len(keys) - 1):
                current = current[keys[i]]
            last_key = keys[-1]
            value = os.getenv(env_var)

            # Convert string values to appropriate types
            if last_key in ["width", "height", "fps", "keepalive", "cooldown_ms"]:
                current[last_key] = int(value)
            elif last_key in ["confidence_threshold"]:
                current[last_key] = float(value)
            else:
                current[last_key] = value

    return config
