import json
import paho.mqtt.client as mqtt

class MQTTHandler:
    def __init__(self, config):
        self.config = config
        self.client = None

    def init_mqtt(self):
        """Initialize MQTT client connection"""
        if not self.config["mqtt"]["enabled"]:
            print("📡 MQTT disabled in configuration")
            return

        self.client = mqtt.Client(client_id=self.config["mqtt"]["client_id"])

        if self.config["mqtt"]["username"]:
            self.client.username_pw_set(self.config["mqtt"]["username"], self.config["mqtt"]["password"])

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        try:
            self.client.connect(self.config["mqtt"]["broker"].replace("mqtt://", ""), keepalive=self.config["mqtt"]["keepalive"])
            self.client.loop_start()
        except Exception as e:
            print(f"❌ MQTT connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print("✅ MQTT connection successful")
        else:
            print(f"❌ MQTT connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        print("🔌 MQTT connection closed")

    def send_message(self, topic, message):
        """Send MQTT message"""
        if not self.client or not self.client.is_connected():
            print("❌ MQTT not connected")
            return

        self.client.publish(topic, json.dumps(message))
        print(f"📤 Sent MQTT message to {topic}: {message}")
