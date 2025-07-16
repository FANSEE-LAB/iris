# Iris

A universal framework for visual target detection using computer vision, designed for exhibition and interactive installations.

## Features

- **Universal Architecture**: Configurable for different use cases
- **Multiple Detection Engines**: Support for ORB, SIFT, SURF algorithms
- **MQTT Integration**: Real-time communication with other systems
- **Performance Optimization**: Optimized for Raspberry Pi and embedded systems
- **Simulation Mode**: Test without physical camera
- **Comprehensive Logging**: Detailed system monitoring

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install from PyPI
pip install mindar==0.1.2 opencv-python paho-mqtt PyYAML
```

### 2. Configuration

Copy and modify the configuration file:

```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml for your specific setup
```

### 3. Prepare Target Images

Place your target images in the `images/` directory and update the configuration:

```yaml
targets:
  directory: "images"
  targets:
    - id: 0
      name: "target_1"
      file: "target_1.png"
      description: "First target"
```

### 4. Run the System

```bash
# Normal mode
python ar_target_detector.py

# With custom config
python ar_target_detector.py --config my_config.yaml

# Simulation mode (no camera)
python ar_target_detector.py --simulation
```

## Configuration

The system uses YAML configuration for easy customization:

### Camera Settings

```yaml
camera:
  enabled: true
  width: 1280
  height: 960
  fps: 8
```

### Detection Parameters

```yaml
detection:
  engine: "orb" # orb, sift, surf
  max_features: 50
  confidence_threshold: 0.8
  min_matches: 10
```

### MQTT Integration

```yaml
mqtt:
  enabled: true
  broker: "localhost"
  topics:
    detection: "ar/detection"
    status: "ar/status"
```

## Architecture

```
ar_target_detector.py     # Main system entry point
├── modules/
│   ├── camera.py         # Camera controller
│   ├── mqtt_handler.py   # MQTT communication
│   └── utils.py          # Utilities and logging
├── images/               # Target images
├── config.yaml           # System configuration
└── requirements.txt      # Dependencies
```

## Usage Examples

### Basic Exhibition Setup

1. Configure your targets in `config.yaml`
2. Connect MQTT broker for receiving detection events
3. Run: `python ar_target_detector.py`

### Development and Testing

```bash
# Test with simulation mode
python ar_target_detector.py --simulation

# Debug mode with detailed logging
# Set log_level: "DEBUG" in config.yaml
```

### Custom Integration

```python
from ar_target_detector import ARTargetDetector

# Create detector with custom config
detector = ARTargetDetector('custom_config.yaml')
detector.start()
```

## Performance Tuning

### For Raspberry Pi

- Use `max_features: 50` or less
- Set `engine: "orb"` for best performance
- Use lower camera resolution for faster processing

### For Desktop/Server

- Increase `max_features: 200+`
- Try `engine: "sift"` for better accuracy
- Use higher resolution cameras

## MQTT Messages

### Detection Events

```json
{
  "timestamp": 1703123456.789,
  "target_id": 0,
  "target_name": "card-a",
  "confidence": 0.85,
  "matches": 15
}
```

### System Status

```json
{
  "timestamp": 1703123456.789,
  "system": "Iris Exhibition Device",
  "status": "running",
  "uptime": 3600.5,
  "detection_count": 42
}
```

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and connections
2. **Slow detection**: Reduce `max_features` and image resolution
3. **False positives**: Increase `confidence_threshold` and `min_matches`
4. **MQTT connection failed**: Check broker settings and network

### Performance Optimization

- Use native camera resolution when possible
- Optimize feature count vs. accuracy trade-off
- Consider image quality and lighting conditions
- Test different detection engines for your use case

## Development

### Adding New Detection Engines

1. Extend the `_setup_detector()` method
2. Add engine-specific configuration in YAML
3. Update documentation

### Custom MQTT Topics

Modify the `mqtt.topics` section in configuration:

```yaml
mqtt:
  topics:
    detection: "custom/detection"
    status: "custom/status"
    heartbeat: "custom/heartbeat"
```

## License

This project is designed for internal use and deployment to exhibition systems.

## Contributing

1. Follow the existing code structure
2. Add comprehensive logging
3. Update configuration documentation
4. Test on target hardware (Raspberry Pi)

## Support

For technical support and deployment assistance, contact the development team.
