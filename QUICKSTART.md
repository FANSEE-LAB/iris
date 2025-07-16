# Iris Quick Start Guide

## 1. Clone and Setup

```bash
git clone [your-repo-url]
cd iris
```

## 2. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## 3. Configuration

```bash
# Copy example configuration
cp config.example.yaml config.yaml

# Edit configuration for your setup
nano config.yaml  # or use your preferred editor
```

### Key Configuration Changes:

1. **Target Images**: Update `targets.targets` section with your image files
2. **MQTT Broker**: Change `mqtt.broker` to your broker IP
3. **Camera Settings**: Adjust `camera` settings for your hardware
4. **Performance**: Tune `detection.max_features` for your device

## 4. Prepare Target Images

```bash
# Create images directory if it doesn't exist
mkdir -p images

# Copy your target images
cp /path/to/your/target_1.png images/
cp /path/to/your/target_2.png images/
cp /path/to/your/target_3.png images/
```

## 5. Test in Simulation Mode

```bash
# Test without camera
poetry run python ar_target_detector.py --simulation
```

## 6. Run with Camera

```bash
# Normal operation
poetry run python ar_target_detector.py

# With custom config
poetry run python ar_target_detector.py --config my_config.yaml
```

## 7. Monitor MQTT Messages

```bash
# Subscribe to detection events
mosquitto_sub -h localhost -t "ar/detection"

# Subscribe to system status
mosquitto_sub -h localhost -t "ar/status"
```

## Troubleshooting

### Common Issues:

1. **Camera not found**:

   - Check camera connections
   - Set `camera.enabled: false` to disable camera
   - Use `--simulation` flag for testing

2. **MQTT connection failed**:

   - Verify broker IP in config
   - Set `mqtt.enabled: false` to disable MQTT

3. **Slow detection**:

   - Reduce `detection.max_features` (try 25-50)
   - Lower camera resolution
   - Use `engine: "orb"` for best performance

4. **No targets detected**:
   - Check image file paths in config
   - Ensure images are clear and well-lit
   - Adjust `detection.confidence_threshold`

### Performance Tuning:

**For Raspberry Pi:**

```yaml
detection:
  max_features: 25
  engine: "orb"
camera:
  width: 640
  height: 480
```

**For Desktop/Server:**

```yaml
detection:
  max_features: 100
  engine: "sift" # or "orb"
camera:
  width: 1280
  height: 960
```

## Next Steps

1. Deploy to your target hardware
2. Configure MQTT integration with your system
3. Optimize detection parameters for your images
4. Set up monitoring and logging

## Support

- Check README.md for detailed documentation
- Review configuration comments in config.example.yaml
- Test with simulation mode first
