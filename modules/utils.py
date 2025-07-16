import logging

import cv2
import numpy as np


def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def generate_feature_visualization(image_path, feature_points, output_path):
    """
    Generate a visualization of feature points on the input image.

    Args:
        image_path (str): Path to the input image.
        feature_points (list): List of feature points (x, y).
        output_path (str): Path to save the output visualization.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    for point in feature_points:
        x, y = int(point["x"]), int(point["y"])
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(output_path, image)
    print(f"Feature visualization saved to {output_path}")


def analyze_image_design(image_path):
    """
    Analyze the input image and provide suggestions for improvement.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Analysis results and suggestions.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Detect edges
    edges = cv2.Canny(image, 100, 200)

    # Count edge pixels
    edge_pixel_count = np.sum(edges > 0)
    total_pixels = image.shape[0] * image.shape[1]
    edge_density = edge_pixel_count / total_pixels

    suggestions = []
    if edge_density < 0.05:
        suggestions.append("Increase contrast or add more distinct features.")
    elif edge_density > 0.2:
        suggestions.append("Reduce noise or simplify the design.")

    return {
        "edge_density": edge_density,
        "suggestions": suggestions,
    }
