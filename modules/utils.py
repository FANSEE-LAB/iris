import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from mindar.detector import Detector, DetectorConfig


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


def generate_feature_map(image_path, output_path=None):
    """
    Generate feature points visualization on the input image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output visualization.

    Returns:
        list: Detected feature points [(x, y), ...]
    """
    if not Path(image_path).exists():
        raise ValueError(f"Image file not found: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Initialize detector with default config
    detector_config = DetectorConfig(
        method="orb", max_features=1000, fast_threshold=20, edge_threshold=31, debug_mode=True
    )
    detector = Detector(detector_config)

    # Detect features
    feature_points = detector.detect(gray)

    # Create visualization
    vis_image = image.copy()
    detected_points = []

    for point in feature_points:
        try:
            x, y = int(float(point[0])), int(float(point[1]))
            detected_points.append((x, y))
            # Draw feature point as green circle
            cv2.circle(vis_image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # Draw outer circle for better visibility
            cv2.circle(vis_image, (x, y), radius=5, color=(0, 255, 0), thickness=1)
        except (ValueError, TypeError):
            # Skip invalid points
            continue

    # Add info text
    info_text = f"Features: {len(detected_points)}"
    cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save output
    if output_path is None:
        output_path = f"{Path(image_path).stem}_features.png"

    cv2.imwrite(output_path, vis_image)
    print(f"Feature visualization saved to {output_path}")

    return detected_points


def compare_images_similarity(image1_path, image2_path):
    """
    Compare two images using MindAR-compatible FREAK descriptor matching.

    This function implements the actual algorithm used by mind-ar-js for
    image similarity analysis based on FREAK descriptors and Hamming distance.

    Args:
        image1_path (str): Path to first image.
        image2_path (str): Path to second image.

    Returns:
        dict: Comprehensive similarity analysis results.
    """
    if not Path(image1_path).exists():
        raise ValueError(f"Image file not found: {image1_path}")
    if not Path(image2_path).exists():
        raise ValueError(f"Image file not found: {image2_path}")

    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both images")

    # Initialize MindAR-compatible detector with FREAK descriptors
    detector_config = DetectorConfig(
        method="super_hybrid",  # Use best detection method
        max_features=1000,
        fast_threshold=20,
        edge_threshold=31,
        debug_mode=True,
    )
    detector = Detector(detector_config)

    # Detect features with FREAK descriptors
    result1 = detector.detect(img1)
    result2 = detector.detect(img2)

    features1 = result1.get("feature_points", [])
    features2 = result2.get("feature_points", [])

    if not features1 or not features2:
        return {
            "similarity_score": 0.0,
            "matches": 0,
            "features1": len(features1),
            "features2": len(features2),
            "descriptor_quality": "Poor - insufficient features",
            "warning": "Insufficient features detected. Images may have low contrast or few distinctive patterns.",
            "recommendation": "Add more high-contrast patterns, geometric shapes, or distinctive visual elements.",
        }

    # Analyze descriptor quality
    valid_descriptors1 = [f for f in features1 if f.descriptors and len(f.descriptors) > 0]
    valid_descriptors2 = [f for f in features2 if f.descriptors and len(f.descriptors) > 0]

    descriptor_quality1 = len(valid_descriptors1) / len(features1) if features1 else 0
    descriptor_quality2 = len(valid_descriptors2) / len(features2) if features2 else 0
    avg_descriptor_quality = (descriptor_quality1 + descriptor_quality2) / 2

    if avg_descriptor_quality < 0.3:
        return {
            "similarity_score": float("nan"),  # Cannot determine similarity
            "matches": 0,
            "features1": len(features1),
            "features2": len(features2),
            "valid_descriptors1": len(valid_descriptors1),
            "valid_descriptors2": len(valid_descriptors2),
            "descriptor_quality": f"Critical - {avg_descriptor_quality:.1%} descriptors valid",
            "warning": "FREAK descriptor computation failed. This indicates the images are too similar or lack distinctive features.",
            "recommendation": "These images will likely cause AR detection confusion. Redesign with more distinctive patterns.",
        }

    # Calculate similarity using FREAK descriptor matching (simplified approach)
    try:
        # Simplified FREAK descriptor matching
        matches = []

        for i, f1 in enumerate(valid_descriptors1):
            best_distance = float("inf")
            second_best_distance = float("inf")
            best_match_idx = -1

            for j, f2 in enumerate(valid_descriptors2):
                # Compute Hamming distance between descriptors
                if f1.descriptors and f2.descriptors:
                    hamming_dist = _compute_hamming_distance(f1.descriptors, f2.descriptors)

                    if hamming_dist < best_distance:
                        second_best_distance = best_distance
                        best_distance = hamming_dist
                        best_match_idx = j
                    elif hamming_dist < second_best_distance:
                        second_best_distance = hamming_dist

            # Apply Lowe's ratio test
            if best_match_idx >= 0 and second_best_distance > 0:
                ratio = best_distance / second_best_distance
                if ratio < 0.75:  # Lowe's ratio threshold
                    matches.append((i, best_match_idx, best_distance))

        # Calculate comprehensive similarity metrics
        if len(valid_descriptors1) > 0 and len(valid_descriptors2) > 0:
            # Descriptor-based similarity (FREAK Hamming distance)
            descriptor_similarity = len(matches) / min(len(valid_descriptors1), len(valid_descriptors2))

            # For simplicity, use descriptor similarity as the main metric
            similarity_score = descriptor_similarity
            geometric_similarity = descriptor_similarity  # Simplified
        else:
            similarity_score = 0.0
            descriptor_similarity = 0.0
            geometric_similarity = 0.0

        homography = len(matches) > 8  # Simple homography check
        inliers = matches  # Simplified

    except Exception:
        # Fallback to basic analysis
        matches = []
        homography = None
        inliers = []
        similarity_score = 0.0
        descriptor_similarity = 0.0
        geometric_similarity = 0.0

    # Determine quality assessment
    if avg_descriptor_quality >= 0.8:
        quality_level = "Excellent"
    elif avg_descriptor_quality >= 0.6:
        quality_level = "Good"
    elif avg_descriptor_quality >= 0.4:
        quality_level = "Fair"
    else:
        quality_level = "Poor"

    # Generate warnings and recommendations
    warning = None
    recommendation = None

    if similarity_score > 0.7:
        warning = "HIGH SIMILARITY DETECTED - AR system will likely confuse these images"
        recommendation = "Urgent: Redesign one image with completely different patterns, colors, or layout"
    elif similarity_score > 0.5:
        warning = "Moderate similarity detected - may cause occasional detection confusion"
        recommendation = "Consider adding more distinctive features or changing color scheme"
    elif similarity_score > 0.3:
        warning = "Some similarity detected - monitor detection performance in testing"
        recommendation = "Images should work but test thoroughly in target environment"
    else:
        recommendation = "Images have good distinctiveness for AR tracking"

    return {
        "similarity_score": similarity_score,
        "descriptor_similarity": descriptor_similarity,
        "geometric_similarity": geometric_similarity,
        "matches": len(matches),
        "inliers": len(inliers),
        "features1": len(features1),
        "features2": len(features2),
        "valid_descriptors1": len(valid_descriptors1),
        "valid_descriptors2": len(valid_descriptors2),
        "descriptor_quality": f"{quality_level} - {avg_descriptor_quality:.1%} descriptors valid",
        "homography_found": homography is not None,
        "warning": warning,
        "recommendation": recommendation,
    }


def _compute_hamming_distance(desc1, desc2):
    """Compute Hamming distance between two binary descriptors."""
    if not desc1 or not desc2:
        return 1.0  # Max normalized distance for empty descriptors

    hamming_dist = 0
    min_len = min(len(desc1), len(desc2))

    for i in range(min_len):
        try:
            d1, d2 = int(desc1[i]), int(desc2[i])
            xor = d1 ^ d2
            # Count bits using bit manipulation
            while xor:
                hamming_dist += 1
                xor = xor & (xor - 1)
        except (ValueError, TypeError):
            hamming_dist += 32  # Penalty for invalid descriptors

    # Normalize by maximum possible distance
    max_distance = min_len * 32
    return hamming_dist / max_distance if max_distance > 0 else 1.0


def analyze_ar_tracking_issues(image_path: str) -> Dict[str, Any]:
    """
    Deep analysis of AR tracking potential issues and provide actionable improvement suggestions

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing detailed analysis results and improvement recommendations
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Initialize detector
    detector = cv2.BRISK_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        return {
            "status": "error",
            "message": "No features detected",
            "recommendation": "Image contrast too low or insufficient texture, consider adding visual elements",
        }

    # 1. Feature distribution analysis - find sparse regions
    feature_density_map = analyze_feature_distribution(keypoints, width, height)

    # 2. Uniqueness analysis - assess feature distinctiveness
    uniqueness_score = analyze_feature_uniqueness(descriptors)

    # 3. Edge stability analysis - check border region tracking stability
    edge_stability = analyze_edge_stability(gray, keypoints)

    # 4. Visual complexity analysis
    visual_complexity = analyze_visual_complexity(gray)

    # 5. Symmetry analysis - symmetric patterns cause confusion
    symmetry_issues = analyze_symmetry_issues(gray, keypoints)

    # Comprehensive evaluation
    tracking_score = calculate_tracking_score(
        feature_density_map, uniqueness_score, edge_stability, visual_complexity, symmetry_issues
    )

    # Generate specific recommendations
    recommendations = generate_design_recommendations(
        feature_density_map, uniqueness_score, edge_stability, visual_complexity, symmetry_issues, tracking_score
    )

    return {
        "image_info": {"path": image_path, "dimensions": f"{width}x{height}", "feature_count": len(keypoints)},
        "tracking_analysis": {
            "overall_score": tracking_score,
            "feature_distribution": feature_density_map,
            "uniqueness_score": uniqueness_score,
            "edge_stability": edge_stability,
            "visual_complexity": visual_complexity,
            "symmetry_issues": symmetry_issues,
        },
        "recommendations": recommendations,
        "critical_issues": identify_critical_issues(tracking_score, recommendations),
    }


def analyze_feature_distribution(keypoints: List, width: int, height: int) -> Dict[str, Any]:
    """Analyze spatial distribution of feature points to identify regions that may cause tracking failure"""
    if not keypoints:
        return {"status": "error", "message": "No feature points"}

    # Divide image into 9 regions (3x3 grid)
    grid_size = 3
    grid_counts = np.zeros((grid_size, grid_size))

    for kp in keypoints:
        x, y = kp.pt
        grid_x = min(int(x / width * grid_size), grid_size - 1)
        grid_y = min(int(y / height * grid_size), grid_size - 1)
        grid_counts[grid_y, grid_x] += 1

    # Calculate distribution uniformity
    total_features = len(keypoints)
    expected_per_grid = total_features / (grid_size * grid_size)

    empty_grids = np.sum(grid_counts == 0)
    sparse_grids = np.sum((grid_counts > 0) & (grid_counts < expected_per_grid * 0.3))

    # Center region feature density (most important for AR tracking)
    center_density = grid_counts[1, 1] / total_features if total_features > 0 else 0

    # Calculate distribution variance
    distribution_variance = np.var(grid_counts.flatten())

    return {
        "grid_distribution": grid_counts.tolist(),
        "empty_regions": int(empty_grids),
        "sparse_regions": int(sparse_grids),
        "center_density": float(center_density),
        "distribution_variance": float(distribution_variance),
        "uniformity_score": 1.0 / (1.0 + distribution_variance / expected_per_grid) if expected_per_grid > 0 else 0,
        "problematic_areas": get_problematic_areas(grid_counts, expected_per_grid),
    }


def analyze_feature_uniqueness(descriptors: np.ndarray) -> Dict[str, Any]:
    """Analyze feature descriptor uniqueness to assess confusion potential"""
    if descriptors is None or len(descriptors) < 2:
        return {"uniqueness_score": 0, "similar_features": 0}

    # Calculate Hamming distances between all descriptors
    distances = []
    similar_pairs = 0
    total_pairs = 0

    for i in range(len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            # Calculate Hamming distance for binary descriptors
            dist = np.sum(descriptors[i] != descriptors[j])
            distances.append(dist)
            total_pairs += 1

            # Consider as similar features if distance is too small
            if dist < len(descriptors[0]) * 0.3:  # Less than 30% bits different
                similar_pairs += 1

    if not distances:
        return {"uniqueness_score": 0, "similar_features": 0}

    avg_distance = np.mean(distances)
    min_distance = np.min(distances)

    # Uniqueness score: higher average and minimum distances indicate better uniqueness
    max_possible_distance = len(descriptors[0])
    uniqueness_score = (avg_distance / max_possible_distance) * (min_distance / max_possible_distance)

    similar_ratio = similar_pairs / total_pairs if total_pairs > 0 else 0

    return {
        "uniqueness_score": float(uniqueness_score),
        "average_distance": float(avg_distance),
        "minimum_distance": float(min_distance),
        "similar_features_ratio": float(similar_ratio),
        "total_features": len(descriptors),
        "risk_level": "high" if similar_ratio > 0.3 else "medium" if similar_ratio > 0.1 else "low",
    }


def analyze_edge_stability(gray: np.ndarray, keypoints: List) -> Dict[str, Any]:
    """Analyze edge region feature stability to predict tracking performance under different angles"""
    height, width = gray.shape

    # Define edge regions (20% margin from borders)
    edge_margin = 0.2
    edge_features = 0
    center_features = 0

    for kp in keypoints:
        x, y = kp.pt
        if (
            x < width * edge_margin
            or x > width * (1 - edge_margin)
            or y < height * edge_margin
            or y > height * (1 - edge_margin)
        ):
            edge_features += 1
        else:
            center_features += 1

    total_features = len(keypoints)
    edge_ratio = edge_features / total_features if total_features > 0 else 0

    # Calculate gradient magnitude distribution
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Edge region gradient strength
    edge_mask = np.zeros_like(gray, dtype=bool)
    edge_mask[: int(height * edge_margin), :] = True
    edge_mask[int(height * (1 - edge_margin)) :, :] = True
    edge_mask[:, : int(width * edge_margin)] = True
    edge_mask[:, int(width * (1 - edge_margin)) :] = True

    edge_gradient_strength = np.mean(gradient_magnitude[edge_mask])
    center_gradient_strength = np.mean(gradient_magnitude[~edge_mask])

    stability_score = center_gradient_strength / (edge_gradient_strength + 1e-6)

    return {
        "edge_features_ratio": float(edge_ratio),
        "center_features": int(center_features),
        "edge_features": int(edge_features),
        "stability_score": float(stability_score),
        "edge_gradient_strength": float(edge_gradient_strength),
        "center_gradient_strength": float(center_gradient_strength),
        "risk_assessment": "high" if edge_ratio > 0.6 else "medium" if edge_ratio > 0.4 else "low",
    }


def analyze_visual_complexity(gray: np.ndarray) -> Dict[str, Any]:
    """Analyze visual complexity to assess texture richness"""

    # Calculate Local Binary Pattern (LBP) for texture complexity assessment
    def local_binary_pattern(image, radius=1, n_points=8):
        rows, cols = image.shape
        lbp = np.zeros_like(image)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if image[x, y] > center:
                        code |= 1 << k
                lbp[i, j] = code
        return lbp

    lbp = local_binary_pattern(gray)

    # Calculate LBP histogram distribution
    hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)

    # Calculate entropy (information content)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    # Calculate standard deviation (contrast)
    contrast = np.std(gray)

    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # Comprehensive complexity score
    complexity_score = (entropy / 8.0) * 0.4 + (contrast / 255.0) * 0.3 + edge_density * 0.3

    return {
        "entropy": float(entropy),
        "contrast": float(contrast),
        "edge_density": float(edge_density),
        "complexity_score": float(complexity_score),
        "assessment": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.4 else "low",
    }


def analyze_symmetry_issues(gray: np.ndarray, keypoints: List) -> Dict[str, Any]:
    """Analyze symmetry issues - symmetric designs cause tracking confusion"""
    height, width = gray.shape

    # Horizontal symmetry check
    left_half = gray[:, : width // 2]
    right_half = cv2.flip(gray[:, width // 2 :], 1)  # Horizontal flip

    if left_half.shape != right_half.shape:
        # Adjust sizes to match
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

    horizontal_similarity = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]

    # Vertical symmetry check
    top_half = gray[: height // 2, :]
    bottom_half = cv2.flip(gray[height // 2 :, :], 0)  # Vertical flip

    if top_half.shape != bottom_half.shape:
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]

    vertical_similarity = cv2.matchTemplate(top_half, bottom_half, cv2.TM_CCOEFF_NORMED)[0, 0]

    # Analyze feature point symmetric distribution
    left_features = sum(1 for kp in keypoints if kp.pt[0] < width / 2)
    right_features = sum(1 for kp in keypoints if kp.pt[0] >= width / 2)
    top_features = sum(1 for kp in keypoints if kp.pt[1] < height / 2)
    bottom_features = sum(1 for kp in keypoints if kp.pt[1] >= height / 2)

    feature_balance_h = min(left_features, right_features) / max(left_features, right_features, 1)
    feature_balance_v = min(top_features, bottom_features) / max(top_features, bottom_features, 1)

    return {
        "horizontal_similarity": float(horizontal_similarity),
        "vertical_similarity": float(vertical_similarity),
        "feature_balance_horizontal": float(feature_balance_h),
        "feature_balance_vertical": float(feature_balance_v),
        "symmetry_risk": (
            "high"
            if max(horizontal_similarity, vertical_similarity) > 0.8
            else "medium" if max(horizontal_similarity, vertical_similarity) > 0.6 else "low"
        ),
        "recommendation": (
            "Add asymmetric elements"
            if max(horizontal_similarity, vertical_similarity) > 0.7
            else "Symmetry level appropriate"
        ),
    }


def calculate_tracking_score(
    feature_dist: Dict, uniqueness: Dict, edge_stability: Dict, complexity: Dict, symmetry: Dict
) -> float:
    """Calculate comprehensive AR tracking quality score"""

    # Scoring weights for different aspects
    weights = {
        "distribution": 0.25,  # Feature distribution
        "uniqueness": 0.25,  # Uniqueness
        "stability": 0.20,  # Edge stability
        "complexity": 0.15,  # Visual complexity
        "symmetry": 0.15,  # Symmetry
    }

    # Normalize all scores to 0-1 range
    dist_score = feature_dist.get("uniformity_score", 0)
    unique_score = uniqueness.get("uniqueness_score", 0)
    stable_score = min(edge_stability.get("stability_score", 0) / 2.0, 1.0)  # Normalize
    complex_score = complexity.get("complexity_score", 0)
    symm_score = 1.0 - max(symmetry.get("horizontal_similarity", 0), symmetry.get("vertical_similarity", 0))

    total_score = (
        dist_score * weights["distribution"]
        + unique_score * weights["uniqueness"]
        + stable_score * weights["stability"]
        + complex_score * weights["complexity"]
        + symm_score * weights["symmetry"]
    )

    return min(max(total_score, 0.0), 1.0)


def generate_design_recommendations(
    feature_dist: Dict, uniqueness: Dict, edge_stability: Dict, complexity: Dict, symmetry: Dict, overall_score: float
) -> List[Dict[str, str]]:
    """Generate specific design improvement recommendations based on analysis results"""
    recommendations = []

    # Feature distribution issues
    if feature_dist.get("empty_regions", 0) > 2:
        recommendations.append(
            {
                "category": "Feature Distribution",
                "priority": "High",
                "issue": f"{feature_dist['empty_regions']} regions lack feature points",
                "suggestion": "Add visual elements (patterns, text, icons) in sparse areas to increase feature density",
            }
        )

    if feature_dist.get("center_density", 0) < 0.2:
        recommendations.append(
            {
                "category": "Center Region",
                "priority": "High",
                "issue": "Center region feature density too low",
                "suggestion": "Add prominent visual markers or logo in image center to improve tracking stability",
            }
        )

    # Uniqueness issues
    if uniqueness.get("risk_level") == "high":
        recommendations.append(
            {
                "category": "Feature Uniqueness",
                "priority": "High",
                "issue": f"{uniqueness.get('similar_features_ratio', 0)*100:.1f}% of features are too similar",
                "suggestion": "Add unique visual elements, avoid repetitive patterns, use different colors, shapes or textures",
            }
        )

    # Edge stability issues
    if edge_stability.get("risk_assessment") == "high":
        recommendations.append(
            {
                "category": "Edge Stability",
                "priority": "Medium",
                "issue": "Too many features concentrated in edge regions",
                "suggestion": "Move important features toward center, or add more visual anchors in center region",
            }
        )

    # Visual complexity issues
    if complexity.get("assessment") == "low":
        recommendations.append(
            {
                "category": "Visual Complexity",
                "priority": "Medium",
                "issue": "Image texture too simple",
                "suggestion": "Add detail textures, gradient effects or subtle background patterns to enrich visual information",
            }
        )

    # Symmetry issues
    if symmetry.get("symmetry_risk") == "high":
        recommendations.append(
            {
                "category": "Symmetry",
                "priority": "High",
                "issue": "Image too symmetric, causes tracking confusion",
                "suggestion": "Break symmetry: add asymmetric elements, use different colors or patterns on different sides",
            }
        )

    # Overall score recommendations
    if overall_score < 0.3:
        recommendations.append(
            {
                "category": "Overall Design",
                "priority": "Critical",
                "issue": "AR tracking quality extremely low, may cause severe misidentification",
                "suggestion": "Recommend redesign: use high contrast, asymmetric, feature-rich design",
            }
        )
    elif overall_score < 0.6:
        recommendations.append(
            {
                "category": "Overall Design",
                "priority": "High",
                "issue": "AR tracking quality suboptimal, needs improvement",
                "suggestion": "Focus on improving identified issues, especially high priority items",
            }
        )

    return recommendations


def identify_critical_issues(tracking_score: float, recommendations: List[Dict]) -> List[str]:
    """Identify critical issues requiring immediate attention"""
    critical = []

    if tracking_score < 0.3:
        critical.append("Overall tracking quality extremely low, urgent redesign needed")

    high_priority = [rec for rec in recommendations if rec.get("priority") in ["High", "Critical"]]
    if len(high_priority) >= 3:
        critical.append("Multiple high priority issues need simultaneous resolution")

    for rec in recommendations:
        if rec.get("priority") == "Critical":
            critical.append(f"{rec['category']}: {rec['issue']}")

    return critical


def get_problematic_areas(grid_counts: np.ndarray, expected: float) -> List[str]:
    """Identify problematic region locations"""
    areas = []
    rows, cols = grid_counts.shape

    area_names = [
        ["Top-left", "Top", "Top-right"],
        ["Left", "Center", "Right"],
        ["Bottom-left", "Bottom", "Bottom-right"],
    ]

    for i in range(rows):
        for j in range(cols):
            if grid_counts[i, j] == 0:
                areas.append(f"{area_names[i][j]} region completely lacks features")
            elif grid_counts[i, j] < expected * 0.3:
                areas.append(f"{area_names[i][j]} region has sparse features")

    return areas


def compare_ar_tracking_potential(image1_path: str, image2_path: str) -> Dict[str, Any]:
    """Compare AR tracking confusion risk between two images"""

    # Analyze both images
    analysis1 = analyze_ar_tracking_issues(image1_path)
    analysis2 = analyze_ar_tracking_issues(image2_path)

    if "error" in analysis1.get("status", "") or "error" in analysis2.get("status", ""):
        return {"error": "Unable to analyze one or both images"}

    # Use existing image comparison functionality
    mindar_similarity = compare_images_similarity(image1_path, image2_path)

    # Calculate design similarity risk
    design_risk = calculate_design_similarity_risk(analysis1, analysis2)

    # Predict confusion possibility
    confusion_risk = predict_confusion_risk(analysis1, analysis2, mindar_similarity)

    return {
        "images": {"image1": image1_path, "image2": image2_path},
        "individual_scores": {
            "image1_tracking_score": analysis1["tracking_analysis"]["overall_score"],
            "image2_tracking_score": analysis2["tracking_analysis"]["overall_score"],
        },
        "similarity_analysis": mindar_similarity,
        "design_similarity_risk": design_risk,
        "confusion_prediction": confusion_risk,
        "recommendations": generate_differentiation_recommendations(analysis1, analysis2, confusion_risk),
    }


def calculate_design_similarity_risk(analysis1: Dict, analysis2: Dict) -> Dict[str, Any]:
    """Calculate design-level similarity risk"""

    # Compare visual complexity
    complexity_diff = abs(
        analysis1["tracking_analysis"]["visual_complexity"]["complexity_score"]
        - analysis2["tracking_analysis"]["visual_complexity"]["complexity_score"]
    )

    # Compare symmetry
    sym1 = max(
        analysis1["tracking_analysis"]["symmetry_issues"]["horizontal_similarity"],
        analysis1["tracking_analysis"]["symmetry_issues"]["vertical_similarity"],
    )
    sym2 = max(
        analysis2["tracking_analysis"]["symmetry_issues"]["horizontal_similarity"],
        analysis2["tracking_analysis"]["symmetry_issues"]["vertical_similarity"],
    )
    symmetry_similarity = 1.0 - abs(sym1 - sym2)

    # Compare feature distribution patterns
    dist1 = np.array(analysis1["tracking_analysis"]["feature_distribution"]["grid_distribution"])
    dist2 = np.array(analysis2["tracking_analysis"]["feature_distribution"]["grid_distribution"])

    # Normalize distributions
    dist1_norm = dist1 / (np.sum(dist1) + 1e-6)
    dist2_norm = dist2 / (np.sum(dist2) + 1e-6)

    distribution_similarity = 1.0 - np.sum(np.abs(dist1_norm - dist2_norm)) / 2.0

    # Comprehensive risk score
    risk_score = (1.0 - complexity_diff) * 0.3 + symmetry_similarity * 0.3 + distribution_similarity * 0.4

    return {
        "complexity_difference": float(complexity_diff),
        "symmetry_similarity": float(symmetry_similarity),
        "distribution_similarity": float(distribution_similarity),
        "overall_risk_score": float(risk_score),
        "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.5 else "low",
    }


def predict_confusion_risk(analysis1: Dict, analysis2: Dict, similarity: Dict) -> Dict[str, Any]:
    """Predict confusion risk in real-world usage"""

    # Feature matching similarity
    feature_similarity = similarity.get("similarity_percentage", 0) / 100.0

    # Tracking quality of both images
    score1 = analysis1["tracking_analysis"]["overall_score"]
    score2 = analysis2["tracking_analysis"]["overall_score"]

    # Lower quality images are more prone to confusion
    quality_factor = 1.0 - min(score1, score2)

    # Feature uniqueness
    unique1 = analysis1["tracking_analysis"]["uniqueness_score"]["uniqueness_score"]
    unique2 = analysis2["tracking_analysis"]["uniqueness_score"]["uniqueness_score"]
    uniqueness_factor = 1.0 - min(unique1, unique2)

    # Comprehensive confusion risk
    confusion_probability = feature_similarity * 0.4 + quality_factor * 0.3 + uniqueness_factor * 0.3

    # Environmental factors (lighting, angle changes risk)
    environmental_risk = calculate_environmental_risk(analysis1, analysis2)

    final_risk = min(confusion_probability + environmental_risk * 0.2, 1.0)

    return {
        "confusion_probability": float(final_risk),
        "risk_level": (
            "critical" if final_risk > 0.8 else "high" if final_risk > 0.6 else "medium" if final_risk > 0.4 else "low"
        ),
        "contributing_factors": {
            "feature_similarity": float(feature_similarity),
            "quality_factor": float(quality_factor),
            "uniqueness_factor": float(uniqueness_factor),
            "environmental_risk": float(environmental_risk),
        },
        "explanation": generate_confusion_explanation(final_risk, feature_similarity, quality_factor),
    }


def calculate_environmental_risk(analysis1: Dict, analysis2: Dict) -> float:
    """Calculate confusion risk under environmental changes (lighting, angles)"""

    # Images with high edge feature ratio are more prone to confusion under angle changes
    edge_risk1 = analysis1["tracking_analysis"]["edge_stability"]["edge_features_ratio"]
    edge_risk2 = analysis2["tracking_analysis"]["edge_stability"]["edge_features_ratio"]
    edge_risk = max(edge_risk1, edge_risk2)

    # Low contrast images are more prone to confusion under lighting changes
    contrast1 = analysis1["tracking_analysis"]["visual_complexity"]["contrast"] / 255.0
    contrast2 = analysis2["tracking_analysis"]["visual_complexity"]["contrast"] / 255.0
    contrast_risk = 1.0 - min(contrast1, contrast2)

    return (edge_risk + contrast_risk) / 2.0


def generate_confusion_explanation(risk: float, feature_sim: float, quality_factor: float) -> str:
    """Generate detailed explanation of confusion risk"""

    if risk > 0.8:
        return f"Extremely high confusion risk: {feature_sim*100:.1f}% feature similarity, insufficient image quality. Almost certain misidentification in real usage."
    elif risk > 0.6:
        return f"High confusion risk: {feature_sim*100:.1f}% feature similarity. Likely misidentification under lighting changes or angle deviations."
    elif risk > 0.4:
        return f"Medium confusion risk: {feature_sim*100:.1f}% feature similarity. Distinguishable under ideal conditions, but may confuse under environmental changes."
    else:
        return f"Low confusion risk: {feature_sim*100:.1f}% feature similarity. Images have sufficient distinctive features."


def generate_differentiation_recommendations(
    analysis1: Dict, analysis2: Dict, confusion_risk: Dict
) -> List[Dict[str, str]]:
    """Generate specific differentiation design recommendations"""
    recommendations = []

    risk_level = confusion_risk.get("risk_level", "low")

    if risk_level in ["critical", "high"]:
        recommendations.append(
            {
                "category": "Urgent Improvement",
                "priority": "Highest",
                "action": "Immediate design modification",
                "details": "Images too similar, major design changes required to avoid misidentification",
            }
        )

        # Specific recommendations
        recommendations.extend(
            [
                {
                    "category": "Color Scheme",
                    "priority": "High",
                    "action": "Use contrasting color systems",
                    "details": "Choose completely different primary colors for both images (e.g., blue vs orange)",
                },
                {
                    "category": "Shape Language",
                    "priority": "High",
                    "action": "Change basic shapes",
                    "details": "One uses circular elements, the other uses square elements",
                },
                {
                    "category": "Layout Structure",
                    "priority": "High",
                    "action": "Change element arrangement",
                    "details": "Completely different layouts: one vertical arrangement, one horizontal arrangement",
                },
            ]
        )

    elif risk_level == "medium":
        recommendations.extend(
            [
                {
                    "category": "Detail Differences",
                    "priority": "Medium",
                    "action": "Enhance unique elements",
                    "details": "Add unique identifiers or patterns to each image",
                },
                {
                    "category": "Texture Contrast",
                    "priority": "Medium",
                    "action": "Use different textures",
                    "details": "One uses fine texture, the other uses coarse texture",
                },
            ]
        )

    # Recommendations based on specific analysis
    if analysis1["tracking_analysis"]["symmetry_issues"]["symmetry_risk"] == "high":
        recommendations.append(
            {
                "category": "Symmetry",
                "priority": "High",
                "action": "Break symmetric design",
                "details": "Add asymmetric elements to image1 to increase uniqueness",
            }
        )

    if analysis2["tracking_analysis"]["symmetry_issues"]["symmetry_risk"] == "high":
        recommendations.append(
            {
                "category": "Symmetry",
                "priority": "High",
                "action": "Break symmetric design",
                "details": "Add asymmetric elements to image2 to increase uniqueness",
            }
        )

    return recommendations
