#!/usr/bin/env python3
"""
Iris CLI - Command Line Interface for Visual Target Detection System
A modern CLI tool for AR target detection, analysis, and visualization
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

from ar_target_detector import main as ar_main
from ar_target_detector import run_simulation
from modules.utils import (
    analyze_ar_tracking_issues,
    compare_ar_tracking_potential,
    generate_feature_map,
)


def setup_logging(level: str):
    """Setup logging configuration"""
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s - %(levelname)s - %(message)s")


def cmd_detect(args):
    """Command: Start real-time target detection"""
    setup_logging(args.log_level)
    logging.info("Starting AR target detection...")

    # Set config file if provided
    if hasattr(args, "config") and args.config != "config.yaml":
        os.environ["CONFIG_FILE"] = args.config

    try:
        ar_main()
    except KeyboardInterrupt:
        logging.info("Detection stopped by user")
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        sys.exit(1)


def cmd_simulate(args):
    """Command: Run in simulation mode"""
    setup_logging(args.log_level)
    logging.info("Starting simulation mode...")

    # Set config file if provided
    if hasattr(args, "config") and args.config != "config.yaml":
        os.environ["CONFIG_FILE"] = args.config

    try:
        run_simulation()
    except KeyboardInterrupt:
        logging.info("Simulation stopped by user")
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        sys.exit(1)


def cmd_analyze_image(args):
    """Command: Deep analysis of image AR tracking quality"""
    setup_logging(args.log_level)

    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        sys.exit(1)

    try:
        print(f"🔍 Analyzing image: {args.image}")
        result = analyze_ar_tracking_issues(args.image)

        if result.get("status") == "error":
            print(f"❌ Analysis failed: {result.get('message')}")
            print(f"💡 Recommendation: {result.get('recommendation')}")
            return

        # Display basic information
        info = result["image_info"]
        print("\n📊 Image Information:")
        print(f"   Dimensions: {info['dimensions']}")
        print(f"   Feature count: {info['feature_count']}")

        # Display tracking quality score
        analysis = result["tracking_analysis"]
        score = analysis["overall_score"]
        print(f"\n🎯 AR Tracking Quality Score: {score:.2f}/1.00")

        if score >= 0.8:
            print("   ✅ Excellent - AR tracking should be very stable")
        elif score >= 0.6:
            print("   ✅ Good - AR tracking quality acceptable")
        elif score >= 0.4:
            print("   ⚠️  Medium - Some aspects need improvement")
        elif score >= 0.2:
            print("   ❌ Poor - Tracking may be unstable")
        else:
            print("   🚨 Critical - Strong recommendation for redesign")

        # Display detailed analysis
        print("\n📈 Detailed Analysis:")

        # Feature distribution
        dist = analysis["feature_distribution"]
        print(f"   Feature distribution uniformity: {dist['uniformity_score']:.2f}")
        print(f"   Empty regions: {dist['empty_regions']}/9")
        print(f"   Center density: {dist['center_density']:.1%}")
        if dist["problematic_areas"]:
            print(f"   Problematic areas: {', '.join(dist['problematic_areas'])}")

        # Uniqueness
        unique = analysis["uniqueness_score"]
        print(f"   Feature uniqueness: {unique['uniqueness_score']:.2f}")
        print(f"   Similar features ratio: {unique['similar_features_ratio']:.1%}")
        print(f"   Risk level: {unique['risk_level']}")

        # Edge stability
        edge = analysis["edge_stability"]
        print(f"   Edge features ratio: {edge['edge_features_ratio']:.1%}")
        print(f"   Stability score: {edge['stability_score']:.2f}")
        print(f"   Risk assessment: {edge['risk_assessment']}")

        # Visual complexity
        complexity = analysis["visual_complexity"]
        print(f"   Visual complexity: {complexity['complexity_score']:.2f}")
        print(f"   Contrast: {complexity['contrast']:.1f}")
        print(f"   Edge density: {complexity['edge_density']:.3f}")

        # Symmetry
        symmetry = analysis["symmetry_issues"]
        print(f"   Symmetry risk: {symmetry['symmetry_risk']}")
        print(f"   Horizontal similarity: {symmetry['horizontal_similarity']:.2f}")
        print(f"   Vertical similarity: {symmetry['vertical_similarity']:.2f}")

        # Improvement recommendations
        recommendations = result["recommendations"]
        if recommendations:
            print("\n💡 Improvement Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"Critical": "🚨", "High": "❗", "Medium": "⚠️"}.get(rec["priority"], "ℹ️")
                print(f"   {i}. [{rec['category']}] {priority_emoji} {rec['priority']}")
                print(f"      Issue: {rec['issue']}")
                print(f"      Suggestion: {rec['suggestion']}")

        # Critical issues
        critical = result["critical_issues"]
        if critical:
            print("\n🚨 Critical Issues:")
            for issue in critical:
                print(f"   • {issue}")

        print("\n✅ Analysis completed")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


def cmd_visualize_feature(args):
    """Command: Generate feature visualization"""
    setup_logging(args.log_level)

    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        sys.exit(1)

    try:
        print(f"🎨 Generating feature visualization for: {args.image}")

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Generate default output name
            input_path = Path(args.image)
            output_path = input_path.parent / f"{input_path.stem}_features{input_path.suffix}"

        result = generate_feature_map(args.image, str(output_path))

        print(f"✅ Feature visualization saved to: {output_path}")
        print(f"   Features detected: {result.get('feature_count', 'Unknown')}")

    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        print(f"❌ Visualization failed: {e}")
        sys.exit(1)


def cmd_compare_images(args):
    """Command: Compare AR tracking confusion risk between two images"""
    setup_logging(args.log_level)

    if not os.path.exists(args.image1):
        print(f"❌ First image file not found: {args.image1}")
        sys.exit(1)

    if not os.path.exists(args.image2):
        print(f"❌ Second image file not found: {args.image2}")
        sys.exit(1)

    try:
        print("🔍 Comparing images:")
        print(f"   Image 1: {args.image1}")
        print(f"   Image 2: {args.image2}")

        result = compare_ar_tracking_potential(args.image1, args.image2)

        if "error" in result:
            print(f"❌ Comparison failed: {result['error']}")
            return

        # Display individual tracking quality
        scores = result["individual_scores"]
        print("\n📊 Individual Tracking Quality:")
        print(f"   Image 1: {scores['image1_tracking_score']:.2f}/1.00")
        print(f"   Image 2: {scores['image2_tracking_score']:.2f}/1.00")

        # Display feature similarity
        similarity = result["similarity_analysis"]
        print("\n🔬 Feature Similarity Analysis:")
        print(f"   FREAK matches: {similarity.get('total_matches', 0)}")
        print(f"   Similarity: {similarity.get('similarity_percentage', 0):.1f}%")
        print(f"   Average Hamming distance: {similarity.get('average_hamming_distance', 0):.3f}")

        # Display design similarity risk
        design_risk = result["design_similarity_risk"]
        print("\n🎨 Design Similarity Risk:")
        print(f"   Risk level: {design_risk['risk_level']}")
        print(f"   Overall risk score: {design_risk['overall_risk_score']:.2f}")
        print(f"   Complexity difference: {design_risk['complexity_difference']:.2f}")
        print(f"   Symmetry similarity: {design_risk['symmetry_similarity']:.2f}")
        print(f"   Distribution similarity: {design_risk['distribution_similarity']:.2f}")

        # Display confusion risk prediction
        confusion = result["confusion_prediction"]
        risk_level = confusion["risk_level"]
        risk_prob = confusion["confusion_probability"]

        print("\n⚠️  Confusion Risk Prediction:")
        risk_emoji = {"critical": "🚨", "high": "❗", "medium": "⚠️", "low": "✅"}.get(risk_level, "❓")
        print(f"   {risk_emoji} Risk level: {risk_level.upper()}")
        print(f"   Confusion probability: {risk_prob:.1%}")
        print(f"   Explanation: {confusion['explanation']}")

        # Display contributing factors
        factors = confusion["contributing_factors"]
        print("\n📈 Risk Factor Analysis:")
        print(f"   Feature similarity impact: {factors['feature_similarity']:.1%}")
        print(f"   Quality factor impact: {factors['quality_factor']:.1%}")
        print(f"   Uniqueness factor impact: {factors['uniqueness_factor']:.1%}")
        print(f"   Environmental risk impact: {factors['environmental_risk']:.1%}")

        # Display differentiation recommendations
        recommendations = result["recommendations"]
        if recommendations:
            print("\n💡 Differentiation Design Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"Highest": "🚨", "High": "❗", "Medium": "⚠️"}.get(rec["priority"], "ℹ️")
                print(f"   {i}. [{rec['category']}] {priority_emoji} {rec['priority']}")
                print(f"      Action: {rec['action']}")
                print(f"      Details: {rec['details']}")

        # Summary
        if risk_level in ["critical", "high"]:
            print("\n🚨 Important Warning: These two images have high confusion risk!")
            print(
                "   Very likely to cause misidentification in actual AR usage. Strong recommendation to follow above suggestions for design modifications."
            )
        elif risk_level == "medium":
            print("\n⚠️  Notice: Medium level confusion risk exists")
            print("   May confuse under certain environmental conditions, consider improvement suggestions.")
        else:
            print("\n✅ Good: These two images have sufficient distinctiveness, low confusion risk.")

        print("\n✅ Comparison analysis completed")

    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        print(f"❌ Comparison failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Iris AR Target Detection CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py detect                          # Start real-time detection
  python cli.py simulate                        # Run simulation mode
  python cli.py analyze-image images/card-a.png # Analyze AR tracking quality
  python cli.py compare-images images/card-a.png images/card-c.png  # Compare similarity
  python cli.py visualize-feature images/card-a.png                 # Generate feature map
        """,
    )

    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path (default: config.yaml)")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Start real-time target detection")
    detect_parser.set_defaults(func=cmd_detect)

    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Run in simulation mode")
    simulate_parser.set_defaults(func=cmd_simulate)

    # Analyze image command
    analyze_parser = subparsers.add_parser(
        "analyze-image", help="Deep analysis of image AR tracking quality and potential issues"
    )
    analyze_parser.add_argument("image", help="Path to image file to analyze")
    analyze_parser.set_defaults(func=cmd_analyze_image)

    # Visualize features command
    visualize_parser = subparsers.add_parser("visualize-feature", help="Generate feature visualization")
    visualize_parser.add_argument("image", help="Path to image file to visualize")
    visualize_parser.add_argument("--output", "-o", help="Output file path (optional)")
    visualize_parser.set_defaults(func=cmd_visualize_feature)

    # Compare images command
    compare_parser = subparsers.add_parser(
        "compare-images", help="Compare AR tracking confusion risk between two images"
    )
    compare_parser.add_argument("image1", help="Path to first image file")
    compare_parser.add_argument("image2", help="Path to second image file")
    compare_parser.set_defaults(func=cmd_compare_images)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the selected command
    args.func(args)


if __name__ == "__main__":
    main()
