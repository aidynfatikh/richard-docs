#!/usr/bin/env python3
"""
Test script for Document Classifier
Tests classification of camera photos vs digital documents.

Usage:
    python test_classifier.py path/to/image.jpg
    python test_classifier.py --test-all  # Test with sample images
"""

import sys
import os
import argparse
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.document_classifier import is_document_photo


def test_single_image(image_path: str):
    """Test classification on a single image."""
    print(f"\n{'='*80}")
    print(f"Testing: {image_path}")
    print(f"{'='*80}\n")

    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Classify
    is_photo, info = is_document_photo(image_bytes)

    # Print results
    print(f"📊 CLASSIFICATION RESULTS\n")
    print(f"  Classification: {info['classification'].upper()}")
    print(f"  Is Camera Photo: {is_photo}")
    print(f"  Confidence: {info['confidence']:.1%}")
    print(f"  Recommendation: {info['recommendation']}\n")

    print(f"📈 SCORES\n")
    print(f"  EXIF Score:   {info['scores']['exif']:.3f}")
    print(f"  Visual Score: {info['scores']['visual']:.3f}")
    print(f"  Final Score:  {info['scores']['final']:.3f}\n")

    print(f"🔍 EXIF INDICATORS\n")
    exif = info['indicators']['exif']
    if exif:
        for key, value in exif.items():
            print(f"  {key}: {value}")
    else:
        print("  No EXIF indicators found")

    print(f"\n👁️  VISUAL INDICATORS\n")
    visual = info['indicators']['visual']
    if visual:
        for key, value in visual.items():
            print(f"  {key}: {value}")
    else:
        print("  No visual indicators analyzed")

    print(f"\n{'='*80}")
    print(f"✓ Classification complete\n")

    return is_photo, info


def test_sample_images():
    """Test with sample images from document_scan folder."""
    sample_dir = Path(__file__).parent / 'document_scan' / 'sample_images'

    if not sample_dir.exists():
        print(f"⚠️  Sample directory not found: {sample_dir}")
        return

    sample_images = list(sample_dir.glob('*.jpg')) + list(sample_dir.glob('*.JPG')) + \
                   list(sample_dir.glob('*.png')) + list(sample_dir.glob('*.jpeg'))

    if not sample_images:
        print(f"⚠️  No sample images found in {sample_dir}")
        return

    print(f"\n{'='*80}")
    print(f"TESTING MULTIPLE IMAGES")
    print(f"{'='*80}\n")
    print(f"Found {len(sample_images)} sample images\n")

    results = []

    for img_path in sample_images:
        print(f"Testing: {img_path.name}...", end=' ')
        try:
            with open(img_path, 'rb') as f:
                image_bytes = f.read()

            is_photo, info = is_document_photo(image_bytes)
            classification = info['classification']
            confidence = info['confidence']

            icon = "📷" if is_photo else "📄"
            print(f"{icon} {classification} (confidence: {confidence:.1%})")

            results.append({
                'filename': img_path.name,
                'classification': classification,
                'confidence': confidence,
                'is_photo': is_photo
            })

        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    camera_photos = [r for r in results if r['is_photo']]
    digital_docs = [r for r in results if not r['is_photo']]

    print(f"Total images tested: {len(results)}")
    print(f"Camera photos: {len(camera_photos)}")
    print(f"Digital documents: {len(digital_docs)}\n")

    if camera_photos:
        print("📷 Camera Photos:")
        for r in camera_photos:
            print(f"  - {r['filename']} (confidence: {r['confidence']:.1%})")

    if digital_docs:
        print("\n📄 Digital Documents:")
        for r in digital_docs:
            print(f"  - {r['filename']} (confidence: {r['confidence']:.1%})")

    print(f"\n{'='*80}\n")


def test_with_curl_examples():
    """Print example curl commands for testing the API."""
    print(f"\n{'='*80}")
    print(f"API TESTING EXAMPLES")
    print(f"{'='*80}\n")

    print("1. Classify a document (classification only, no processing):\n")
    print("   curl -X POST http://localhost:8000/classify-document \\")
    print("     -F 'file=@path/to/document.jpg'\n")

    print("2. Process document with auto-classification:\n")
    print("   curl -X POST http://localhost:8000/process-document \\")
    print("     -F 'file=@path/to/document.jpg' \\")
    print("     -F 'confidence=0.25'\n")

    print("3. Force perspective correction (even if digital):\n")
    print("   curl -X POST http://localhost:8000/process-document \\")
    print("     -F 'file=@path/to/document.jpg' \\")
    print("     -F 'force_scan=true'\n")

    print("4. Skip perspective correction (even if camera photo):\n")
    print("   curl -X POST http://localhost:8000/process-document \\")
    print("     -F 'file=@path/to/document.jpg' \\")
    print("     -F 'skip_scan=true'\n")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Document Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python test_classifier.py photo.jpg

  # Test all sample images
  python test_classifier.py --test-all

  # Show API usage examples
  python test_classifier.py --api-examples
        """
    )

    parser.add_argument(
        'image',
        nargs='?',
        help='Path to image file to classify'
    )
    parser.add_argument(
        '--test-all',
        action='store_true',
        help='Test all sample images'
    )
    parser.add_argument(
        '--api-examples',
        action='store_true',
        help='Show API usage examples'
    )

    args = parser.parse_args()

    if args.api_examples:
        test_with_curl_examples()
    elif args.test_all:
        test_sample_images()
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: File not found: {args.image}")
            return 1
        test_single_image(args.image)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
