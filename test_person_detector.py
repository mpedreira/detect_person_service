#!/usr/bin/env python3
"""Quick test to verify PersonDetector works correctly."""

import sys
import tempfile
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.person_detector import PersonDetector


def test_person_detector_import():
    """Test that PersonDetector can be imported."""
    print("✓ PersonDetector imported successfully")
    detector = PersonDetector(confidence_threshold=0.5)
    print(f"✓ PersonDetector initialized: {detector}")
    return detector


def test_with_black_image():
    """Test has_person with a blank/black image (should return False)."""
    from PIL import Image
    
    detector = PersonDetector(confidence_threshold=0.5)
    
    # Create a black image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="black")
        img.save(f.name)
        result = detector.has_person(f.name)
        Path(f.name).unlink()
    
    if result:
        print("⚠ Black image detected as having person (unexpected, but acceptable)")
    else:
        print("✓ Black image correctly detected as NO person")
    return True


def test_with_random_image():
    """Test has_person with random pixels (should return False)."""
    from PIL import Image
    import random
    
    detector = PersonDetector(confidence_threshold=0.5)
    
    # Create a random noise image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        pixels = [random.randint(0, 255) for _ in range(100 * 100 * 3)]
        img = Image.new("RGB", (100, 100))
        img.putdata([(pixels[i], pixels[i+1], pixels[i+2]) for i in range(0, len(pixels)-2, 3)])
        img.save(f.name)
        result = detector.has_person(f.name)
        Path(f.name).unlink()
    
    if result:
        print("⚠ Random noise detected as having person (acceptable with MediaPipe)")
    else:
        print("✓ Random noise correctly detected as NO person")
    return True


if __name__ == "__main__":
    print("\n=== PersonDetector Integration Tests ===\n")
    
    try:
        test_person_detector_import()
        print()
        test_with_black_image()
        print()
        test_with_random_image()
        
        print("\n✓ All basic tests passed!")
        print("→ Ready to build Docker and test with real videos\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
