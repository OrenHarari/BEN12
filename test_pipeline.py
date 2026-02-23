#!/usr/bin/env python3
"""
Dry-run test for the pipeline with dummy images.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

# Add workspace to path
sys.path.insert(0, '/workspaces/BEN12')

from app.config import get_config
from pipeline.orchestrator import GrowingUpPipeline


def create_dummy_face_image(width=512, height=512, seed=42):
    """
    Create a realistic dummy face image for testing.
    This is a simple gradient-based face pattern.
    """
    rng = np.random.RandomState(seed)
    
    # Start with skin tone
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :] = [220, 180, 150]  # Skin tone
    
    # Add forehead
    img[:height//3, :] = [200, 160, 130]
    
    # Add eyes
    eye_y = height // 3
    eye_x_left = width // 3
    eye_x_right = 2 * width // 3
    eye_radius = width // 12
    
    for y in range(max(0, eye_y - eye_radius), min(height, eye_y + eye_radius)):
        for x in range(max(0, eye_x_left - eye_radius), min(width, eye_x_left + eye_radius)):
            if (x - eye_x_left)**2 + (y - eye_y)**2 < eye_radius**2:
                img[y, x] = [50, 50, 50]  # Eye
    
    for y in range(max(0, eye_y - eye_radius), min(height, eye_y + eye_radius)):
        for x in range(max(0, eye_x_right - eye_radius), min(width, eye_x_right + eye_radius)):
            if (x - eye_x_right)**2 + (y - eye_y)**2 < eye_radius**2:
                img[y, x] = [50, 50, 50]  # Eye
    
    # Add nose  
    nose_y = height // 2
    nose_x = width // 2
    nose_size = width // 20
    img[nose_y:nose_y + nose_size*2, nose_x - nose_size//2:nose_x + nose_size//2] = [180, 140, 120]
    
    # Add mouth
    mouth_y = 2 * height // 3
    mouth_x = width // 2
    mouth_width = width // 5
    img[mouth_y, mouth_x - mouth_width:mouth_x + mouth_width] = [150, 80, 80]
    
    # Add some noise for realism
    noise = rng.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img, mode='RGB')


def test_single_image():
    """Test with a single image (AI generation mode)."""
    print("\n" + "="*60)
    print("TEST 1: Single Image (AI Generation Mode)")
    print("="*60)
    
    try:
        print("Skipping single-image mode test (requires model weights)")
        print("(This mode generates age stages using SDXL)")
        return None  # Skip, not FAIL
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_image():
    """Test with multiple images."""
    print("\n" + "="*60)
    print("TEST 2: Multiple Images (Timeline Mode)")
    print("="*60)
    
    try:
        config = get_config()
        pipeline = GrowingUpPipeline(config)
        
        # Create dummy images
        print("Creating 3 dummy face images...")
        images = [
            create_dummy_face_image(seed=i)
            for i in range(1, 4)
        ]
        
        for i, img in enumerate(images):
            print(f"  Image {i+1}: {img.size}")
        
        print("Running multi-image pipeline...")
        
        def progress_cb(p, msg):
            print(f"  [{p*100:3.0f}%] {msg}")
        
        output_path = pipeline.run_multi_image(images, progress_callback=progress_cb)
        
        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"\n✅ SUCCESS: Video created at {output_path}")
            print(f"   File size: {size_mb:.2f} MB")
            return True
        else:
            print(f"\n❌ FAILED: Output file not found: {output_path}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🎬 BEN12 Pipeline Dry-Run Test")
    print("="*60)
    
    # Ensure output directory exists
    config = get_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Test 1: Single image
    results.append(("Single Image (AI)", test_single_image()))
    
    # Test 2: Multiple images
    results.append(("Multiple Images", test_multi_image()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    print("\n" + ("="*60))
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
