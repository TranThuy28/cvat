#!/usr/bin/env python3
"""
Standalone script to import fake logit map data into FrameLogit model.

This script:
1. Reads 8-bit JPG images from a folder
2. Converts them to 16-bit PNG format (float32 -> uint16)
3. Stores the binary data directly in the database

Usage:
    python utils/import_logits_to_db.py <job_id> <folder_path>

Example:
    python utils/import_logits_to_db.py 4 /path/to/logit/images/
"""

import os
import sys
import re
import argparse
import django

# Setup Django environment
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.development')

# Initialize Django
django.setup()

# Now we can import Django models
import cv2
import numpy as np
from cvat.apps.engine.models import Job, FrameLogit


def convert_jpg_to_16bit_png(jpg_path: str) -> tuple[bytes, int, int]:
    """
    Convert 8-bit JPG image to 16-bit PNG binary data.

    Process:
    1. Read JPG as grayscale
    2. Convert to float32 (0.0 - 1.0) by dividing by 255.0
    3. Scale to 16-bit: Multiply by 65535 and cast to np.uint16
    4. Encode as PNG using cv2.imencode

    Args:
        jpg_path: Path to the input JPG file

    Returns:
        tuple: (binary_data, width, height)
    """
    # Read JPG as grayscale (returns uint8, 0-255)
    img_8bit = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)

    if img_8bit is None:
        raise ValueError(f"Failed to read image: {jpg_path}")

    height, width = img_8bit.shape

    # Convert to float32 (0.0 - 1.0) by dividing by 255.0
    img_float32 = img_8bit.astype(np.float32) / 255.0

    # Scale to 16-bit: Multiply by 65535 and cast to np.uint16
    img_16bit = (img_float32 * 65535.0).astype(np.uint16)

    # Encode as PNG using cv2.imencode
    # cv2.imencode returns (success, buffer)
    success, buffer = cv2.imencode('.png', img_16bit)

    if not success:
        raise ValueError(f"Failed to encode PNG: {jpg_path}")

    # Convert buffer to bytes
    binary_data = buffer.tobytes()

    return binary_data, width, height


def parse_frame_number(filename: str) -> int | None:
    """
    Parse frame number from filename.

    Example: slice_0001.jpg -> 0 (frame 0)
    Example: slice_0002.jpg -> 1 (frame 1)

    Args:
        filename: Filename like "slice_0001.jpg"

    Returns:
        Frame number (0-indexed) or None if parsing fails
    """
    # Find the last group of digits in the filename
    match = re.search(r'(\d+)', filename)
    if not match:
        return None

    file_num = int(match.group(1))
    # Convert to 0-indexed frame number (slice_0001 -> frame 0)
    frame_number = file_num - 1

    if frame_number < 0:
        return None

    return frame_number


def import_logits(job_id: int, folder_path: str) -> None:
    """
    Import logit maps from folder into FrameLogit model.

    Args:
        job_id: Job ID to associate logits with
        folder_path: Path to folder containing JPG images
    """
    # Validate job exists
    try:
        job = Job.objects.get(pk=job_id)
        print(f"✓ Found Job ID: {job_id}")
    except Job.DoesNotExist:
        print(f"✗ Error: Job {job_id} does not exist!")
        sys.exit(1)

    # Validate folder exists
    if not os.path.isdir(folder_path):
        print(f"✗ Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    # Get all JPG files
    jpg_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]

    if not jpg_files:
        print(f"✗ Error: No JPG files found in {folder_path}")
        sys.exit(1)

    jpg_files.sort()
    print(f"✓ Found {len(jpg_files)} JPG files")

    # Process each file
    count = 0
    errors = 0
    frame_mapping = {}

    for filename in jpg_files:
        file_path = os.path.join(folder_path, filename)

        # Parse frame number
        frame_number = parse_frame_number(filename)
        if frame_number is None:
            print(f"⚠ Skipping {filename}: Could not parse frame number")
            continue

        try:
            # Convert JPG to 16-bit PNG binary
            print(f"  Processing {filename} -> Frame {frame_number}...", end=' ')
            binary_data, width, height = convert_jpg_to_16bit_png(file_path)

            # Save to database using update_or_create
            frame_logit, created = FrameLogit.objects.update_or_create(
                job=job,
                frame=frame_number,
                defaults={
                    'data': binary_data,
                    'dtype': 'float32',
                    'compression': 'png_16bit',
                    'width': width,
                    'height': height,
                }
            )

            status = "Created" if created else "Updated"
            frame_mapping[frame_number] = filename
            count += 1
            print(f"✓ {status} ({width}x{height}, {len(binary_data)} bytes)")

        except Exception as e:
            errors += 1
            print(f"✗ Error: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print(f"Import completed!")
    print(f"  Successfully imported: {count} logit maps")
    if errors > 0:
        print(f"  Errors: {errors}")

    if frame_mapping:
        frames_list = sorted(frame_mapping.keys())
        print(f"  Frame range: {frames_list[0]} to {frames_list[-1]} (total: {len(frames_list)})")
        if 0 in frame_mapping:
            print(f"  ✓ Frame 0: {frame_mapping[0]}")
        else:
            print(f"  ⚠ Frame 0 NOT found in the list!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Import fake logit map data into FrameLogit model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/import_logits_to_db.py 4 /path/to/logit/images/
  python utils/import_logits_to_db.py 10 ./logits/
        """
    )

    parser.add_argument(
        'job_id',
        type=int,
        help='Job ID to associate logits with'
    )

    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing JPG images (e.g., slice_0001.jpg)'
    )

    args = parser.parse_args()

    # Convert to absolute path
    folder_path = os.path.abspath(args.folder_path)

    print("="*60)
    print("CVAT Logit Map Import Tool")
    print("="*60)
    print(f"Job ID: {args.job_id}")
    print(f"Folder: {folder_path}")
    print("="*60)
    print()

    try:
        import_logits(args.job_id, folder_path)
    except KeyboardInterrupt:
        print("\n\n✗ Import interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()






