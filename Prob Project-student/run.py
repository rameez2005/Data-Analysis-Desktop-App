#!/usr/bin/env python
"""
Student Habits and Performance Analysis Application
Run this script to start the application.
"""

import os
import sys
from PIL import Image, ImageDraw

# Check if down_arrow.png exists, if not create it
if not os.path.exists('down_arrow.png'):
    print("Creating down arrow icon...")
    # Create a new image with a transparent background
    img = Image.new('RGBA', (12, 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Draw a white down arrow
    draw.polygon([(2, 4), (10, 4), (6, 10)], fill=(255, 255, 255, 255))
    # Save the image
    img.save('down_arrow.png')

# Check if the dataset exists
if not os.path.exists('student_habits_performance.csv'):
    print("Error: 'student_habits_performance.csv' not found.")
    print("Please make sure the dataset file is in the same directory as this script.")
    sys.exit(1)

# Import and run the application
from modern_project import main

if __name__ == '__main__':
    main() 