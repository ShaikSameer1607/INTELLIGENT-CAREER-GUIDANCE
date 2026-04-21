"""
===============================================================================
OCR MODULE - Resume Text Extraction
Extracts and cleans text from resume images using Tesseract OCR
===============================================================================
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os

# Set Tesseract path (Windows default installation location)
# This fixes the issue where Tesseract is installed but not in PATH
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def preprocess_image(image_path):
    """
    Preprocess resume image for better OCR accuracy.
    
    Why: Raw resume images often have noise, poor contrast, or skew.
    Preprocessing improves text extraction quality significantly.
    
    Args:
        image_path: Path to resume image
    
    Returns:
        Preprocessed image (numpy array)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (binarization)
    # Why: Makes text stand out from background
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    
    # Denoise using Gaussian blur
    # Why: Removes small noise that can confuse OCR
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray


def extract_text_from_image(image_path, use_preprocessing=True):
    """
    Extract text from resume image using Tesseract OCR.
    
    Why Tesseract: 
    - Free, open-source OCR engine
    - Supports multiple languages
    - Works well with printed text (like resumes)
    
    Args:
        image_path: Path to resume image
        use_preprocessing: Whether to preprocess image first
    
    Returns:
        Extracted text string
    """
    try:
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("\n⚠️  ERROR: Tesseract OCR is not installed!")
            print("\nInstallation steps:")
            print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install to default location")
            print("3. Add to PATH or set tesseract_cmd in code")
            return ""
        
        # Load and preprocess image
        if use_preprocessing:
            img = preprocess_image(image_path)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Custom OCR configuration
        # --oem 3: Use both LSTM and legacy OCR engines (best accuracy)
        # --psm 6: Assume a single uniform block of text
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text
        text = pytesseract.image_to_string(img, config=custom_config)
        
        print(f"✅ OCR Complete: Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"\n❌ OCR Error: {str(e)}")
        return ""


def clean_extracted_text(text):
    """
    Clean and normalize extracted text.
    
    Why: OCR often introduces errors like:
    - Extra whitespace
    - Broken words
    - Special characters
    - Line breaks in wrong places
    
    Args:
        text: Raw OCR text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\[\]\/]', '', text)
    
    # Fix common OCR errors
    text = text.replace('  ', ' ')
    
    # Remove standalone numbers that might be page numbers
    text = re.sub(r'\b\d{1,2}\b', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_text_with_fallback(image_path):
    """
    Extract text with manual input fallback.
    
    Why: If OCR fails or Tesseract is not installed,
    allow user to paste resume text manually.
    
    Args:
        image_path: Path to resume image
    
    Returns:
        Extracted or manually input text
    """
    print("\n" + "="*80)
    print("RESUME TEXT EXTRACTION")
    print("="*80)
    
    # Try OCR first
    print(f"\nAttempting OCR on: {image_path}")
    text = extract_text_from_image(image_path)
    
    if text and len(text) > 50:  # Minimum viable text length
        print(f"✓ Successfully extracted {len(text)} characters via OCR")
        cleaned_text = clean_extracted_text(text)
        return cleaned_text
    else:
        print("\n⚠️  OCR extraction failed or returned insufficient text")
        print("\nFalling back to manual text input...")
        
        # Manual input
        print("\n" + "="*80)
        print("MANUAL TEXT INPUT")
        print("="*80)
        print("Please paste your resume text below:")
        print("(Type 'END' on a new line when finished)\n")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        
        manual_text = '\n'.join(lines)
        
        if manual_text.strip():
            print(f"\n✓ Captured {len(manual_text)} characters")
            return clean_extracted_text(manual_text)
        else:
            print("\n❌ No text provided")
            return ""


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        text = extract_text_with_fallback(image_path)
        
        if text:
            print("\n" + "="*80)
            print("EXTRACTED TEXT PREVIEW (First 500 chars):")
            print("="*80)
            print(text[:500])
            print("...")
    else:
        print("Usage: python ocr_module.py <image_path>")
