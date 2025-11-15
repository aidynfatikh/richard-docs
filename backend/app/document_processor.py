"""
Document Processor
Handles conversion of various document formats (PDF, images) to processable format.
Converts PDFs to images per page and normalizes image formats.
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image


class DocumentProcessor:
    """Processes documents (PDFs and images) for detection."""
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.bmp'}
    SUPPORTED_PDF_FORMAT = '.pdf'
    
    def __init__(self, dpi: int = 200):
        """
        Initialize document processor.
        
        Args:
            dpi: DPI for PDF rendering (default: 200 for good quality/speed balance)
        """
        self.dpi = dpi
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get lowercase file extension."""
        return Path(filename).suffix.lower()
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported."""
        ext = self.get_file_extension(filename)
        return ext in self.SUPPORTED_IMAGE_FORMATS or ext == self.SUPPORTED_PDF_FORMAT
    
    def is_pdf(self, filename: str) -> bool:
        """Check if file is a PDF."""
        return self.get_file_extension(filename) == self.SUPPORTED_PDF_FORMAT
    
    def process_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Process image bytes to OpenCV format.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array in BGR format, or None if failed
        """
        try:
            # Try with PIL first (handles more formats)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode not in ('RGB', 'L'):
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy
            img_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            return img_array
            
        except Exception as e:
            print(f"Error processing image with PIL: {e}")
            
            # Fallback to OpenCV
            try:
                arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            except Exception as e2:
                print(f"Error processing image with OpenCV: {e2}")
                return None
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Tuple[np.ndarray, int]]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_bytes: Raw PDF bytes
            
        Returns:
            List of (image_array, page_number) tuples
        """
        images = []
        
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Process each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Render page to image
                # zoom factor: dpi/72 (72 is default PDF DPI)
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to OpenCV format (numpy array, BGR)
                img_array = np.array(pil_image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                images.append((img_array, page_num + 1))  # 1-indexed page numbers
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        return images
    
    def process_file(self, file_bytes: bytes, filename: str) -> List[Tuple[np.ndarray, Optional[int]]]:
        """
        Process any supported file format.
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            
        Returns:
            List of (image_array, page_number) tuples.
            For images, page_number is None. For PDFs, it's 1-indexed.
            
        Raises:
            ValueError: If file format is not supported or processing fails
        """
        if not self.is_supported_format(filename):
            ext = self.get_file_extension(filename)
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_IMAGE_FORMATS | {self.SUPPORTED_PDF_FORMAT})}"
            )
        
        # Process PDF
        if self.is_pdf(filename):
            return self.pdf_to_images(file_bytes)
        
        # Process image
        img = self.process_image_bytes(file_bytes)
        if img is None:
            raise ValueError("Failed to process image file")
        
        return [(img, None)]  # Single image, no page number
    
    def get_format_info(self, filename: str) -> dict:
        """Get information about file format."""
        ext = self.get_file_extension(filename)
        is_pdf = ext == self.SUPPORTED_PDF_FORMAT
        is_image = ext in self.SUPPORTED_IMAGE_FORMATS
        
        return {
            'extension': ext,
            'is_pdf': is_pdf,
            'is_image': is_image,
            'is_supported': is_pdf or is_image
        }


# Global processor instance
_processor_instance = None


def get_processor(dpi: int = 200) -> DocumentProcessor:
    """
    Get or create document processor instance (singleton).
    
    Args:
        dpi: DPI for PDF rendering
        
    Returns:
        DocumentProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = DocumentProcessor(dpi=dpi)
    
    return _processor_instance
