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
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


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
    
    def pdf_to_images(self, pdf_bytes: bytes, enable_base64: bool = True, parallel: bool = True) -> List[Tuple[np.ndarray, int, Optional[str]]]:
        """
        Convert PDF pages to images with parallel processing.
        
        Args:
            pdf_bytes: Raw PDF bytes
            enable_base64: Whether to generate base64 encoded images (default: True)
            parallel: Whether to use parallel processing (default: True)
            
        Returns:
            List of (image_array, page_number, base64_image) tuples
            base64_image is None if enable_base64=False
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = pdf_document.page_count
            
            if parallel and page_count > 1:
                # Parallel processing for multi-page PDFs
                images = self._process_pdf_parallel(pdf_document, enable_base64)
            else:
                # Sequential processing for single page or when parallel disabled
                images = self._process_pdf_sequential(pdf_document, enable_base64)
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        return images
    
    def _process_single_page(self, page, page_num: int, enable_base64: bool) -> Tuple[np.ndarray, int, Optional[str]]:
        """
        Process a single PDF page to image.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            enable_base64: Whether to generate base64 encoded image
            
        Returns:
            Tuple of (image_array, page_number, base64_image)
        """
        # Render page to image
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert directly from pixmap to numpy (skip PIL for efficiency)
        img_data = pix.samples
        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGB to BGR for OpenCV
        if pix.n == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # Create base64 encoded image only if requested
        base64_str = None
        if enable_base64:
            # Encode to PNG for base64
            success, encoded_img = cv2.imencode('.png', img_array)
            if success:
                img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                base64_str = f"data:image/png;base64,{img_base64}"
        
        return (img_array, page_num + 1, base64_str)  # 1-indexed page numbers
    
    def _process_pdf_sequential(self, pdf_document, enable_base64: bool) -> List[Tuple[np.ndarray, int, Optional[str]]]:
        """
        Process PDF pages sequentially.
        """
        images = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            result = self._process_single_page(page, page_num, enable_base64)
            images.append(result)
        return images
    
    def _process_pdf_parallel(self, pdf_document, enable_base64: bool) -> List[Tuple[np.ndarray, int, Optional[str]]]:
        """
        Process PDF pages in parallel using ThreadPoolExecutor.
        """
        page_count = pdf_document.page_count
        max_workers = min(multiprocessing.cpu_count(), page_count, 8)  # Limit to 8 workers
        
        images = [None] * page_count  # Pre-allocate list to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self._process_single_page, pdf_document[i], i, enable_base64): i
                for i in range(page_count)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    result = future.result()
                    images[page_idx] = result
                except Exception as e:
                    print(f"Error processing page {page_idx + 1}: {e}")
                    raise
        
        return images
    
    def process_file(self, file_bytes: bytes, filename: str) -> List[Tuple[np.ndarray, Optional[int], Optional[str]]]:
        """
        Process any supported file format.
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            
        Returns:
            List of (image_array, page_number, base64_image) tuples.
            For images, page_number is None and base64_image is None. 
            For PDFs, page_number is 1-indexed and base64_image contains the rendered page.
            
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
        
        return [(img, None, None)]  # Single image, no page number, no base64
    
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
