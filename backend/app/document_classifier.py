"""
Document Classifier - Distinguishes camera photos from digital documents
Uses EXIF metadata analysis + visual heuristics for robust classification.

Author: Digital Inspector Team
"""

import io
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Classifies uploaded files as either:
    - 'camera_photo': Real photo taken with phone camera (needs perspective correction)
    - 'digital_document': Digital file (PDF export, scan, screenshot) - use as-is

    Uses multi-layered approach:
    1. EXIF metadata analysis (primary, most reliable)
    2. Visual contour detection (fallback)
    3. Combined confidence scoring
    """

    # Known camera manufacturers that indicate phone photos
    CAMERA_MAKES = [
        'Apple', 'Samsung', 'Google', 'Huawei', 'Xiaomi', 'OnePlus',
        'OPPO', 'vivo', 'Motorola', 'Nokia', 'Sony', 'LG', 'HTC',
        'HONOR', 'realme', 'Lenovo', 'ASUS', 'ZTE'
    ]

    # Phone model keywords
    PHONE_MODELS = [
        'iPhone', 'iPad', 'Galaxy', 'Pixel', 'SM-', 'Mi ', 'Redmi',
        'Poco', 'Note', 'Pro', 'Ultra', 'Mate', 'P30', 'P40', 'P50',
        'OnePlus', 'Nord', 'Find', 'Reno', 'A50', 'A70', 'M3', 'M4'
    ]

    # Software that indicates digital document (not camera photo)
    DIGITAL_SOFTWARE = [
        'photoshop', 'illustrator', 'word', 'excel', 'powerpoint',
        'scanner', 'gimp', 'inkscape', 'acrobat', 'preview',
        'paint', 'snagit', 'camscanner', 'office', 'libreoffice'
    ]

    def __init__(
        self,
        exif_weight: float = 0.7,
        visual_weight: float = 0.3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize classifier.

        Args:
            exif_weight: Weight for EXIF evidence (0-1), default 0.7
            visual_weight: Weight for visual evidence (0-1), default 0.3
            confidence_threshold: Threshold for classification (0-1), default 0.5
        """
        self.exif_weight = exif_weight
        self.visual_weight = visual_weight
        self.confidence_threshold = confidence_threshold

    def is_document_photo(self, image_bytes: bytes, fast_mode: bool = False) -> Tuple[bool, Dict]:
        """
        Main classification function.

        Args:
            image_bytes: Raw image bytes
            fast_mode: If True, skip heavy visual analysis (10x faster)
                      Useful for batch processing where EXIF is sufficient

        Returns:
            Tuple of (is_camera_photo, metadata_dict)
            - is_camera_photo: True if camera photo, False if digital document
            - metadata_dict: Classification details and confidence

        Example:
            is_photo, info = classifier.is_document_photo(image_bytes)
            if is_photo:
                # Apply perspective correction
                pass
            else:
                # Use image as-is
                pass
        """
        try:
            # Step 1: Load image safely
            image = Image.open(io.BytesIO(image_bytes))

            # Step 2: Extract and analyze EXIF
            exif_score, exif_indicators = self._analyze_exif(image)
            logger.info(f"EXIF analysis: score={exif_score:.2f}, indicators={exif_indicators}")

            # Step 3: Visual analysis (skip in fast_mode or if EXIF is highly confident)
            visual_score = 0.5  # Neutral default
            visual_indicators = {}
            
            if fast_mode:
                # Fast mode: Use EXIF only (10x faster)
                final_score = exif_score
                visual_indicators['skipped'] = 'fast_mode_enabled'
            elif exif_score > 0.8 or exif_score < 0.2:
                # Strong EXIF evidence, skip expensive visual analysis
                final_score = exif_score
                visual_indicators['skipped'] = 'exif_confident'
                logger.info(f"Skipping visual analysis (EXIF highly confident: {exif_score:.2f})")
            else:
                # Ambiguous EXIF, run full visual analysis
                visual_score, visual_indicators = self._analyze_visual_features(image_bytes)
                logger.info(f"Visual analysis: score={visual_score:.2f}, indicators={visual_indicators}")
                # Combine scores with balanced weights
                final_score = exif_score * self.exif_weight + visual_score * self.visual_weight

            # Step 5: Make classification decision
            is_camera_photo = final_score > self.confidence_threshold

            # Step 6: Build metadata response
            metadata = {
                'classification': 'camera_photo' if is_camera_photo else 'digital_document',
                'confidence': round(final_score, 3),
                'scores': {
                    'exif': round(exif_score, 3),
                    'visual': round(visual_score, 3),
                    'final': round(final_score, 3)
                },
                'indicators': {
                    'exif': exif_indicators,
                    'visual': visual_indicators
                },
                'recommendation': 'apply_perspective_correction' if is_camera_photo else 'use_as_is',
                'threshold': self.confidence_threshold
            }

            logger.info(f"Classification: {metadata['classification']} (confidence: {final_score:.2f})")
            return is_camera_photo, metadata

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            # Fail safe: assume digital document (don't apply correction if unsure)
            return False, {
                'classification': 'digital_document',
                'confidence': 0.0,
                'error': str(e),
                'recommendation': 'use_as_is'
            }

    def _analyze_exif(self, image: Image.Image) -> Tuple[float, Dict]:
        """
        Analyze EXIF metadata to determine if image is from camera.

        Returns:
            Tuple of (score, indicators)
            - score: 0.0 (definitely digital) to 1.0 (definitely camera photo)
            - indicators: Dict of detected EXIF properties
        """
        score = 0.5  # Start neutral
        indicators = {}

        try:
            exif_data = image.getexif()

            if not exif_data:
                indicators['has_exif'] = False
                return 0.3, indicators  # Slightly favor digital if no EXIF

            indicators['has_exif'] = True

            # Parse EXIF tags
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value

            # 1. Camera Make (strong indicator)
            make = exif_dict.get('Make', '')
            if make:
                indicators['make'] = str(make)
                if any(brand in str(make) for brand in self.CAMERA_MAKES):
                    score += 0.25
                    indicators['is_phone_camera'] = True
                    logger.debug(f"Camera make detected: {make}")

            # 2. Camera Model (strong indicator)
            model = exif_dict.get('Model', '')
            if model:
                indicators['model'] = str(model)
                if any(phone_keyword in str(model) for phone_keyword in self.PHONE_MODELS):
                    score += 0.20
                    indicators['is_phone_model'] = True
                    logger.debug(f"Phone model detected: {model}")

            # 3. Software (negative indicator if document software)
            software = exif_dict.get('Software', '')
            if software:
                indicators['software'] = str(software)
                software_lower = str(software).lower()
                if any(app in software_lower for app in self.DIGITAL_SOFTWARE):
                    score -= 0.35
                    indicators['is_digital_software'] = True
                    logger.debug(f"Digital software detected: {software}")

            # 4. Camera-specific settings (moderate indicators)
            iso = exif_dict.get('ISOSpeedRatings') or exif_dict.get('PhotographicSensitivity')
            if iso:
                score += 0.08
                indicators['has_iso'] = True
                indicators['iso'] = iso

            aperture = exif_dict.get('FNumber') or exif_dict.get('ApertureValue')
            if aperture:
                score += 0.08
                indicators['has_aperture'] = True

            focal_length = exif_dict.get('FocalLength')
            if focal_length:
                score += 0.08
                indicators['has_focal_length'] = True

            exposure_time = exif_dict.get('ExposureTime')
            if exposure_time:
                score += 0.05
                indicators['has_exposure_time'] = True

            # 5. GPS (indicates phone camera)
            gps_info = exif_dict.get('GPSInfo')
            if gps_info:
                score += 0.10
                indicators['has_gps'] = True
                logger.debug("GPS data found (phone camera indicator)")

            # 6. Orientation (common in phone photos)
            orientation = exif_dict.get('Orientation')
            if orientation and orientation != 1:
                score += 0.05
                indicators['has_orientation'] = True

            # 7. DateTime vs DateTimeOriginal (scans often lack Original)
            datetime_original = exif_dict.get('DateTimeOriginal')
            datetime = exif_dict.get('DateTime')
            if datetime_original and datetime:
                # Photos have both, scans often just DateTime
                score += 0.05
                indicators['has_datetime_original'] = True

            # 8. Image dimensions check
            # Scanned documents often have standard DPI-based dimensions
            # Phone photos have camera sensor dimensions
            width = exif_dict.get('ExifImageWidth') or image.width
            height = exif_dict.get('ExifImageHeight') or image.height
            if width and height:
                aspect_ratio = width / height if height > 0 else 1
                # Common phone aspect ratios: 4:3 (~1.33), 16:9 (~1.78), 3:2 (~1.5)
                phone_ratios = [4/3, 16/9, 3/2, 9/16, 2/3, 3/4]
                is_phone_ratio = any(abs(aspect_ratio - ratio) < 0.05 for ratio in phone_ratios)
                if is_phone_ratio:
                    score += 0.03
                    indicators['phone_aspect_ratio'] = True

        except Exception as e:
            logger.warning(f"EXIF analysis error: {e}")
            indicators['exif_error'] = str(e)

        # Clamp score to [0, 1]
        final_score = max(0.0, min(1.0, score))
        return final_score, indicators

    def _analyze_visual_features(self, image_bytes: bytes) -> Tuple[float, Dict]:
        """
        Analyze visual features to detect camera photos vs digital documents.

        Uses:
        - Contour detection for document edges
        - Edge proximity analysis
        - Perspective distortion detection
        - Background detection

        Returns:
            Tuple of (score, indicators)
        """
        score = 0.5  # Start neutral
        indicators = {}

        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Failed to decode image for visual analysis")
                return 0.5, {'decode_error': True}

            height, width = img.shape[:2]
            image_area = height * width

            # 1. Document contour detection
            contour_info = self._detect_document_contour(img)
            if contour_info['found']:
                indicators['document_contour'] = True
                indicators['contour_area_ratio'] = round(contour_info['area_ratio'], 3)

                # If document detected and doesn't fill frame, likely camera photo
                if contour_info['area_ratio'] < 0.92:
                    score += 0.25
                    indicators['document_within_frame'] = True
                    logger.debug(f"Document contour detected, area ratio: {contour_info['area_ratio']:.2f}")
                elif contour_info['area_ratio'] > 0.98:
                    # Nearly fills frame - likely digital document
                    score -= 0.20
                    indicators['document_fills_frame'] = True

            # 2. Edge proximity check
            edge_touch = self._check_edges_touch_boundaries(img)
            if edge_touch:
                score -= 0.15
                indicators['edges_touch_boundaries'] = True
                logger.debug("Document edges touch image boundaries (digital indicator)")
            else:
                score += 0.10
                indicators['edges_have_margin'] = True

            # 3. Perspective distortion detection
            if contour_info['found'] and contour_info['corners'] is not None:
                has_perspective = self._detect_perspective_distortion(contour_info['corners'], width, height)
                if has_perspective:
                    score += 0.20
                    indicators['perspective_distortion'] = True
                    logger.debug("Perspective distortion detected (camera photo indicator)")

            # 4. Background detection (non-white pixels around edges)
            has_background = self._detect_visible_background(img)
            if has_background:
                score += 0.15
                indicators['visible_background'] = True
                logger.debug("Visible background detected (camera photo indicator)")

            # 5. Blur/focus variation (cameras have varying focus, scans are uniform)
            blur_variance = self._measure_blur_variance(img)
            if blur_variance > 0.15:  # Threshold determined empirically
                score += 0.08
                indicators['blur_variance'] = round(blur_variance, 3)
                indicators['has_focus_variation'] = True

            # 6. Lighting uniformity
            lighting_score = self._analyze_lighting_uniformity(img)
            if lighting_score < 0.7:  # Non-uniform lighting
                score += 0.07
                indicators['non_uniform_lighting'] = True

        except Exception as e:
            logger.warning(f"Visual analysis error: {e}")
            indicators['visual_error'] = str(e)

        # Clamp score
        final_score = max(0.0, min(1.0, score))
        return final_score, indicators

    def _detect_document_contour(self, img: np.ndarray) -> Dict:
        """
        Detect quadrilateral document contour in image.

        Returns dict with:
        - found: bool
        - area_ratio: float (contour area / image area)
        - corners: np.ndarray or None
        """
        result = {
            'found': False,
            'area_ratio': 0.0,
            'corners': None
        }

        try:
            height, width = img.shape[:2]
            image_area = height * width

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise while keeping edges sharp
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Dilate edges to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return result

            # Sort by area and take largest
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for contour in contours:
                # Approximate contour to polygon
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                # Look for quadrilateral (4 corners)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    area_ratio = area / image_area

                    # Valid if area is reasonable (20% to 99% of image)
                    if 0.20 < area_ratio < 0.99:
                        result['found'] = True
                        result['area_ratio'] = area_ratio
                        result['corners'] = approx.reshape(4, 2)
                        break

        except Exception as e:
            logger.warning(f"Contour detection error: {e}")

        return result

    def _check_edges_touch_boundaries(self, img: np.ndarray, threshold: int = 10) -> bool:
        """
        Check if document edges touch image boundaries.
        Digital documents typically fill the frame edge-to-edge.

        Args:
            img: Input image
            threshold: Pixel distance from edge to consider "touching"

        Returns:
            True if edges touch boundaries
        """
        try:
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 100, 200)

            # Check border regions
            top = edges[:threshold, :]
            bottom = edges[-threshold:, :]
            left = edges[:, :threshold]
            right = edges[:, -threshold:]

            # Count edge pixels in borders
            border_edges = (
                np.sum(top > 0) + np.sum(bottom > 0) +
                np.sum(left > 0) + np.sum(right > 0)
            )

            # Total edge pixels
            total_edges = np.sum(edges > 0)

            if total_edges == 0:
                return False

            # If more than 30% of edges are on borders, likely touches boundaries
            ratio = border_edges / total_edges
            return ratio > 0.30

        except Exception as e:
            logger.warning(f"Edge proximity check error: {e}")
            return False

    def _detect_perspective_distortion(
        self,
        corners: np.ndarray,
        img_width: int,
        img_height: int
    ) -> bool:
        """
        Detect if corners show perspective distortion (trapezoid shape).
        Camera photos of documents typically have perspective distortion.

        Args:
            corners: 4 corner points as numpy array (4x2)
            img_width: Image width
            img_height: Image height

        Returns:
            True if perspective distortion detected
        """
        try:
            if corners is None or len(corners) != 4:
                return False

            # Calculate side lengths
            def distance(p1, p2):
                return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

            # Order corners: top-left, top-right, bottom-right, bottom-left
            # Simple ordering by y-coordinate then x-coordinate
            corners_sorted = sorted(corners, key=lambda p: (p[1], p[0]))

            # Top two points
            top_points = sorted(corners_sorted[:2], key=lambda p: p[0])
            # Bottom two points
            bottom_points = sorted(corners_sorted[2:], key=lambda p: p[0])

            tl, tr = top_points
            bl, br = bottom_points

            # Calculate side lengths
            top_width = distance(tl, tr)
            bottom_width = distance(bl, br)
            left_height = distance(tl, bl)
            right_height = distance(tr, br)

            # Check for trapezoid effect (opposite sides differ significantly)
            width_diff_ratio = abs(top_width - bottom_width) / max(top_width, bottom_width)
            height_diff_ratio = abs(left_height - right_height) / max(left_height, right_height)

            # If sides differ by more than 8%, likely perspective distortion
            has_distortion = width_diff_ratio > 0.08 or height_diff_ratio > 0.08

            return has_distortion

        except Exception as e:
            logger.warning(f"Perspective detection error: {e}")
            return False

    def _detect_visible_background(self, img: np.ndarray) -> bool:
        """
        Detect if there's visible background around document.
        Camera photos typically show background, digital docs don't.

        Returns:
            True if background detected
        """
        try:
            height, width = img.shape[:2]

            # Sample border regions (outer 8% of image)
            border_width = int(width * 0.08)
            border_height = int(height * 0.08)

            # Extract border pixels
            top = img[:border_height, :]
            bottom = img[-border_height:, :]
            left = img[:, :border_width]
            right = img[:, -border_width:]

            borders = np.vstack([
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3)
            ])

            # Calculate color variance in border
            # White/light backgrounds have low variance
            # Colorful/textured backgrounds have high variance
            variance = np.var(borders)

            # Also check if borders are significantly different from center
            center = img[
                border_height:-border_height,
                border_width:-border_width
            ].reshape(-1, 3)

            border_mean = np.mean(borders, axis=0)
            center_mean = np.mean(center, axis=0)
            mean_diff = np.linalg.norm(border_mean - center_mean)

            # Background detected if:
            # - High variance in borders (textured background), OR
            # - Significant difference between border and center
            has_background = variance > 800 or mean_diff > 25

            return has_background

        except Exception as e:
            logger.warning(f"Background detection error: {e}")
            return False

    def _measure_blur_variance(self, img: np.ndarray) -> float:
        """
        Measure spatial variance in blur/sharpness across image.
        Camera photos have varying focus, scans are uniformly sharp.

        Returns:
            Variance score (0-1), higher = more variation
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # Divide image into grid
            grid_size = 4
            h_step = height // grid_size
            w_step = width // grid_size

            sharpness_scores = []

            for i in range(grid_size):
                for j in range(grid_size):
                    # Extract region
                    y1 = i * h_step
                    y2 = (i + 1) * h_step
                    x1 = j * w_step
                    x2 = (j + 1) * w_step

                    region = gray[y1:y2, x1:x2]

                    # Calculate Laplacian variance (sharpness metric)
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    sharpness = laplacian.var()
                    sharpness_scores.append(sharpness)

            # Calculate variance of sharpness scores
            if len(sharpness_scores) > 0:
                # Normalize by mean to get coefficient of variation
                mean_sharp = np.mean(sharpness_scores)
                if mean_sharp > 0:
                    variance = np.std(sharpness_scores) / mean_sharp
                    return min(1.0, variance)

            return 0.0

        except Exception as e:
            logger.warning(f"Blur variance measurement error: {e}")
            return 0.0

    def _analyze_lighting_uniformity(self, img: np.ndarray) -> float:
        """
        Analyze lighting uniformity across image.
        Scans have uniform lighting, camera photos often don't.

        Returns:
            Uniformity score (0-1), 1 = perfectly uniform
        """
        try:
            # Convert to LAB color space (L channel = lightness)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            # Calculate standard deviation of lightness
            std_dev = np.std(l_channel)

            # Normalize (empirically, std_dev < 15 is very uniform)
            # std_dev > 40 is quite non-uniform
            uniformity = max(0.0, min(1.0, 1.0 - (std_dev - 15) / 25))

            return uniformity

        except Exception as e:
            logger.warning(f"Lighting analysis error: {e}")
            return 0.5


# Singleton instance for global use
_classifier_instance = None


def get_classifier() -> DocumentClassifier:
    """
    Get or create singleton classifier instance.

    Returns:
        DocumentClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DocumentClassifier()
    return _classifier_instance


def is_document_photo(image_bytes: bytes, fast_mode: bool = False) -> Tuple[bool, Dict]:
    """
    Convenient function to classify document.

    Args:
        image_bytes: Raw image bytes
        fast_mode: If True, skip heavy visual analysis (10x faster)
                  Use for batch processing where speed matters

    Returns:
        Tuple of (is_camera_photo, classification_metadata)

    Example:
        is_photo, info = is_document_photo(file_bytes)
        print(f"Classification: {info['classification']}")
        print(f"Confidence: {info['confidence']}")
        if is_photo:
            # Apply perspective correction
            corrected = apply_docscanner(file_bytes)
        else:
            # Use as-is
            process_document(file_bytes)
    """
    classifier = get_classifier()
    return classifier.is_document_photo(image_bytes, fast_mode=fast_mode)
