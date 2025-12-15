"""
Rank Recognition System

OCR-based English word recognition with closed vocabulary validation
for guard point/rank recognition in attendance forms.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import logging
from difflib import SequenceMatcher
import re
from paddleocr import PaddleOCR
from collections import Counter

logger = logging.getLogger(__name__)


class RankRecognizer:
    """Recognizes rank/guard point text using OCR and vocabulary validation."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize rank recognizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        
        # Initialize PaddleOCR for English
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info(f"PaddleOCR initialized for English with GPU={use_gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
        
        # Define closed vocabulary for ranks/guard points
        self.vocabulary = self._get_rank_vocabulary()
        self.vocabulary_lower = {word.lower() for word in self.vocabulary}
        
        # Common OCR errors and corrections
        self.ocr_corrections = self._get_ocr_corrections()
        
        # Minimum confidence threshold
        self.min_confidence = 0.3
        
        # Similarity threshold for fuzzy matching
        self.similarity_threshold = 0.7
    
    def _get_rank_vocabulary(self) -> Set[str]:
        """
        Get the closed vocabulary of valid ranks/guard points.
        
        Returns:
            Set of valid rank terms
        """
        # Common military/security ranks and positions in English
        vocabulary = {
            # Military ranks
            'PRIVATE', 'CORPORAL', 'SERGEANT', 'LIEUTENANT', 'CAPTAIN',
            'MAJOR', 'COLONEL', 'GENERAL',
            
            # Security positions
            'GUARD', 'SECURITY', 'OFFICER', 'SUPERVISOR', 'CHIEF',
            'INSPECTOR', 'COMMANDER', 'DIRECTOR',
            
            # Guard points/positions
            'GATE', 'ENTRANCE', 'EXIT', 'MAIN', 'SIDE', 'BACK',
            'FRONT', 'NORTH', 'SOUTH', 'EAST', 'WEST',
            'BUILDING', 'TOWER', 'POST', 'STATION',
            
            # Common abbreviations
            'PVT', 'CPL', 'SGT', 'LT', 'CAPT', 'MAJ', 'COL', 'GEN',
            'SEC', 'OFF', 'SUP', 'CMD', 'DIR',
            
            # Numbers and positions
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            'FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH',
            
            # Combined terms
            'GATE1', 'GATE2', 'GATE3', 'GATE4', 'GATE5',
            'POST1', 'POST2', 'POST3', 'POST4', 'POST5',
            'SECTOR1', 'SECTOR2', 'SECTOR3', 'SECTOR4',
            'ZONE1', 'ZONE2', 'ZONE3', 'ZONE4',
            
            # Arabic transliterations (common in mixed documents)
            'HARAS', 'AMEEN', 'RAQEEB', 'MUDIR', 'RAEES'
        }
        
        return vocabulary
    
    def _get_ocr_corrections(self) -> Dict[str, str]:
        """
        Get common OCR error corrections.
        
        Returns:
            Dictionary mapping common OCR errors to correct words
        """
        return {
            # Common character confusions
            'QUARD': 'GUARD',
            'CUARD': 'GUARD',
            'GVARD': 'GUARD',
            'GHARD': 'GUARD',
            'OFPICER': 'OFFICER',
            'OPFICER': 'OFFICER',
            'OFFICEH': 'OFFICER',
            'SECUHITY': 'SECURITY',
            'SECVRITY': 'SECURITY',
            'SECURLTY': 'SECURITY',
            'CATE': 'GATE',
            'GAFE': 'GATE',
            'GAIE': 'GATE',
            'POSI': 'POST',
            'POSF': 'POST',
            'P0ST': 'POST',
            'SERGEANF': 'SERGEANT',
            'SERGEANL': 'SERGEANT',
            'LIEUIENANT': 'LIEUTENANT',
            'LIEUTENANF': 'LIEUTENANT',
            'CAPFAIN': 'CAPTAIN',
            'CAPIAIN': 'CAPTAIN',
            
            # Number confusions
            'I': '1',
            'l': '1',
            'O': '0',
            'o': '0',
            'S': '5',
            'G': '6',
            'B': '8',
            
            # Common word variations
            'SEC': 'SECURITY',
            'OFF': 'OFFICER',
            'SUP': 'SUPERVISOR',
            'CMD': 'COMMANDER',
            'DIR': 'DIRECTOR'
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR recognition.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_with_ocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image
            
        Returns:
            List of (text, confidence) tuples
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run OCR
            result = self.ocr.ocr(processed_image, cls=True)
            
            extracted_texts = []
            if result and len(result) > 0:
                for item in result[0]:
                    if len(item) >= 2:
                        bbox, (text, confidence) = item
                        
                        # Clean and normalize text
                        cleaned_text = self._clean_text(text)
                        
                        if cleaned_text and confidence >= self.min_confidence:
                            extracted_texts.append((cleaned_text, float(confidence)))
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', ' ', text.strip().upper())
        
        # Remove special characters except alphanumeric and common punctuation
        cleaned = re.sub(r'[^A-Z0-9\s\-\.]', '', cleaned)
        
        # Apply OCR corrections
        for error, correction in self.ocr_corrections.items():
            cleaned = cleaned.replace(error, correction)
        
        return cleaned.strip()
    
    def validate_against_vocabulary(self, text: str) -> Tuple[Optional[str], float]:
        """
        Validate text against closed vocabulary using fuzzy matching.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (best_match, similarity_score)
        """
        if not text:
            return None, 0.0
        
        text_lower = text.lower()
        
        # Exact match
        if text_lower in self.vocabulary_lower:
            # Find the original case version
            for vocab_word in self.vocabulary:
                if vocab_word.lower() == text_lower:
                    return vocab_word, 1.0
        
        # Fuzzy matching
        best_match = None
        best_similarity = 0.0
        
        for vocab_word in self.vocabulary:
            similarity = SequenceMatcher(None, text_lower, vocab_word.lower()).ratio()
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = vocab_word
        
        return best_match, best_similarity
    
    def recognize_rank(self, image: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """
        Recognize rank/guard point from image.
        
        Args:
            image: Input image containing rank text
            
        Returns:
            Tuple of (recognized_rank, confidence, metadata)
        """
        # Extract text using OCR
        ocr_results = self.extract_text_with_ocr(image)
        
        if not ocr_results:
            return None, 0.0, {'error': 'No text detected'}
        
        # Try to find best match from OCR results
        best_rank = None
        best_confidence = 0.0
        best_metadata = {}
        
        for text, ocr_confidence in ocr_results:
            # Validate against vocabulary
            vocab_match, vocab_similarity = self.validate_against_vocabulary(text)
            
            if vocab_match:
                # Combined confidence score
                combined_confidence = (ocr_confidence * 0.6) + (vocab_similarity * 0.4)
                
                if combined_confidence > best_confidence:
                    best_confidence = combined_confidence
                    best_rank = vocab_match
                    best_metadata = {
                        'ocr_text': text,
                        'ocr_confidence': ocr_confidence,
                        'vocab_similarity': vocab_similarity,
                        'combined_confidence': combined_confidence
                    }
        
        # If no vocabulary match found, try partial matching
        if not best_rank:
            best_rank, best_confidence, best_metadata = self._try_partial_matching(ocr_results)
        
        return best_rank, best_confidence, best_metadata
    
    def _try_partial_matching(self, ocr_results: List[Tuple[str, float]]) -> Tuple[Optional[str], float, Dict]:
        """
        Try partial matching for compound terms or partial OCR results.
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            Tuple of (best_match, confidence, metadata)
        """
        # Combine all OCR texts
        combined_text = ' '.join([text for text, _ in ocr_results])
        
        # Look for partial matches
        best_match = None
        best_score = 0.0
        best_metadata = {}
        
        for vocab_word in self.vocabulary:
            # Check if vocabulary word is contained in combined text
            if vocab_word.lower() in combined_text.lower():
                # Calculate confidence based on OCR confidences
                avg_ocr_confidence = np.mean([conf for _, conf in ocr_results])
                partial_confidence = avg_ocr_confidence * 0.8  # Penalty for partial match
                
                if partial_confidence > best_score:
                    best_score = partial_confidence
                    best_match = vocab_word
                    best_metadata = {
                        'match_type': 'partial',
                        'combined_text': combined_text,
                        'avg_ocr_confidence': avg_ocr_confidence
                    }
        
        # Try word-by-word matching
        if not best_match:
            words = combined_text.split()
            for word in words:
                vocab_match, vocab_similarity = self.validate_against_vocabulary(word)
                if vocab_match and vocab_similarity > best_score:
                    best_score = vocab_similarity * 0.7  # Penalty for word-level match
                    best_match = vocab_match
                    best_metadata = {
                        'match_type': 'word_level',
                        'matched_word': word,
                        'vocab_similarity': vocab_similarity
                    }
        
        return best_match, best_score, best_metadata
    
    def batch_recognize_ranks(self, images: List[np.ndarray]) -> List[Tuple[Optional[str], float, Dict]]:
        """
        Recognize ranks from multiple images.
        
        Args:
            images: List of images
            
        Returns:
            List of recognition results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                rank, confidence, metadata = self.recognize_rank(image)
                results.append((rank, confidence, metadata))
                
                if rank:
                    logger.debug(f"Image {i}: Recognized '{rank}' with confidence {confidence:.3f}")
                else:
                    logger.debug(f"Image {i}: No rank recognized")
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append((None, 0.0, {'error': str(e)}))
        
        return results
    
    def get_vocabulary_stats(self) -> Dict:
        """
        Get statistics about the vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            'total_words': len(self.vocabulary),
            'word_lengths': {
                'min': min(len(word) for word in self.vocabulary),
                'max': max(len(word) for word in self.vocabulary),
                'avg': np.mean([len(word) for word in self.vocabulary])
            },
            'categories': {
                'ranks': len([w for w in self.vocabulary if any(r in w for r in ['PRIVATE', 'SERGEANT', 'LIEUTENANT', 'CAPTAIN', 'MAJOR', 'COLONEL', 'GENERAL'])]),
                'positions': len([w for w in self.vocabulary if any(p in w for p in ['GUARD', 'OFFICER', 'SECURITY', 'SUPERVISOR', 'CHIEF'])]),
                'locations': len([w for w in self.vocabulary if any(l in w for l in ['GATE', 'POST', 'BUILDING', 'TOWER', 'STATION'])]),
                'abbreviations': len([w for w in self.vocabulary if len(w) <= 3])
            }
        }


class RankValidator:
    """Validates recognized ranks against business rules."""
    
    def __init__(self):
        """Initialize rank validator."""
        # Define rank hierarchies and valid combinations
        self.rank_hierarchies = {
            'military': ['PRIVATE', 'CORPORAL', 'SERGEANT', 'LIEUTENANT', 'CAPTAIN', 'MAJOR', 'COLONEL', 'GENERAL'],
            'security': ['GUARD', 'SECURITY', 'OFFICER', 'SUPERVISOR', 'CHIEF', 'INSPECTOR', 'COMMANDER', 'DIRECTOR']
        }
        
        # Valid location patterns
        self.location_patterns = [
            r'^GATE\d+$',
            r'^POST\d+$',
            r'^SECTOR\d+$',
            r'^ZONE\d+$',
            r'^[A-H]$',
            r'^\d+$'
        ]
    
    def validate_rank(self, rank: str, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Validate a recognized rank.
        
        Args:
            rank: Recognized rank
            context: Optional context information
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not rank:
            return False, "Empty rank"
        
        # Check if rank exists in any hierarchy
        is_in_hierarchy = any(rank in hierarchy for hierarchy in self.rank_hierarchies.values())
        
        # Check if it's a valid location pattern
        is_location = any(re.match(pattern, rank) for pattern in self.location_patterns)
        
        if is_in_hierarchy or is_location:
            return True, "Valid rank/location"
        
        # Check for common valid patterns
        if len(rank) <= 10 and rank.isalnum():
            return True, "Valid alphanumeric identifier"
        
        return False, f"Unrecognized rank pattern: {rank}"
    
    def suggest_corrections(self, invalid_rank: str) -> List[str]:
        """
        Suggest corrections for invalid ranks.
        
        Args:
            invalid_rank: Invalid rank string
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        # Get all valid ranks
        all_ranks = set()
        for hierarchy in self.rank_hierarchies.values():
            all_ranks.update(hierarchy)
        
        # Find similar ranks
        for valid_rank in all_ranks:
            similarity = SequenceMatcher(None, invalid_rank.lower(), valid_rank.lower()).ratio()
            if similarity > 0.6:
                suggestions.append(valid_rank)
        
        # Sort by similarity
        suggestions.sort(key=lambda x: SequenceMatcher(None, invalid_rank.lower(), x.lower()).ratio(), reverse=True)
        
        return suggestions[:3]  # Return top 3 suggestions