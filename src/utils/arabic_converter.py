"""
Arabic to English Converter

Converts Arabic numerals and text to English equivalents.
"""

import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ArabicToEnglishConverter:
    """Converts Arabic numerals and text to English."""
    
    def __init__(self):
        """Initialize converter with mapping dictionaries."""
        # Arabic-Indic digits to Western Arabic digits
        self.arabic_to_english_digits = {
            '٠': '0',
            '١': '1', 
            '٢': '2',
            '٣': '3',
            '٤': '4',
            '٥': '5',
            '٦': '6',
            '٧': '7',
            '٨': '8',
            '٩': '9'
        }
        
        # Extended Arabic numerals (Persian/Farsi variants)
        self.extended_arabic_digits = {
            '۰': '0',
            '۱': '1',
            '۲': '2', 
            '۳': '3',
            '۴': '4',
            '۵': '5',
            '۶': '6',
            '۷': '7',
            '۸': '8',
            '۹': '9'
        }
        
        # Combine all digit mappings
        self.all_digit_mappings = {
            **self.arabic_to_english_digits,
            **self.extended_arabic_digits
        }
        
        # Arabic number words to English digits
        self.arabic_number_words = {
            'صفر': '0',
            'واحد': '1',
            'اثنان': '2',
            'ثلاثة': '3',
            'أربعة': '4',
            'خمسة': '5',
            'ستة': '6',
            'سبعة': '7',
            'ثمانية': '8',
            'تسعة': '9',
            'عشرة': '10',
            'أحد عشر': '11',
            'اثنا عشر': '12',
            'ثلاثة عشر': '13',
            'أربعة عشر': '14',
            'خمسة عشر': '15',
            'ستة عشر': '16',
            'سبعة عشر': '17',
            'ثمانية عشر': '18',
            'تسعة عشر': '19',
            'عشرون': '20',
            'واحد وعشرون': '21',
            'اثنان وعشرون': '22',
            'ثلاثة وعشرون': '23',
            'أربعة وعشرون': '24'
        }
    
    def convert_digits(self, text: str) -> str:
        """
        Convert Arabic digits to English digits.
        
        Args:
            text: Text containing Arabic digits
            
        Returns:
            Text with English digits
        """
        if not text:
            return text
        
        result = text
        
        # Convert each Arabic digit to English
        for arabic_digit, english_digit in self.all_digit_mappings.items():
            result = result.replace(arabic_digit, english_digit)
        
        return result
    
    def convert_number_words(self, text: str) -> str:
        """
        Convert Arabic number words to English digits.
        
        Args:
            text: Text containing Arabic number words
            
        Returns:
            Text with English digits
        """
        if not text:
            return text
        
        result = text
        
        # Sort by length (longest first) to handle compound numbers
        sorted_words = sorted(self.arabic_number_words.items(), 
                            key=lambda x: len(x[0]), reverse=True)
        
        for arabic_word, english_digit in sorted_words:
            result = result.replace(arabic_word, english_digit)
        
        return result
    
    def extract_numbers(self, text: str) -> list:
        """
        Extract all numbers from text (both Arabic and English).
        
        Args:
            text: Input text
            
        Returns:
            List of extracted numbers as strings
        """
        if not text:
            return []
        
        # First convert Arabic digits to English
        converted_text = self.convert_digits(text)
        
        # Extract all number sequences
        numbers = re.findall(r'\d+', converted_text)
        
        return numbers
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text by converting digits and cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Convert digits
        result = self.convert_digits(text)
        
        # Convert number words
        result = self.convert_number_words(result)
        
        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result.strip())
        
        return result
    
    def is_arabic_digit(self, char: str) -> bool:
        """
        Check if character is an Arabic digit.
        
        Args:
            char: Single character
            
        Returns:
            True if Arabic digit, False otherwise
        """
        return char in self.all_digit_mappings
    
    def contains_arabic_digits(self, text: str) -> bool:
        """
        Check if text contains Arabic digits.
        
        Args:
            text: Input text
            
        Returns:
            True if contains Arabic digits, False otherwise
        """
        if not text:
            return False
        
        return any(self.is_arabic_digit(char) for char in text)
    
    def convert_time_format(self, time_text: str) -> Optional[str]:
        """
        Convert Arabic time format to standard format.
        
        Args:
            time_text: Time text (e.g., "١٢:٣٠" or "12:30")
            
        Returns:
            Standardized time format or None if invalid
        """
        if not time_text:
            return None
        
        # Convert Arabic digits
        converted = self.convert_digits(time_text)
        
        # Extract time components
        time_pattern = r'(\d{1,2})[:\-\.]?(\d{2})?'
        match = re.match(time_pattern, converted.strip())
        
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            
            # Validate time
            if 0 <= hour <= 24 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        
        return None
    
    def validate_attendance_time(self, time_value: str) -> tuple:
        """
        Validate attendance time value (1-24 range).
        
        Args:
            time_value: Time value to validate
            
        Returns:
            Tuple of (is_valid, normalized_value, error_message)
        """
        if not time_value:
            return False, None, "Empty time value"
        
        try:
            # Convert Arabic digits
            converted = self.convert_digits(time_value.strip())
            
            # Extract numeric value
            numbers = re.findall(r'\d+', converted)
            
            if not numbers:
                return False, None, "No numeric value found"
            
            # Use first number found
            time_int = int(numbers[0])
            
            # Validate range (1-24 for attendance time)
            if 1 <= time_int <= 24:
                return True, str(time_int), None
            else:
                return False, str(time_int), f"Time value {time_int} out of range (1-24)"
                
        except ValueError as e:
            return False, None, f"Invalid numeric format: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"
    
    def get_conversion_stats(self, text: str) -> Dict:
        """
        Get statistics about Arabic content in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with conversion statistics
        """
        if not text:
            return {
                'total_chars': 0,
                'arabic_digits': 0,
                'english_digits': 0,
                'arabic_digit_ratio': 0.0,
                'contains_arabic': False
            }
        
        total_chars = len(text)
        arabic_digits = sum(1 for char in text if self.is_arabic_digit(char))
        english_digits = sum(1 for char in text if char.isdigit())
        
        return {
            'total_chars': total_chars,
            'arabic_digits': arabic_digits,
            'english_digits': english_digits,
            'arabic_digit_ratio': arabic_digits / total_chars if total_chars > 0 else 0.0,
            'contains_arabic': arabic_digits > 0
        }