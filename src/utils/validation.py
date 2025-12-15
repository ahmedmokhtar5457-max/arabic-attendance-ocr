"""
Data Validation Utilities

Validates extracted attendance data against business rules.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates extracted attendance data."""
    
    def __init__(self, validation_rules: Optional[Dict] = None):
        """
        Initialize data validator.
        
        Args:
            validation_rules: Dictionary with validation rules
        """
        self.rules = validation_rules or self._get_default_rules()
        
        # Valid rank patterns
        self.valid_rank_patterns = [
            r'^[A-Z]+\d*$',  # GUARD1, OFFICER, etc.
            r'^[A-Z]+$',     # GUARD, OFFICER, etc.
            r'^\d+$',        # 1, 2, 3, etc.
            r'^[A-H]$',      # A, B, C, etc.
            r'^GATE\d+$',    # GATE1, GATE2, etc.
            r'^POST\d+$',    # POST1, POST2, etc.
        ]
    
    def _get_default_rules(self) -> Dict:
        """Get default validation rules."""
        return {
            'attendance_time_range': (1, 24),
            'max_shift_id': 999,
            'min_shift_id': 1,
            'require_all_fields': False,
            'max_rank_length': 20,
            'allow_empty_cells': True,
            'confidence_threshold': 0.3
        }
    
    def validate_row(self, row_data: Dict) -> Dict:
        """
        Validate a single row of attendance data.
        
        Args:
            row_data: Dictionary with row data
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'field_validations': {}
        }
        
        # Validate attendance time
        attendance_validation = self.validate_attendance_time(
            row_data.get('attendance_time')
        )
        validation_result['field_validations']['attendance_time'] = attendance_validation
        
        if not attendance_validation['is_valid']:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(attendance_validation['errors'])
        
        # Validate rank
        rank_validation = self.validate_rank(row_data.get('rank'))
        validation_result['field_validations']['rank'] = rank_validation
        
        if not rank_validation['is_valid']:
            if not self.rules.get('allow_empty_cells', True) or row_data.get('rank'):
                validation_result['is_valid'] = False
                validation_result['errors'].extend(rank_validation['errors'])
            else:
                validation_result['warnings'].extend(rank_validation['errors'])
        
        # Validate shift ID
        shift_validation = self.validate_shift_id(row_data.get('shift_id'))
        validation_result['field_validations']['shift_id'] = shift_validation
        
        if not shift_validation['is_valid']:
            if not self.rules.get('allow_empty_cells', True) or row_data.get('shift_id'):
                validation_result['is_valid'] = False
                validation_result['errors'].extend(shift_validation['errors'])
            else:
                validation_result['warnings'].extend(shift_validation['errors'])
        
        # Check if all required fields are present
        if self.rules.get('require_all_fields', False):
            missing_fields = []
            for field in ['attendance_time', 'rank', 'shift_id']:
                if not row_data.get(field):
                    missing_fields.append(field)
            
            if missing_fields:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
        
        return validation_result
    
    def validate_attendance_time(self, value: Any) -> Dict:
        """
        Validate attendance time value.
        
        Args:
            value: Attendance time value
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'errors': [],
            'normalized_value': None
        }
        
        if value is None or value == '':
            if not self.rules.get('allow_empty_cells', True):
                result['is_valid'] = False
                result['errors'].append("Attendance time is required")
            return result
        
        try:
            # Convert to integer
            if isinstance(value, str):
                # Remove any non-digit characters
                clean_value = re.sub(r'[^\d]', '', value)
                if not clean_value:
                    result['is_valid'] = False
                    result['errors'].append("No numeric value found in attendance time")
                    return result
                time_int = int(clean_value)
            else:
                time_int = int(value)
            
            result['normalized_value'] = time_int
            
            # Check range
            min_time, max_time = self.rules.get('attendance_time_range', (1, 24))
            if not (min_time <= time_int <= max_time):
                result['is_valid'] = False
                result['errors'].append(f"Attendance time {time_int} out of range ({min_time}-{max_time})")
            
        except (ValueError, TypeError):
            result['is_valid'] = False
            result['errors'].append(f"Invalid attendance time format: {value}")
        
        return result
    
    def validate_rank(self, value: Any) -> Dict:
        """
        Validate rank/guard point value.
        
        Args:
            value: Rank value
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'errors': [],
            'normalized_value': None
        }
        
        if value is None or value == '':
            if not self.rules.get('allow_empty_cells', True):
                result['is_valid'] = False
                result['errors'].append("Rank is required")
            return result
        
        # Convert to string and clean
        rank_str = str(value).strip().upper()
        result['normalized_value'] = rank_str
        
        # Check length
        max_length = self.rules.get('max_rank_length', 20)
        if len(rank_str) > max_length:
            result['is_valid'] = False
            result['errors'].append(f"Rank too long (max {max_length} characters)")
            return result
        
        # Check against valid patterns
        is_valid_pattern = any(
            re.match(pattern, rank_str) for pattern in self.valid_rank_patterns
        )
        
        if not is_valid_pattern:
            result['is_valid'] = False
            result['errors'].append(f"Invalid rank format: {rank_str}")
        
        return result
    
    def validate_shift_id(self, value: Any) -> Dict:
        """
        Validate shift ID value.
        
        Args:
            value: Shift ID value
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'errors': [],
            'normalized_value': None
        }
        
        if value is None or value == '':
            if not self.rules.get('allow_empty_cells', True):
                result['is_valid'] = False
                result['errors'].append("Shift ID is required")
            return result
        
        try:
            # Convert to integer
            if isinstance(value, str):
                # Remove any non-digit characters
                clean_value = re.sub(r'[^\d]', '', value)
                if not clean_value:
                    result['is_valid'] = False
                    result['errors'].append("No numeric value found in shift ID")
                    return result
                shift_int = int(clean_value)
            else:
                shift_int = int(value)
            
            result['normalized_value'] = shift_int
            
            # Check range
            min_shift = self.rules.get('min_shift_id', 1)
            max_shift = self.rules.get('max_shift_id', 999)
            
            if not (min_shift <= shift_int <= max_shift):
                result['is_valid'] = False
                result['errors'].append(f"Shift ID {shift_int} out of range ({min_shift}-{max_shift})")
            
        except (ValueError, TypeError):
            result['is_valid'] = False
            result['errors'].append(f"Invalid shift ID format: {value}")
        
        return result
    
    def validate_batch(self, data_rows: List[Dict]) -> Dict:
        """
        Validate multiple rows of data.
        
        Args:
            data_rows: List of row dictionaries
            
        Returns:
            Dictionary with batch validation results
        """
        batch_result = {
            'total_rows': len(data_rows),
            'valid_rows': 0,
            'invalid_rows': 0,
            'row_validations': [],
            'summary': {
                'common_errors': {},
                'field_error_counts': {
                    'attendance_time': 0,
                    'rank': 0,
                    'shift_id': 0
                }
            }
        }
        
        all_errors = []
        
        for i, row_data in enumerate(data_rows):
            row_validation = self.validate_row(row_data)
            row_validation['row_index'] = i
            
            batch_result['row_validations'].append(row_validation)
            
            if row_validation['is_valid']:
                batch_result['valid_rows'] += 1
            else:
                batch_result['invalid_rows'] += 1
                all_errors.extend(row_validation['errors'])
                
                # Count field-specific errors
                for field, field_validation in row_validation['field_validations'].items():
                    if not field_validation['is_valid']:
                        batch_result['summary']['field_error_counts'][field] += 1
        
        # Analyze common errors
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Get top 5 most common errors
        batch_result['summary']['common_errors'] = dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        return batch_result
    
    def validate_confidence_scores(self, confidence_data: Dict) -> Dict:
        """
        Validate confidence scores against thresholds.
        
        Args:
            confidence_data: Dictionary with confidence information
            
        Returns:
            Dictionary with confidence validation results
        """
        threshold = self.rules.get('confidence_threshold', 0.3)
        
        result = {
            'meets_threshold': True,
            'low_confidence_rows': [],
            'average_confidence': confidence_data.get('average_confidence', 0.0),
            'threshold': threshold
        }
        
        row_confidences = confidence_data.get('row_confidences', [])
        
        for i, confidence in enumerate(row_confidences):
            if confidence < threshold:
                result['meets_threshold'] = False
                result['low_confidence_rows'].append({
                    'row_index': i,
                    'confidence': confidence
                })
        
        return result
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from validate_batch
            
        Returns:
            Formatted validation report
        """
        report_lines = []
        
        # Header
        report_lines.append("ATTENDANCE DATA VALIDATION REPORT")
        report_lines.append("=" * 40)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total = validation_results['total_rows']
        valid = validation_results['valid_rows']
        invalid = validation_results['invalid_rows']
        
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total rows processed: {total}")
        report_lines.append(f"  Valid rows: {valid} ({valid/total*100:.1f}%)")
        report_lines.append(f"  Invalid rows: {invalid} ({invalid/total*100:.1f}%)")
        report_lines.append("")
        
        # Field-specific errors
        field_errors = validation_results['summary']['field_error_counts']
        report_lines.append("FIELD ERROR COUNTS:")
        for field, count in field_errors.items():
            if count > 0:
                report_lines.append(f"  {field}: {count} errors")
        report_lines.append("")
        
        # Common errors
        common_errors = validation_results['summary']['common_errors']
        if common_errors:
            report_lines.append("MOST COMMON ERRORS:")
            for error, count in common_errors.items():
                report_lines.append(f"  {error} ({count} occurrences)")
            report_lines.append("")
        
        # Invalid rows details
        invalid_rows = [r for r in validation_results['row_validations'] if not r['is_valid']]
        if invalid_rows:
            report_lines.append("INVALID ROWS DETAILS:")
            for row_val in invalid_rows[:10]:  # Show first 10 invalid rows
                row_idx = row_val['row_index']
                errors = ', '.join(row_val['errors'])
                report_lines.append(f"  Row {row_idx}: {errors}")
            
            if len(invalid_rows) > 10:
                report_lines.append(f"  ... and {len(invalid_rows) - 10} more invalid rows")
        
        return '\n'.join(report_lines)


class BusinessRuleValidator:
    """Validates data against specific business rules."""
    
    def __init__(self):
        """Initialize business rule validator."""
        pass
    
    def validate_attendance_consistency(self, data_rows: List[Dict]) -> Dict:
        """
        Validate consistency of attendance data.
        
        Args:
            data_rows: List of attendance rows
            
        Returns:
            Dictionary with consistency validation results
        """
        result = {
            'is_consistent': True,
            'issues': [],
            'statistics': {}
        }
        
        if not data_rows:
            return result
        
        # Check for duplicate attendance times
        attendance_times = [row.get('attendance_time') for row in data_rows if row.get('attendance_time')]
        unique_times = set(attendance_times)
        
        if len(attendance_times) != len(unique_times):
            result['is_consistent'] = False
            result['issues'].append("Duplicate attendance times found")
        
        # Check for sequential attendance times
        try:
            numeric_times = [int(t) for t in attendance_times if t and str(t).isdigit()]
            if numeric_times:
                sorted_times = sorted(numeric_times)
                gaps = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
                large_gaps = [g for g in gaps if g > 3]
                
                if large_gaps:
                    result['issues'].append(f"Large gaps in attendance times: {large_gaps}")
        except (ValueError, TypeError):
            pass
        
        # Statistics
        result['statistics'] = {
            'total_rows': len(data_rows),
            'rows_with_attendance_time': len(attendance_times),
            'unique_attendance_times': len(unique_times),
            'attendance_time_range': (min(attendance_times), max(attendance_times)) if attendance_times else None
        }
        
        return result