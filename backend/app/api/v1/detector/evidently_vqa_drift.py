"""
Evidently Drift Detector for VQA (Visual Question Answering)
Monitors drift in visual features, question patterns, and answer distributions
"""

from flask import json
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset


class EvidentlyVQADriftDetector:
    """Drift detector for VQA models using Evidently"""
    
    def __init__(
        self,
        reference_window_size: int = 30,
        detection_window_size: int = 20,
        drift_threshold: float = 0.5
    ):
        """
        Initialize VQA drift detector
        
        Args:
            reference_window_size: Number of samples for reference dataset
            detection_window_size: Number of samples for current dataset
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.drift_threshold = drift_threshold
        
        # Data storage
        self.reference_data: List[Dict[str, Any]] = []
        self.current_data: List[Dict[str, Any]] = []
        
        # Drift results
        self.drift_detected = False
        self.drift_report: Optional[Dict[str, Any]] = None
        self.last_check_time: Optional[datetime] = None
        
        # Feature columns to monitor
        self.numeric_features = [
            'brightness', 'contrast', 'aspect_ratio', 'width', 'height',
            'question_length', 'question_char_length', 'question_tokens',
            'answer_length', 'answer_char_length', 'inference_time'
        ]
        self.categorical_features = ['question_type']
        
    def add_sample(self, features: Dict[str, Any]):
        """
        Add a new sample for drift monitoring
        
        Args:
            features: Dictionary of features from VQA inference
        """
        # Prepare sample with all features
        sample = {
            'timestamp': datetime.now().isoformat(),
            **features
        }
        
        # Add to current window
        self.current_data.append(sample)
        
        # Maintain window size
        if len(self.current_data) > self.detection_window_size:
            # Move oldest current samples to reference if needed
            if len(self.reference_data) < self.reference_window_size:
                self.reference_data.append(self.current_data.pop(0))
            else:
                self.current_data.pop(0)
        
        # Build reference from current if reference is not full
        if len(self.reference_data) < self.reference_window_size and \
           len(self.current_data) >= self.detection_window_size:
            # Use older samples as reference
            samples_needed = self.reference_window_size - len(self.reference_data)
            self.reference_data.extend(self.current_data[:samples_needed])
    
    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect drift using Evidently
        
        Returns:
            Dictionary containing drift detection results
        """
        # Check if we have enough data
        if len(self.reference_data) < self.reference_window_size:
            return {
                'drift_detected': False,
                'reason': 'insufficient_reference_data',
                'reference_size': len(self.reference_data),
                'required_reference_size': self.reference_window_size,
                'current_size': len(self.current_data),
                'message': f'Need {self.reference_window_size - len(self.reference_data)} more reference samples'
            }
        
        if len(self.current_data) < self.detection_window_size:
            return {
                'drift_detected': False,
                'reason': 'insufficient_current_data',
                'reference_size': len(self.reference_data),
                'current_size': len(self.current_data),
                'required_current_size': self.detection_window_size,
                'message': f'Need {self.detection_window_size - len(self.current_data)} more current samples'
            }
        
        try:
            # Convert to DataFrames
            reference_df = pd.DataFrame(self.reference_data)
            current_df = pd.DataFrame(self.current_data)
            
            # Create drift report with DataDriftPreset
            drift_report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            snapshot = drift_report.run(
                reference_data=reference_df,
                current_data=current_df
            )
            
            # Extract results
            report_dict = json.loads(snapshot.json())
            metrics = report_dict.get('metrics', [])
            
            # Get dataset drift metric
            dataset_drift_metric = None
            
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    dataset_drift_metric = metric['result']
                    break
            
            if dataset_drift_metric is None:
                return {
                    'drift_detected': False,
                    'reason': 'report_generation_failed',
                    'message': 'Could not extract drift metrics from report'
                }
            
            # Get drift by columns from the dataset drift metric result
            drift_by_columns = dataset_drift_metric.get('drift_by_columns', {})
            
            # Calculate drift statistics
            drift_detected = dataset_drift_metric.get('dataset_drift', False)
            drift_share = dataset_drift_metric.get('drift_share', 0.0)
            num_drifted_features = dataset_drift_metric.get('number_of_drifted_columns', 0)
            
            # Feature-level drift scores
            feature_drift_scores = {}
            for col_name, col_data in drift_by_columns.items():
                if col_name in self.numeric_features + self.categorical_features:
                    feature_drift_scores[col_name] = {
                        'drift_detected': col_data.get('drift_detected', False),
                        'drift_score': col_data.get('drift_score', 0.0),
                        'stattest_name': col_data.get('stattest_name', 'unknown')
                    }
            
            # Store results
            self.drift_detected = drift_detected
            self.drift_report = {
                'dataset_drift': drift_detected,
                'drift_share': drift_share,
                'num_drifted_features': num_drifted_features,
                'feature_drift_scores': feature_drift_scores,
                'reference_size': len(self.reference_data),
                'current_size': len(self.current_data),
                'timestamp': datetime.now().isoformat()
            }
            self.last_check_time = datetime.now()
            
            return self.drift_report
            
        except Exception as e:
            return {
                'drift_detected': False,
                'reason': 'error',
                'error': str(e),
                'message': f'Error during drift detection: {str(e)}'
            }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of current drift status
        
        Returns:
            Dictionary with drift summary
        """
        if self.drift_report is None:
            return {
                'status': 'no_analysis',
                'reference_samples': len(self.reference_data),
                'current_samples': len(self.current_data),
                'message': 'No drift analysis performed yet'
            }
        
        return {
            'status': 'drift_detected' if self.drift_detected else 'no_drift',
            'drift_detected': self.drift_detected,
            'drift_share': self.drift_report.get('drift_share', 0.0),
            'num_drifted_features': self.drift_report.get('num_drifted_features', 0),
            'reference_samples': len(self.reference_data),
            'current_samples': len(self.current_data),
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'feature_drift_scores': self.drift_report.get('feature_drift_scores', {})
        }
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate data quality report
        
        Returns:
            Dictionary with data quality metrics
        """
        if len(self.current_data) == 0:
            return {
                'status': 'no_data',
                'message': 'No data available for quality analysis'
            }
        
        try:
            current_df = pd.DataFrame(self.current_data)
            
            # Extract key quality metrics
            quality_summary = {
                'total_samples': len(current_df),
                'num_features': len(current_df.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate basic statistics
            for feature in self.numeric_features:
                if feature in current_df.columns:
                    quality_summary[f'{feature}_mean'] = float(current_df[feature].mean())
                    quality_summary[f'{feature}_std'] = float(current_df[feature].std())
                    quality_summary[f'{feature}_min'] = float(current_df[feature].min())
                    quality_summary[f'{feature}_max'] = float(current_df[feature].max())
            
            # Question type distribution
            if 'question_type' in current_df.columns:
                type_dist = current_df['question_type'].value_counts().to_dict()
                quality_summary['question_type_distribution'] = type_dist
            
            return quality_summary
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': f'Error generating quality report: {str(e)}'
            }
    
    def reset_reference(self):
        """Reset reference dataset and use current data as new reference"""
        if len(self.current_data) >= self.reference_window_size:
            self.reference_data = self.current_data[:self.reference_window_size].copy()
            self.current_data = self.current_data[self.reference_window_size:].copy()
        else:
            self.reference_data = self.current_data.copy()
            self.current_data = []
        
        self.drift_detected = False
        self.drift_report = None
        self.last_check_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics
        
        Returns:
            Dictionary with detector stats
        """
        stats = {
            'reference_size': len(self.reference_data),
            'current_size': len(self.current_data),
            'reference_window_size': self.reference_window_size,
            'detection_window_size': self.detection_window_size,
            'drift_threshold': self.drift_threshold,
            'drift_detected': self.drift_detected,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'monitored_numeric_features': self.numeric_features,
            'monitored_categorical_features': self.categorical_features
        }
        
        # Add current data statistics if available
        if len(self.current_data) > 0:
            current_df = pd.DataFrame(self.current_data)
            
            # Question type distribution
            if 'question_type' in current_df.columns:
                stats['question_type_distribution'] = current_df['question_type'].value_counts().to_dict()
            
            # Average metrics
            for feature in ['brightness', 'question_length', 'answer_length', 'inference_time']:
                if feature in current_df.columns:
                    stats[f'avg_{feature}'] = float(current_df[feature].mean())
        
        return stats


# Global instance
vqa_drift_detector = None


def get_vqa_drift_detector() -> EvidentlyVQADriftDetector:
    """Get or create VQA drift detector instance"""
    global vqa_drift_detector
    if vqa_drift_detector is None:
        vqa_drift_detector = EvidentlyVQADriftDetector()
    return vqa_drift_detector
