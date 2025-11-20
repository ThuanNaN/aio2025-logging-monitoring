"""
YOLO Object Detection Controller
Handles YOLO model loading and inference for object detection
"""

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import Dict, Any, List
import time


class YOLOController:
    """Controller for YOLO Object Detection model"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        if model_path:
            self.load_model()
        
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to(self.device)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def set_model(self, model: YOLO):
        """
        Set YOLO model directly (for pre-loaded models)
        
        Args:
            model: Pre-loaded YOLO model
        """
        self.model = model
        self.device = str(next(model.model.parameters()).device)
    
    def detect_objects(
        self,
        image: Image.Image,
        annotated_filepath: str = None,
        conf_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Detect objects in an image
        
        Args:
            image: PIL Image
            annotated_filepath: Path to save annotated image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing detections and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first or set_model().")
        
        start_time = time.time()
        
        try:
            # Perform object detection
            results = self.model(image, conf=conf_threshold)
            
            # Save annotated image if path provided
            if annotated_filepath:
                results[0].save(annotated_filepath)
            
            # Process and format detection results
            detections = []
            total_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract detection information
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    total_confidence += conf
                    bbox = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
                    
                    detections.append({
                        'class': self.model.names[cls],
                        'confidence': conf,
                        'bounding_box': {
                            'x1': bbox[0],
                            'y1': bbox[1],
                            'x2': bbox[2],
                            'y3': bbox[3]
                        }
                    })
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(detections) if len(detections) > 0 else 0.0
            
            # Extract features for monitoring
            features = self._extract_features(image, detections, avg_confidence, results)
            
            return {
                "detections": detections,
                "total_objects": len(detections),
                "avg_confidence": avg_confidence,
                "inference_time": inference_time,
                "device": self.device,
                "features": features
            }
            
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            raise
    
    def batch_detect(
        self,
        images: List[Image.Image],
        annotated_filepaths: List[str] = None,
        conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in multiple images in batch
        
        Args:
            images: List of PIL Images
            annotated_filepaths: List of paths to save annotated images
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries
        """
        results = []
        for i, image in enumerate(images):
            filepath = annotated_filepaths[i] if annotated_filepaths and i < len(annotated_filepaths) else None
            result = self.detect_objects(image, filepath, conf_threshold)
            results.append(result)
        return results
    
    def _extract_features(
        self,
        image: Image.Image,
        detections: List[Dict],
        avg_confidence: float,
        results: Any
    ) -> Dict[str, Any]:
        """
        Extract features for drift detection
        
        Args:
            image: Input image
            detections: List of detections
            avg_confidence: Average confidence score
            results: YOLO model results
            
        Returns:
            Dictionary of extracted features
        """
        # Image features
        grayscale_image = image.convert("L")
        image_array = np.asarray(grayscale_image)
        brightness = float(image_array.mean())
        contrast = float(image_array.std())
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Detection features
        num_detections = len(detections)
        
        # Extract embeddings from detection features
        embedding_features = self._extract_embedding_features(results, brightness)
        
        # GPU memory if available
        vram_allocated = 0.0
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated(0) / 1e9
        
        return {
            # Image features
            "brightness": brightness,
            "contrast": contrast,
            "aspect_ratio": aspect_ratio,
            "width": int(width),
            "height": int(height),
            
            # Detection features
            "num_detections": num_detections,
            "avg_confidence": float(avg_confidence),
            
            # Embedding features
            "embedding_features": embedding_features,
            
            # System features
            "vram_allocated": vram_allocated
        }
    
    def _extract_embedding_features(self, results, brightness: float) -> np.ndarray:
        """
        Extract embedding features from YOLO results
        
        Args:
            results: YOLO model results
            brightness: Image brightness value
            
        Returns:
            Numpy array of embedding features
        """
        try:
            # Extract features from detection boxes
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes_data = results[0].boxes.data
                if len(boxes_data) > 0:
                    # Average box features as embedding
                    embedding = boxes_data[:, :4].mean(dim=0).unsqueeze(0).cpu().numpy()
                else:
                    # Default embedding if no detections
                    embedding = np.array([[brightness, brightness, brightness, brightness]])
            else:
                # Default embedding if no detections
                embedding = np.array([[brightness, brightness, brightness, brightness]])
            
            return embedding
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Fallback embedding
            return np.array([[brightness, brightness, brightness, brightness]])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        if self.model is None:
            return {
                "model_loaded": False,
                "error": "Model not loaded"
            }
        
        try:
            return {
                "model_name": self.model.model.yaml.get("model", "YOLO"),
                "model_version": self.model.model.yaml.get("version", "unknown"),
                "num_classes": self.model.model.yaml.get("nc", None),
                "class_names": self.model.names,
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "model_loaded": True
            }
        except Exception as e:
            return {
                "model_loaded": True,
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "error": f"Could not retrieve full model info: {str(e)}"
            }


# Global instance
yolo_controller = None


def get_yolo_controller() -> YOLOController:
    """Get or create YOLO controller instance"""
    global yolo_controller
    if yolo_controller is None:
        yolo_controller = YOLOController()
    return yolo_controller