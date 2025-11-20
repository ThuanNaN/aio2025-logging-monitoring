import numpy as np
import cv2
from PIL import Image

def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on image for YOLO detections
    
    :param image: PIL Image or numpy array
    :param detections: List of detection dictionaries
    :return: Image with bounding boxes drawn
    """
    image = np.array(image)
    for detection in detections:
        # Extract bounding box coordinates and class
        x1 = int(detection["bounding_box"]["x1"])
        y1 = int(detection["bounding_box"]["y1"])
        x2 = int(detection["bounding_box"]["x2"])
        y2 = int(detection["bounding_box"]["y2"])
        label = detection["class"]
        confidence = detection["confidence"]
        
        # Draw rectangle
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label and confidence
        text = f"{label} ({confidence:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def calculate_brightness(image):
    """
    Calculate average brightness of an image
    
    :param image: PIL Image or numpy array
    :return: Average brightness value (0-255)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    return np.mean(gray)


def format_drift_info(drift_info):
    """
    Format drift detection information for display
    
    :param drift_info: Dictionary containing drift information
    :return: Formatted string
    """
    if not drift_info:
        return "No drift information available"
    
    output = "### Drift Detection Results\n\n"
    output += f"**Dataset Drift Detected:** {'Yes' if drift_info.get('dataset_drift', False) else 'No'}\n\n"
    output += f"**Drift Share:** {drift_info.get('drift_share', 0):.2%}\n\n"
    output += f"**Number of Drifted Features:** {drift_info.get('num_drifted_features', 0)}\n\n"
    
    feature_scores = drift_info.get('feature_drift_scores', {})
    if feature_scores:
        output += "**Feature Drift Scores:**\n"
        for feature, score_info in feature_scores.items():
            drift_detected = score_info.get('drift_detected', False)
            drift_score = score_info.get('drift_score', 0.0)
            output += f"- {feature}: {drift_score:.4f} ({'Drift' if drift_detected else 'No Drift'})\n"
    
    return output
