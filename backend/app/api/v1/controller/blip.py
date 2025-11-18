"""
BLIP VQA Controller
Handles BLIP model loading and inference for Visual Question Answering
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Dict, Any, List
import time
import numpy as np


class BLIPController:
    """Controller for BLIP Visual Question Answering model"""
    
    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        """
        Initialize BLIP model and processor
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load BLIP model and processor"""
        try:
            print(f"Loading BLIP model: {self.model_name} on {self.device}")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("BLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            raise
    
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 50,
        num_beams: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question about an image
        
        Args:
            image: PIL Image
            question: Question text
            max_length: Maximum answer length
            num_beams: Number of beams for beam search
            
        Returns:
            Dictionary containing answer and metadata
        """
        start_time = time.time()
        
        try:
            # Process inputs
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Extract features for monitoring
            features = self._extract_features(image, question, answer, inputs)
            
            return {
                "answer": answer,
                "question": question,
                "inference_time": inference_time,
                "model_name": self.model_name,
                "device": self.device,
                "features": features
            }
            
        except Exception as e:
            print(f"Error during VQA inference: {e}")
            raise
    
    def batch_answer(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_length: int = 50,
        num_beams: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch
        
        Args:
            images: List of PIL Images
            questions: List of question texts
            max_length: Maximum answer length
            num_beams: Number of beams for beam search
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for image, question in zip(images, questions):
            result = self.answer_question(image, question, max_length, num_beams)
            results.append(result)
        return results
    
    def _extract_features(
        self,
        image: Image.Image,
        question: str,
        answer: str,
        inputs: Dict
    ) -> Dict[str, Any]:
        """
        Extract features for drift detection
        
        Args:
            image: Input image
            question: Question text
            answer: Generated answer
            inputs: Processed model inputs
            
        Returns:
            Dictionary of extracted features
        """
        # Image features
        img_array = np.array(image)
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Question features
        question_length = len(question.split())
        question_char_length = len(question)
        question_type = self._detect_question_type(question)
        
        # Answer features
        answer_length = len(answer.split())
        answer_char_length = len(answer)
        
        # Text embedding features (simplified)
        # In production, you might want to use more sophisticated embeddings
        question_tokens = len(inputs['input_ids'][0])
        
        return {
            # Image features
            "brightness": float(brightness),
            "contrast": float(contrast),
            "aspect_ratio": float(aspect_ratio),
            "width": int(width),
            "height": int(height),
            
            # Question features
            "question_length": int(question_length),
            "question_char_length": int(question_char_length),
            "question_type": question_type,
            "question_tokens": int(question_tokens),
            
            # Answer features
            "answer_length": int(answer_length),
            "answer_char_length": int(answer_char_length),
        }
    
    def _detect_question_type(self, question: str) -> str:
        """
        Detect question type from question text
        
        Args:
            question: Question text
            
        Returns:
            Question type string
        """
        question_lower = question.lower().strip()
        
        if question_lower.startswith("what"):
            return "what"
        elif question_lower.startswith("where"):
            return "where"
        elif question_lower.startswith("when"):
            return "when"
        elif question_lower.startswith("who"):
            return "who"
        elif question_lower.startswith("why"):
            return "why"
        elif question_lower.startswith("how many") or question_lower.startswith("how much"):
            return "how_many"
        elif question_lower.startswith("how"):
            return "how"
        elif question_lower.startswith("is") or question_lower.startswith("are") or \
             question_lower.startswith("do") or question_lower.startswith("does"):
            return "yes_no"
        else:
            return "other"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "processor_loaded": self.processor is not None,
            "model_loaded": self.model is not None
        }


# Global instance
blip_controller = None


def get_blip_controller() -> BLIPController:
    """Get or create BLIP controller instance"""
    global blip_controller
    if blip_controller is None:
        blip_controller = BLIPController()
    return blip_controller
