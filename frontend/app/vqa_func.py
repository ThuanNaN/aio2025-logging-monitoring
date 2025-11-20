import gradio as gr
import requests
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")
if API_URL is None:
    raise ValueError("API_URL environment variable not set")

def answer_question(image, question, max_length=50, num_beams=5):
    """
    Send image and question to VQA API and get answer
    
    :param image: Input image (numpy array or PIL Image)
    :param question: Question about the image
    :param max_length: Maximum answer length
    :param num_beams: Number of beams for beam search
    :return: Answer text, metadata text
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create a temporary file to upload
    temp_filename = "temp_vqa_upload.jpg"
    image.save(temp_filename)
    
    # Prepare file and form data for upload
    files = {'image': open(temp_filename, 'rb')}
    data = {
        'question': question,
        'max_length': max_length,
        'num_beams': num_beams
    }
    
    try:
        # Send request to API
        url = f"{API_URL}/v1/vqa/answer"
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise an exception for bad responses
        
        # Parse response
        result = response.json()
        
        # Extract answer and metadata
        answer = result.get('answer', 'No answer generated')
        
        # Prepare metadata text
        metadata_text = f"**Answer:** {answer}\n\n"
        metadata_text += f"**Inference Time:** {result.get('inference_time', 0):.3f}s\n"
        metadata_text += f"**Model:** {result.get('model_name', 'N/A')}\n"
        metadata_text += f"**Device:** {result.get('device', 'N/A')}\n"
        
        # Add drift information if available
        drift_info = result.get('drift_detection', {})
        if drift_info:
            metadata_text += f"\n**Drift Detection:**\n"
            metadata_text += f"- Dataset Drift: {'Yes' if drift_info.get('dataset_drift', False) else 'No'}\n"
            metadata_text += f"- Drift Share: {drift_info.get('drift_share', 0):.2%}\n"
            metadata_text += f"- Drifted Features: {drift_info.get('num_drifted_features', 0)}\n"
        
        # Clean up temporary file
        os.remove(temp_filename)
        
        return answer, metadata_text
    
    except requests.RequestException as e:
        return f"Error: {str(e)}", "Request failed"
    except Exception as e:
        return f"Error: {str(e)}", "Processing failed"
    finally:
        # Ensure file is closed and cleaned up
        if 'files' in locals():
            files['image'].close()
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
