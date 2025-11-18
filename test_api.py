#!/usr/bin/env python3
"""
ML API Testing Script with Drift Detection
Tests both YOLO Object Detection and VQA services with monitoring
"""

import argparse
import requests
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import sys
from datetime import datetime
import random


class MLAPITester:
    """Test client for YOLO and VQA APIs with drift monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000", service: str = "yolo"):
        self.base_url = base_url.rstrip('/')
        self.service = service.lower()
        
        # Service-specific endpoints
        if self.service == "yolo":
            self.detect_url = f"{self.base_url}/v1/yolo/detect/"
            self.drift_status_url = f"{self.base_url}/v1/yolo/drift/status"
            self.drift_summary_url = f"{self.base_url}/v1/yolo/drift/summary"
            self.drift_reset_url = f"{self.base_url}/v1/yolo/drift/reset-reference"
            self.health_url = f"{self.base_url}/v1/yolo/health"
            self.model_info_url = f"{self.base_url}/v1/yolo/model/info"
        elif self.service == "vqa":
            self.answer_url = f"{self.base_url}/v1/vqa/answer"
            self.drift_status_url = f"{self.base_url}/v1/vqa/drift/status"
            self.drift_summary_url = f"{self.base_url}/v1/vqa/drift/summary"
            self.drift_reset_url = f"{self.base_url}/v1/vqa/drift/reset-reference"
            self.health_url = f"{self.base_url}/v1/vqa/health"
            self.model_info_url = f"{self.base_url}/v1/vqa/model/info"
        
        self.general_health_url = f"{self.base_url}/health"
        
    def check_health(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ {self.service.upper()} service is healthy")
                if isinstance(health_data, dict):
                    if 'model' in health_data:
                        print(f"   Model: {health_data.get('model', {}).get('name', 'N/A')}")
                    if 'drift_detector' in health_data:
                        dd = health_data['drift_detector']
                        print(f"   Drift Detector: {dd.get('reference_samples', 0)} ref, {dd.get('current_samples', 0)} current")
                return True
            else:
                print(f"❌ {self.service.upper()} health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to {self.service.upper()} API: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """Get model information"""
        try:
            response = requests.get(self.model_info_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None
    
    def submit_image(self, image_path: str, question: str = None, verbose: bool = True) -> Optional[Dict]:
        """Submit an image for detection (YOLO) or question answering (VQA)"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        try:
            if self.service == "yolo":
                with open(image_path, 'rb') as f:
                    files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                    response = requests.post(self.detect_url, files=files, timeout=30)
            
            elif self.service == "vqa":
                # Generate a question if not provided
                if not question:
                    question = self._generate_question()
                
                with open(image_path, 'rb') as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                    data = {
                        'question': question,
                        'max_length': 50,
                        'num_beams': 5
                    }
                    response = requests.post(self.answer_url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    if self.service == "yolo":
                        self._print_detection_result(image_path, result)
                    else:
                        self._print_vqa_result(image_path, result, question)
                return result
            else:
                print(f"❌ Request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def _generate_question(self) -> str:
        """Generate a random question for VQA testing"""
        questions = [
            "What is in this image?",
            "What objects can you see?",
            "Describe this image",
            "What is the main subject?",
            "What colors do you see?",
            "Where is this photo taken?",
            "What is happening in this picture?",
            "How many objects are there?",
            "What is the person doing?",
            "What time of day is it?",
        ]
        return random.choice(questions)
    
    def _print_detection_result(self, image_path: str, result: Dict):
        """Pretty print YOLO detection results"""
        print(f"\n{'='*80}")
        print(f"📸 Image: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        print(f"🎯 Objects Detected: {result.get('total_objects', 0)}")
        print(f"💡 Brightness: {result.get('brightness', 0):.2f}")
        print(f"📊 Avg Confidence: {result.get('avg_confidence', 0):.2f}")
        print(f"🔄 Embedding Drift: {result.get('embedding_drift', 0):.4f}")
        
        # Evidently drift info
        evidently_drift = result.get('evidently_drift', {})
        if evidently_drift and evidently_drift.get('drift_share') is not None:
            print(f"\n🔬 Evidently Drift Analysis:")
            print(f"   Dataset Drift: {'🚨 DETECTED' if evidently_drift.get('dataset_drift') else '✅ No Drift'}")
            print(f"   Drift Share: {evidently_drift.get('drift_share', 0):.2%}")
            print(f"   Drifted Features: {evidently_drift.get('num_drifted_features', 0)}")
        
        # Detections
        if result.get('detections'):
            print(f"\n🎯 Detections:")
            for i, det in enumerate(result['detections'][:5], 1):
                print(f"   {i}. {det['class']} (confidence: {det['confidence']:.2f})")
            if len(result['detections']) > 5:
                print(f"   ... and {len(result['detections']) - 5} more")
        print(f"{'='*80}\n")
    
    def _print_vqa_result(self, image_path: str, result: Dict, question: str):
        """Pretty print VQA results"""
        print(f"\n{'='*80}")
        print(f"📸 Image: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        print(f"❓ Question: {result.get('question', question)}")
        print(f"💬 Answer: {result.get('answer', 'N/A')}")
        print(f"⏱️  Inference Time: {result.get('inference_time', 0):.3f}s")
        
        # Features
        features = result.get('features', {})
        if features:
            print(f"\n📊 Features:")
            print(f"   Brightness: {features.get('brightness', 0):.2f}")
            print(f"   Question Type: {features.get('question_type', 'N/A')}")
            print(f"   Question Length: {features.get('question_length', 0)} words")
            print(f"   Answer Length: {features.get('answer_length', 0)} words")
        
        # Evidently drift info
        evidently_drift = result.get('evidently_drift', {})
        if evidently_drift and evidently_drift.get('drift_share') is not None:
            print(f"\n🔬 Evidently Drift Analysis:")
            print(f"   Dataset Drift: {'🚨 DETECTED' if evidently_drift.get('dataset_drift') else '✅ No Drift'}")
            print(f"   Drift Share: {evidently_drift.get('drift_share', 0):.2%}")
            print(f"   Drifted Features: {evidently_drift.get('num_drifted_features', 0)}")
        
        print(f"{'='*80}\n")
    
    def get_drift_status(self, verbose: bool = True) -> Optional[Dict]:
        """Get current drift detection status"""
        try:
            response = requests.get(self.drift_status_url, timeout=5)
            if response.status_code == 200:
                status = response.json()
                if verbose:
                    self._print_drift_status(status)
                return status
            else:
                print(f"❌ Failed to get drift status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def _print_drift_status(self, status: Dict):
        """Pretty print drift status"""
        print(f"\n{'='*80}")
        print(f"🔬 DRIFT DETECTION STATUS")
        print(f"{'='*80}")
        
        if status.get('status') == 'insufficient_data':
            print(f"⚠️  {status.get('message', 'Not enough data')}")
            stats = status.get('statistics', {})
            print(f"📊 Current Window: {stats.get('current_window_size', 0)} samples")
            print(f"📊 Reference Window: {stats.get('reference_window_size', 0)} samples")
            print(f"💡 Need 100 samples for reference, then 50+ for drift detection")
        else:
            drift = status.get('drift_detection', {})
            stats = status.get('statistics', {})
            
            print(f"📊 Sample Counts:")
            print(f"   Current Window: {drift.get('current_samples', 0)}")
            print(f"   Reference Window: {drift.get('reference_samples', 0)}")
            
            print(f"\n🚨 Drift Detection:")
            print(f"   Dataset Drift: {'🔴 DETECTED' if drift.get('dataset_drift') else '🟢 No Drift'}")
            print(f"   Drift Share: {drift.get('drift_share', 0):.2%}")
            print(f"   Drifted Features: {drift.get('num_drifted_features', 0)}/{drift.get('total_features', 0)}")
            
            print(f"\n📈 Statistics:")
            print(f"   Mean Brightness: {stats.get('mean_brightness', 0):.2f}")
            print(f"   Mean Detections: {stats.get('mean_detections', 0):.2f}")
            print(f"   Mean Confidence: {stats.get('mean_confidence', 0):.2f}")
            
            # Feature-level drift
            if drift.get('feature_drifts'):
                print(f"\n🔍 Feature-Level Drift:")
                for feature, info in drift['feature_drifts'].items():
                    if isinstance(info, dict):
                        status_icon = '🔴' if info.get('drift_detected') else '🟢'
                        print(f"   {status_icon} {feature}: {info.get('drift_score', 0):.4f}")
        
        print(f"{'='*80}\n")
    
    def get_drift_summary(self) -> Optional[Dict]:
        """Get drift summary statistics"""
        try:
            response = requests.get(self.drift_summary_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def reset_reference(self) -> bool:
        """Reset the reference data baseline"""
        try:
            response = requests.post(self.drift_reset_url, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {self.service.upper()} - {result.get('message', 'Reference reset successful')}")
                return True
            else:
                print(f"❌ Reset failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False
    
    def batch_submit(self, image_paths: List[str], questions: List[str] = None,
                     delay: float = 0.5, check_drift_every: int = 10, verbose: bool = False) -> Dict:
        """Submit multiple images in batch"""
        results = {
            'service': self.service,
            'total': len(image_paths),
            'successful': 0,
            'failed': 0,
            'results': [],
            'start_time': datetime.now().isoformat(),
        }
        
        print(f"🚀 Starting batch submission of {len(image_paths)} images to {self.service.upper()}...")
        print(f"⏱️  Delay between requests: {delay}s")
        print(f"🔬 Checking drift every {check_drift_every} requests\n")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing {os.path.basename(image_path)}...", end=' ')
            
            # Get question for VQA
            question = questions[i % len(questions)] if questions else None
            
            result = self.submit_image(image_path, question=question, verbose=verbose)
            if result:
                results['successful'] += 1
                
                if self.service == "yolo":
                    results['results'].append({
                        'image': os.path.basename(image_path),
                        'objects': result.get('total_objects', 0),
                        'brightness': result.get('brightness', 0),
                        'drift': result.get('embedding_drift', 0),
                    })
                    print(f"✅ ({result.get('total_objects', 0)} objects)")
                else:  # VQA
                    results['results'].append({
                        'image': os.path.basename(image_path),
                        'question': result.get('question', ''),
                        'answer': result.get('answer', ''),
                        'inference_time': result.get('inference_time', 0),
                    })
                    print(f"✅ {result.get('answer', 'N/A')[:40]}...")
            else:
                results['failed'] += 1
                print(f"❌ Failed")
            
            # Check drift status periodically
            if i % check_drift_every == 0:
                print(f"\n📊 Drift Check at {i} requests:")
                self.get_drift_status(verbose=True)
            
            # Delay between requests
            if i < len(image_paths):
                time.sleep(delay)
        
        results['end_time'] = datetime.now().isoformat()
        
        print(f"\n{'='*80}")
        print(f"✅ Batch Complete!")
        print(f"   Service: {self.service.upper()}")
        print(f"   Total: {results['total']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        print(f"{'='*80}\n")
        
        return results


def find_images(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all images in a directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).glob(f"**/*{ext}"))
        image_paths.extend(Path(directory).glob(f"**/*{ext.upper()}"))
    
    return sorted([str(p) for p in image_paths])


def main():
    parser = argparse.ArgumentParser(
        description='Test YOLO and VQA APIs with drift detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO - Check API health
  python test_api.py --service yolo --health
  
  # YOLO - Submit a single image
  python test_api.py --service yolo --image path/to/image.jpg
  
  # YOLO - Submit all images in a directory
  python test_api.py --service yolo --dir path/to/images --delay 1.0
  
  # YOLO - Load test (submit same image 200 times)
  python test_api.py --service yolo --image test.jpg --repeat 200 --delay 0.5
  
  # VQA - Submit with custom question
  python test_api.py --service vqa --image img.jpg --question "What is this?"
  
  # VQA - Submit with random questions
  python test_api.py --service vqa --dir images/ --delay 1.0
  
  # VQA - Multiple questions
  python test_api.py --service vqa --image img.jpg --questions "What is this?" "Where is it?" "How many?"
  
  # Get drift status
  python test_api.py --service yolo --drift-status
  python test_api.py --service vqa --drift-status
  
  # Reset reference data
  python test_api.py --service yolo --reset-reference
  python test_api.py --service vqa --reset-reference
        """
    )
    
    parser.add_argument('--service', default='yolo', choices=['yolo', 'vqa'],
                        help='Service to test: yolo or vqa (default: yolo)')
    parser.add_argument('--url', default='http://localhost:8000',
                        help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--image', help='Path to a single image')
    parser.add_argument('--dir', help='Directory containing images')
    parser.add_argument('--repeat', type=int, help='Repeat the image submission N times')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests in seconds (default: 0.5)')
    parser.add_argument('--check-drift', type=int, default=10,
                        help='Check drift every N requests (default: 10)')
    parser.add_argument('--health', action='store_true',
                        help='Check API health')
    parser.add_argument('--drift-status', action='store_true',
                        help='Get current drift status')
    parser.add_argument('--reset-reference', action='store_true',
                        help='Reset reference data baseline')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output for each request')
    parser.add_argument('--output', help='Save results to JSON file')
    
    # VQA-specific arguments
    parser.add_argument('--question', help='Question for VQA (single question)')
    parser.add_argument('--questions', nargs='+', help='Multiple questions for VQA')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = MLAPITester(base_url=args.url, service=args.service)
    
    # Health check
    if args.health or not any([args.image, args.dir, args.drift_status, args.reset_reference]):
        if not tester.check_health():
            sys.exit(1)
        if args.health:
            sys.exit(0)
    
    # Drift status
    if args.drift_status:
        tester.get_drift_status(verbose=True)
        sys.exit(0)
    
    # Reset reference
    if args.reset_reference:
        tester.reset_reference()
        sys.exit(0)
    
    # Submit images
    image_paths = []
    
    if args.image:
        if args.repeat:
            image_paths = [args.image] * args.repeat
            print(f"🔁 Repeating {args.image} {args.repeat} times")
        else:
            image_paths = [args.image]
    elif args.dir:
        image_paths = find_images(args.dir)
        if not image_paths:
            print(f"❌ No images found in {args.dir}")
            sys.exit(1)
        print(f"📁 Found {len(image_paths)} images in {args.dir}")
    else:
        parser.print_help()
        sys.exit(1)
    
    # Check health before starting
    if not tester.check_health():
        sys.exit(1)
    
    # Prepare questions for VQA
    questions = None
    if args.service == 'vqa':
        if args.questions:
            questions = args.questions
        elif args.question:
            questions = [args.question]
    
    # Submit images
    if len(image_paths) == 1 and not args.repeat:
        question = questions[0] if questions else None
        result = tester.submit_image(image_paths[0], question=question, verbose=True)
        if result and args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to {args.output}")
    else:
        results = tester.batch_submit(
            image_paths,
            questions=questions,
            delay=args.delay, 
            check_drift_every=args.check_drift,
            verbose=args.verbose
        )
        
        # Final drift check
        print(f"📊 Final {args.service.upper()} Drift Status:")
        tester.get_drift_status(verbose=True)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {args.output}")


if __name__ == '__main__':
    main()
