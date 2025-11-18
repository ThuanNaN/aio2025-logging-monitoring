from prometheus_client import Gauge, Histogram, Counter

gpu_allocated_metric = Gauge("process_vram_memory_GB", "GPU memory size in gigabytes.")

image_brightness_metric = Gauge("image_brightness", "Brightness of processed images")

brightness_histogram = Histogram("image_brightness_histogram", 
                                 "Histogram of image brightness",
                                 buckets=[100, 200, 255])

# --- Prometheus Metrics ---
INFERENCE_COUNT = Counter("inference_count", "Number of inferences")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency")
DRIFT_EMBEDDING_DISTANCE = Gauge("drift_embedding_distance", "Embedding drift score")
DRIFT_BRIGHTNESS = Gauge("drift_image_brightness", "Image brightness drift")

# --- Evidently Drift Metrics (YOLO) ---
EVIDENTLY_DATASET_DRIFT = Gauge("evidently_dataset_drift", "Evidently dataset-level drift detection (1=drift, 0=no drift)")
EVIDENTLY_DRIFT_SHARE = Gauge("evidently_drift_share", "Share of drifted features detected by Evidently")
EVIDENTLY_NUM_DRIFTED_FEATURES = Gauge("evidently_num_drifted_features", "Number of features with detected drift")
EVIDENTLY_BRIGHTNESS_DRIFT_SCORE = Gauge("evidently_brightness_drift_score", "Evidently drift score for brightness feature")
EVIDENTLY_CONFIDENCE_DRIFT_SCORE = Gauge("evidently_confidence_drift_score", "Evidently drift score for confidence feature")
EVIDENTLY_DETECTIONS_DRIFT_SCORE = Gauge("evidently_detections_drift_score", "Evidently drift score for detections count")

# --- VQA Metrics ---
VQA_INFERENCE_COUNT = Counter("vqa_inference_count", "Number of VQA inferences")
VQA_INFERENCE_LATENCY = Histogram("vqa_inference_latency_seconds", "VQA inference latency")
VQA_QUESTION_LENGTH = Histogram("vqa_question_length", "VQA question length in words", buckets=[1, 3, 5, 10, 15, 20])
VQA_ANSWER_LENGTH = Histogram("vqa_answer_length", "VQA answer length in words", buckets=[1, 2, 3, 5, 10, 15])
VQA_QUESTION_TYPE = Counter("vqa_question_type", "VQA question types", ["question_type"])

# --- Evidently VQA Drift Metrics ---
VQA_EVIDENTLY_DATASET_DRIFT = Gauge("vqa_evidently_dataset_drift", "VQA dataset-level drift detection (1=drift, 0=no drift)")
VQA_EVIDENTLY_DRIFT_SHARE = Gauge("vqa_evidently_drift_share", "VQA share of drifted features")
VQA_EVIDENTLY_NUM_DRIFTED_FEATURES = Gauge("vqa_evidently_num_drifted_features", "VQA number of drifted features")
VQA_EVIDENTLY_BRIGHTNESS_DRIFT_SCORE = Gauge("vqa_evidently_brightness_drift_score", "VQA brightness drift score")
VQA_EVIDENTLY_QUESTION_LENGTH_DRIFT_SCORE = Gauge("vqa_evidently_question_length_drift_score", "VQA question length drift score")
VQA_EVIDENTLY_ANSWER_LENGTH_DRIFT_SCORE = Gauge("vqa_evidently_answer_length_drift_score", "VQA answer length drift score")
VQA_EVIDENTLY_INFERENCE_TIME_DRIFT_SCORE = Gauge("vqa_evidently_inference_time_drift_score", "VQA inference time drift score")