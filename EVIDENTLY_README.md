# Evidently Drift Detection Integration

This project includes **Evidently** as an advanced drift detector for both YOLO object detection and VQA (Visual Question Answering) services, providing comprehensive monitoring alongside cosine similarity-based drift detection.

## Overview

The system implements separate Evidently drift detectors for two AI services:
1. **YOLO Object Detection** - Monitors image and detection features
2. **VQA (BLIP)** - Monitors visual, textual, and performance features

Both detectors use the same core architecture but track different feature sets appropriate to their respective tasks.

## Features

### 1. **Evidently Drift Detector**
- **Dataset-level drift detection**: Detects when the overall data distribution has shifted
- **Feature-level drift detection**: Identifies which specific features are drifting
- **Statistical tests**: Uses advanced statistical methods (Kolmogorov-Smirnov, Chi-squared, etc.)
- **Reference window**: Maintains a baseline (default: 100 samples) for comparison
- **Detection window**: Analyzes current samples (default: 50 samples) against reference
- **Data quality monitoring**: Tracks data quality metrics over time
- **Flexible window management**: Automatically maintains reference and current windows

### 2. **Monitored Features**

#### YOLO Object Detection
- `brightness`: Image brightness values
- `num_detections`: Number of objects detected per image
- `avg_confidence`: Average detection confidence scores
- `embedding_*`: Up to 10 embedding features from the model

#### VQA (Visual Question Answering)
**Visual Features:**
- `brightness`: Average pixel intensity
- `contrast`: Standard deviation of pixel values
- `aspect_ratio`: Image width/height ratio
- `width`, `height`: Image dimensions

**Question Features:**
- `question_length`: Number of words
- `question_char_length`: Character count
- `question_tokens`: Tokenized input count
- `question_type`: Detected category (what, where, who, how_many, etc.)

**Answer Features:**
- `answer_length`: Number of words in response
- `answer_char_length`: Character count in response

**Performance:**
- `inference_time`: Time to generate answer

### 3. **Prometheus Metrics**

The following metrics are exposed for Prometheus/Grafana:

#### YOLO Metrics

- `evidently_dataset_drift`: Binary indicator (1=drift detected, 0=no drift)
- `evidently_drift_share`: Proportion of features showing drift (0.0 to 1.0)
- `evidently_num_drifted_features`: Count of drifted features
- `evidently_brightness_drift_score`: Drift score for brightness feature
- `evidently_confidence_drift_score`: Drift score for confidence feature
- `evidently_detections_drift_score`: Drift score for detections count

#### VQA Metrics

- `vqa_evidently_dataset_drift`: Binary indicator for VQA drift
- `vqa_evidently_drift_share`: Proportion of VQA features drifting
- `vqa_evidently_num_drifted_features`: Count of drifted VQA features
- `vqa_evidently_brightness_drift_score`: Visual brightness drift
- `vqa_evidently_question_length_drift_score`: Question length drift
- `vqa_evidently_answer_length_drift_score`: Answer length drift
- `vqa_evidently_inference_time_drift_score`: Inference time drift

### 4. **API Endpoints**

#### YOLO Object Detection

**Detection Endpoint (Enhanced)**

```bash
POST /v1/yolo/detect/
```

Response includes Evidently drift information:

```json
{
  "detections": [...],
  "total_objects": 5,
  "device": "cuda:0",
  "brightness": 128.5,
  "embedding_drift": 0.023,
  "avg_confidence": 0.87,
  "evidently_drift": {
    "dataset_drift": false,
    "drift_share": 0.0,
    "num_drifted_features": 0,
    "feature_drift_scores": {...}
  }
}
```

**Monitoring Endpoints**

```bash
GET /v1/yolo/drift/status          # Get drift status
GET /v1/yolo/drift/summary         # Get detailed drift summary
POST /v1/yolo/drift/reset-reference # Reset reference dataset
GET /v1/yolo/data-quality          # Get data quality report
GET /v1/yolo/model/info            # Get model information
GET /v1/yolo/health                # Health check
```

#### VQA (Visual Question Answering)

**VQA Endpoint**

```bash
POST /v1/vqa/answer
```

Response includes comprehensive drift information:

```json
{
  "question": "What is in this image?",
  "answer": "a cat sitting on a couch",
  "inference_time": 0.234,
  "model": "blip-vqa-base",
  "device": "cuda:0",
  "features": {
    "brightness": 145.3,
    "contrast": 52.1,
    "question_length": 5,
    "question_type": "what",
    "answer_length": 6,
    ...
  },
  "evidently_drift": {
    "dataset_drift": false,
    "drift_share": 0.0,
    "num_drifted_features": 0,
    "feature_drift_scores": {...}
  }
}
```

**VQA Monitoring Endpoints**

```bash
GET /v1/vqa/drift/status           # Get drift status
GET /v1/vqa/drift/summary          # Get detailed drift summary
POST /v1/vqa/drift/reset-reference # Reset reference dataset
GET /v1/vqa/data-quality           # Get data quality report
GET /v1/vqa/model/info             # Get model information
GET /v1/vqa/health                 # Health check
```

## How It Works

1. **Data Collection**: Each inference request (YOLO or VQA) adds a sample with relevant features to the Evidently detector.
   - **YOLO**: brightness, num_detections, avg_confidence, embeddings
   - **VQA**: brightness, contrast, dimensions, question/answer features, inference time

2. **Reference Window**: The first 100 samples establish the reference baseline for comparison.

3. **Current Window Management**: Subsequent samples populate the current window (50 samples).
   - Once current window is full and reference exists, drift detection begins
   - Older current samples automatically move to reference if reference is not full

4. **Drift Detection**: When sufficient data exists, Evidently compares current window against reference using:
   - Statistical tests (Kolmogorov-Smirnov for numerical features, Chi-squared for categorical)
   - Dataset-level drift calculation
   - Feature-level drift scoring

5. **Metrics Update**: Drift results are exposed as Prometheus metrics for Grafana visualization.

6. **Continuous Monitoring**: The system tracks both:
   - Simple cosine similarity drift (YOLO embeddings)
   - Advanced Evidently statistical drift (both YOLO and VQA)

## Usage Examples

### 1. Start the Backend

```bash
cd backend
docker compose up -d --build
```

### 2. YOLO Object Detection

**Send Detection Request**

```bash
curl -X POST "http://localhost:8000/v1/yolo/detect/" \
  -F "file=@path/to/image.jpg"
```

**Check Drift Status**

```bash
curl http://localhost:8000/v1/yolo/drift/status
```

**Get Detailed Summary**

```bash
curl http://localhost:8000/v1/yolo/drift/summary
```

**Reset Reference**

```bash
curl -X POST http://localhost:8000/v1/yolo/drift/reset-reference
```

### 3. VQA (Visual Question Answering)

**Send VQA Request**

```bash
curl -X POST "http://localhost:8000/v1/vqa/answer" \
  -F "image=@path/to/image.jpg" \
  -F "question=What is in this image?" \
  -F "max_length=50" \
  -F "num_beams=5"
```

**Check VQA Drift Status**

```bash
curl http://localhost:8000/v1/vqa/drift/status
```

**Get VQA Drift Summary**

```bash
curl http://localhost:8000/v1/vqa/drift/summary
```

**Reset VQA Reference**

```bash
curl -X POST http://localhost:8000/v1/vqa/drift/reset-reference
```

### 4. View Metrics in Prometheus

Access Prometheus at `http://localhost:9090` and query:

**YOLO Metrics:**

- `evidently_dataset_drift`
- `evidently_drift_share`
- `evidently_brightness_drift_score`
- `evidently_confidence_drift_score`

**VQA Metrics:**

- `vqa_evidently_dataset_drift`
- `vqa_evidently_drift_share`
- `vqa_evidently_question_length_drift_score`
- `vqa_evidently_answer_length_drift_score`

### 5. View Dashboards in Grafana

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

- **YOLO Dashboard**: `yolo-evidently-dashboard.json`
- **VQA Dashboard**: `vqa-evidently-dashboard.json`

## Configuration

### YOLO Drift Detector

Configured in `backend/app/api/v1/detector/evidently_drift.py`:

```python
detector = EvidentlyDriftDetector(
    reference_window_size=100,    # Baseline samples
    detection_window_size=50,     # Current window size
    drift_threshold=0.5           # Drift threshold (0-1)
)
```

### VQA Drift Detector

Configured in `backend/app/api/v1/detector/evidently_vqa_drift.py`:

```python
detector = EvidentlyVQADriftDetector(
    reference_window_size=100,    # Baseline samples
    detection_window_size=50,     # Current window size
    drift_threshold=0.5           # Drift threshold (0-1)
)
```

### Window Management

- **Reference Window**: First N samples establishing the baseline
- **Current Window**: Most recent M samples for comparison
- **Auto-rotation**: When current window exceeds size, oldest samples move to reference
- **Reset**: Use reset endpoints to re-establish reference after model updates

## Grafana Dashboards

### YOLO Dashboard

Pre-configured dashboard: `platform/monitor/grafana/yolo-evidently-dashboard.json`

**Key Panels:**

1. **Dataset Drift Indicator**: `evidently_dataset_drift`
2. **Drift Share**: `evidently_drift_share`
3. **Drifted Features Count**: `evidently_num_drifted_features`
4. **Feature Drift Scores**:
   - `evidently_brightness_drift_score`
   - `evidently_confidence_drift_score`
   - `evidently_detections_drift_score`
5. **Performance Metrics**: Inference latency, GPU usage, etc.

### VQA Dashboard

Pre-configured dashboard: `platform/monitor/grafana/vqa-evidently-dashboard.json`

**Key Panels:**

1. **VQA Dataset Drift**: `vqa_evidently_dataset_drift`
2. **VQA Drift Share**: `vqa_evidently_drift_share`
3. **VQA Drifted Features**: `vqa_evidently_num_drifted_features`
4. **Visual Features Drift**:
   - `vqa_evidently_brightness_drift_score`
5. **Text Features Drift**:
   - `vqa_evidently_question_length_drift_score`
   - `vqa_evidently_answer_length_drift_score`
6. **Performance Drift**:
   - `vqa_evidently_inference_time_drift_score`
7. **Question Type Distribution**: By category
8. **Answer Length Trends**: Over time

## Comparison: Simple vs Evidently Drift Detection

| Feature | Simple (Cosine) | Evidently |
|---------|----------------|-----------||
| Method | Cosine similarity on embeddings | Statistical tests (KS, Chi2, etc.) |
| Scope | YOLO only | YOLO + VQA |
| Granularity | Single drift score | Per-feature + dataset-level |
| Statistical rigor | Basic | Advanced |
| Interpretability | Moderate | High (feature-level details) |
| Performance overhead | Low | Moderate |
| Features tracked | Embeddings only | Multiple feature types |
| Categorical support | No | Yes (VQA question types) |

Both methods run in parallel for complementary insights.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  YOLO Routes │              │  VQA Routes  │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                              │                     │
│         ▼                              ▼                     │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ Evidently    │              │ Evidently    │            │
│  │ YOLO Drift   │              │ VQA Drift    │            │
│  │ Detector     │              │ Detector     │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                              │                     │
│         └──────────────┬───────────────┘                    │
│                        ▼                                     │
│                ┌───────────────┐                            │
│                │   Prometheus  │                            │
│                │    Metrics    │                            │
│                └───────┬───────┘                            │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Prometheus   │
                 │   (Scraper)   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    Grafana    │
                 │  (Dashboards) │
                 └───────────────┘
```

## Dependencies

- `evidently==0.4.47`: Core drift detection library
- `pandas`: Data manipulation for Evidently
- `numpy`: Numerical operations
- Automatically installed via `backend/requirements.txt`

## Troubleshooting

### Common Issues

**Issue**: "Not enough data for drift detection" or `insufficient_reference_data`

- **Reason**: Reference window not yet full
- **Solution**: Send at least 100 requests to establish reference baseline

**Issue**: "insufficient_current_data"

- **Reason**: Current window needs more samples
- **Solution**: Send 50+ more requests after reference is established

**Issue**: Drift always showing 0.0 or no drift detected

- **Possible Causes**:
  - Data lacks variation (all similar images/questions)
  - Insufficient sample size
  - Features not changing between windows
- **Solutions**:
  - Test with diverse inputs (different images, question types)
  - Ensure sufficient samples in both windows
  - Check data quality report for feature statistics

**Issue**: High memory usage

- **Causes**: Large window sizes, many features
- **Solutions**:
  - Reduce `reference_window_size` (e.g., 50 instead of 100)
  - Reduce `detection_window_size` (e.g., 30 instead of 50)
  - Periodically reset reference with `/drift/reset-reference`

**Issue**: Drift detection results seem incorrect

- **Solutions**:
  - Check `/drift/summary` for detailed feature-level scores
  - Review `/data-quality` for data statistics
  - Verify reference window represents expected baseline
  - Consider resetting reference if data distribution has legitimately changed

### Debugging

```bash
# Check detector status
curl http://localhost:8000/v1/yolo/drift/summary
curl http://localhost:8000/v1/vqa/drift/summary

# Get data quality metrics
curl http://localhost:8000/v1/yolo/data-quality
curl http://localhost:8000/v1/vqa/data-quality

# Check model health
curl http://localhost:8000/v1/yolo/health
curl http://localhost:8000/v1/vqa/health

# View Prometheus metrics directly
curl http://localhost:8000/metrics | grep evidently
```

## Advanced Usage

### Custom Window Sizes

Modify detector initialization in the respective detector files:

```python
# For specialized use cases
detector = EvidentlyDriftDetector(
    reference_window_size=200,  # Larger reference for stability
    detection_window_size=30,   # Smaller window for faster detection
    drift_threshold=0.3         # Lower threshold for sensitivity
)
```

### Periodic Reference Updates

For production systems with evolving data:

```bash
# Schedule reference resets (e.g., weekly)
0 0 * * 0 curl -X POST http://localhost:8000/v1/yolo/drift/reset-reference
0 0 * * 0 curl -X POST http://localhost:8000/v1/vqa/drift/reset-reference
```

### Integration with CI/CD

```bash
# Check drift before deployment
DRIFT_SHARE=$(curl -s http://localhost:8000/v1/yolo/drift/status | jq -r '.drift_share')
if (( $(echo "$DRIFT_SHARE > 0.5" | bc -l) )); then
  echo "High drift detected: $DRIFT_SHARE"
  # Trigger retraining or alert
fi
```

## Next Steps

### Monitoring & Alerting

1. Configure Grafana alerts based on:
   - `evidently_dataset_drift` (binary drift detection)
   - `evidently_drift_share` (proportion of drifted features)
   - `vqa_evidently_drift_share` (VQA-specific drift)

2. Set up Prometheus alert rules (see `platform/monitor/prometheus/alert_rules.yml`)

### Automation

1. **Automated Retraining**:
   - Trigger when `drift_share > 0.5`
   - Collect drifted samples for retraining dataset

2. **Reference Updates**:
   - Schedule periodic resets after model updates
   - Reset when distributional shift is expected (seasonal changes, etc.)

3. **Reporting**:
   - Export Evidently HTML reports for stakeholders
   - Generate weekly drift summaries

### Advanced Monitoring

1. **Custom Drift Tests**:
   - Implement domain-specific drift detection
   - Add custom metrics for business KPIs

2. **Data Quality Checks**:
   - Monitor data quality metrics alongside drift
   - Set up alerts for data quality degradation

3. **Multi-model Comparison**:
   - Compare drift across multiple model versions
   - A/B test drift detection configurations

## Additional Resources

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [YOLO Monitoring Guide](./README.md)
- [VQA Setup Guide](./VQA_README.md)
- [Prometheus Configuration](./platform/monitor/prometheus/)
- [Grafana Dashboards](./platform/monitor/grafana/)
