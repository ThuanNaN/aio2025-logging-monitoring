# Monitoring Setup Guide: Evidently + Prometheus + Grafana + Alertmanager

This guide walks you through setting up complete monitoring, visualization, and alerting for both **YOLO Object Detection** and **VQA (Visual Question Answering)** ML models with Evidently drift detection.

## 🏗️ Architecture

```plaintext
┌────────────────────────────────────────────┐
│      FastAPI Backend (Port 8000)           │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────┐    ┌──────────────┐      │
│  │ YOLO Service │    │  VQA Service │      │
│  │  /v1/yolo/*  │    │  /v1/vqa/*   │      │
│  └──────┬───────┘    └──────┬───────┘      │
│         │                   │              │
│         ▼                   ▼              │
│  ┌──────────────────────────────────┐      │
│  │   Evidently Drift Detectors      │      │
│  │   • YOLO Detector                │      │
│  │   • VQA Detector                 │      │
│  └──────────────┬───────────────────┘      │
│                 │                          │
│                 ▼                          │
│       /metrics (Prometheus format)         │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Prometheus   │ (scrapes every 15s)
          │  Port 9090    │ (evaluates alerts)
          └───────┬───────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  ┌─────────────┐    ┌──────────────┐
  │Alertmanager │    │   Grafana    │
  │  Port 9093  │    │  Port 3000   │
  └─────────────┘    └──────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │     Loki    │
                     │  Port 3100  │
                     └─────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose installed
- Loki Docker driver installed (for log collection)
- Network `aivn-network` created

### Install Loki Docker Driver

```bash
docker plugin install grafana/loki-docker-driver:3.3.2-arm64 --alias loki --grant-all-permissions
```

### Create Docker Network

```bash
docker network create aivn-network
```

## 🚀 Setup Instructions

### Step 1: Start the Backend Service

```bash
cd backend
docker compose up -d --build
```

This will:

- Build the FastAPI backend with both YOLO and VQA services
- Integrate Evidently drift detection for both models
- Expose metrics at `http://localhost:8000/metrics`
- Start collecting drift detection data for both services
- Enable the following endpoints:
  - YOLO: `/v1/yolo/detect/`, `/v1/yolo/drift/*`
  - VQA: `/v1/vqa/answer`, `/v1/vqa/drift/*`

### Step 2: Start Monitoring Stack

```bash
cd platform/monitor
docker compose up -d --build
```

This will start:

- **Prometheus** (port 9090): Metrics collection and alerting
- **Alertmanager** (port 9093): Alert routing and notification
- **Grafana** (port 3000): Visualization dashboards
- **Loki** (port 3100): Log aggregation

### Step 3: Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |

## 📊 Grafana Dashboard Setup

### Option 1: Auto-Provisioned (Recommended)

Both dashboards are automatically loaded when Grafana starts. Navigate to:

- **YOLO Dashboard**: Dashboards → ML Monitoring Dashboards → YOLO ML Model Monitoring with Evidently
- **VQA Dashboard**: Dashboards → ML Monitoring Dashboards → VQA BLIP Model Monitoring with Evidently

### Option 2: Manual Import

**For YOLO:**

1. Login to Grafana (http://localhost:3000)
2. Click **+** → **Import Dashboard**
3. Upload `platform/monitor/grafana/yolo-evidently-dashboard.json`
4. Select **Prometheus** as the datasource
5. Click **Import**

**For VQA:**

1. Click **+** → **Import Dashboard**
2. Upload `platform/monitor/grafana/vqa-evidently-dashboard.json`
3. Select **Prometheus** as the datasource
4. Click **Import**

## 📈 Dashboard Panels

### YOLO Dashboard (10 Panels)

#### 1. **Dataset Drift Status** (Stat)

- Shows binary drift detection (Green = No Drift, Red = Drift Detected)
- Metric: `evidently_dataset_drift`

#### 2. **Drift Share** (Gauge)

- Percentage of features showing drift
- Metric: `evidently_drift_share`
- Thresholds: Green (0-30%), Yellow (30-50%), Red (>50%)

#### 3. **Number of Drifted Features** (Time Series)

- Count of features currently drifting
- Metric: `evidently_num_drifted_features`

#### 4. **Inference Rate** (Time Series)

- Requests per second
- Metric: `rate(inference_count_total[1m])`

#### 5. **Feature-Level Drift Scores** (Time Series)

- Individual drift scores for:
  - Brightness: `evidently_brightness_drift_score`
  - Confidence: `evidently_confidence_drift_score`
  - Detections: `evidently_detections_drift_score`
- Thresholds: Yellow (0.5), Red (0.7)

#### 6. **Inference Latency Percentiles** (Time Series)

- p50, p95, p99 latency
- Helps identify performance degradation

#### 7. **Image Brightness** (Time Series)

- Raw brightness values over time
- Metric: `drift_image_brightness`

#### 8. **Embedding Drift** (Time Series)

- Cosine similarity-based drift
- Metric: `drift_embedding_distance`

#### 9. **GPU Memory Usage** (Time Series)

- VRAM consumption in GB
- Metric: `process_vram_memory_GB`

#### 10. **Active Alerts** (Table)

- Real-time view of firing alerts
- Shows alert name, severity, component

### VQA Dashboard (10 Panels)

#### 1. **VQA Dataset Drift Status** (Stat)

- Binary drift indicator for VQA
- Metric: `vqa_evidently_dataset_drift`

#### 2. **VQA Drift Share** (Gauge)

- Percentage of VQA features showing drift
- Metric: `vqa_evidently_drift_share`
- Thresholds: Green (0-30%), Yellow (30-50%), Red (>50%)

#### 3. **VQA Drifted Features Count** (Time Series)

- Number of drifted VQA features
- Metric: `vqa_evidently_num_drifted_features`

#### 4. **VQA Inference Rate** (Time Series)

- VQA requests per second
- Metric: `rate(vqa_inference_count_total[1m])`

#### 5. **VQA Feature-Level Drift Scores** (Time Series)

- Brightness: `vqa_evidently_brightness_drift_score`
- Question Length: `vqa_evidently_question_length_drift_score`
- Answer Length: `vqa_evidently_answer_length_drift_score`
- Inference Time: `vqa_evidently_inference_time_drift_score`

#### 6. **VQA Inference Latency** (Time Series)

- p50, p95, p99 latency for VQA
- Metric: `vqa_inference_latency_seconds`

#### 7. **Question Type Distribution** (Bar Chart/Time Series)

- Distribution of question types (what, where, who, how_many, etc.)
- Metric: `vqa_question_type_total`

#### 8. **Question Length Trend** (Time Series)

- Average question length over time
- Metric: `vqa_question_length`

#### 9. **Answer Length Trend** (Time Series)

- Average answer length over time
- Metric: `vqa_answer_length`

#### 10. **VQA Active Alerts** (Table)

- Real-time VQA-specific alerts
- Shows alert name, severity, component

## 🚨 Alert Rules

### Drift Alerts

| Alert Name | Condition | Duration | Severity |
|------------|-----------|----------|----------|
| EvidentlyDatasetDriftDetected | dataset_drift == 1 | 2m | Warning |
| EvidentlyHighDriftShare | drift_share > 30% | 3m | Warning |
| EvidentlyCriticalDriftShare | drift_share > 50% | 5m | Critical |
| BrightnessDriftHigh | brightness_drift > 0.7 | 5m | Warning |
| ConfidenceDriftHigh | confidence_drift > 0.7 | 5m | Warning |
| MultipleFeaturesDrifting | drifted_features >= 3 | 5m | Warning |
| HighEmbeddingDrift | embedding_drift > 0.5 | 5m | Warning |

### Performance Alerts

| Alert Name | Condition | Duration | Severity |
|------------|-----------|----------|----------|
| HighInferenceLatency | p95_latency > 2s | 3m | Warning |
| CriticalInferenceLatency | p95_latency > 5s | 2m | Critical |
| LowInferenceRate | rate < 0.01 req/s | 10m | Warning |

### System Alerts

| Alert Name | Condition | Duration | Severity |
|------------|-----------|----------|----------|
| HighGPUMemoryUsage | vram > 10GB | 5m | Warning |
| BackendServiceDown | up == 0 | 1m | Critical |
| AbnormalImageBrightness | brightness < 50 or > 220 | 5m | Info |

## 🔔 Alert Configuration

### View Alerts in Prometheus

1. Go to http://localhost:9090/alerts
2. See all configured rules and their states (Inactive, Pending, Firing)

### View Alerts in Alertmanager

1. Go to http://localhost:9093
2. See active alerts grouped by severity and component

### Configure Alert Notifications

Edit `platform/monitor/alertmanager/alertmanager.yml` to add your notification channels:

**Slack Example:**

```yaml
receivers:
  - name: 'drift-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#ml-alerts'
        title: 'ML Drift Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

**Email Example:**

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
```

**PagerDuty Example:**

```yaml
receivers:
  - name: 'critical-alerts'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
```

After updating, restart Alertmanager:

```bash
docker compose restart alertmanager
```

## 🧪 Testing the Setup

### 1. Send Test Requests

#### YOLO Object Detection

```bash
# Send a single YOLO request
curl -X POST "http://localhost:8000/v1/yolo/detect/" \
  -F "file=@path/to/image.jpg"

# Send multiple YOLO requests (100+ for reference, then 50+ for drift detection)
for i in {1..200}; do
  curl -X POST "http://localhost:8000/v1/yolo/detect/" \
    -F "file=@path/to/image.jpg"
  sleep 0.5
done
```

#### VQA (Visual Question Answering)

```bash
# Send a single VQA request
curl -X POST "http://localhost:8000/v1/vqa/answer" \
  -F "image=@path/to/image.jpg" \
  -F "question=What is in this image?" \
  -F "max_length=50" \
  -F "num_beams=5"

# Send multiple VQA requests with different questions
for i in {1..200}; do
  curl -X POST "http://localhost:8000/v1/vqa/answer" \
    -F "image=@path/to/image.jpg" \
    -F "question=What is in this image?" \
    -F "max_length=50" \
    -F "num_beams=5"
  sleep 0.5
done
```

### 2. Check Metrics in Prometheus

1. Go to http://localhost:9090
2. Query examples:

**YOLO Metrics:**

```promql
evidently_dataset_drift
evidently_drift_share
rate(inference_count_total[1m])
evidently_brightness_drift_score
```

**VQA Metrics:**

```promql
vqa_evidently_dataset_drift
vqa_evidently_drift_share
rate(vqa_inference_count_total[1m])
vqa_evidently_question_length_drift_score
```

### 3. View Dashboards in Grafana

1. Go to http://localhost:3000
2. Navigate to:
   - **YOLO Dashboard**: ML Monitoring Dashboards → YOLO ML Model Monitoring
   - **VQA Dashboard**: ML Monitoring Dashboards → VQA BLIP Model Monitoring
3. Watch metrics update in real-time (10s refresh)

### 4. Check Drift Status via API

**YOLO Drift Status:**

```bash
# Get YOLO drift status
curl http://localhost:8000/v1/yolo/drift/status | jq

# Get YOLO drift summary
curl http://localhost:8000/v1/yolo/drift/summary | jq

# Get YOLO data quality
curl http://localhost:8000/v1/yolo/data-quality | jq
```

**VQA Drift Status:**

```bash
# Get VQA drift status
curl http://localhost:8000/v1/vqa/drift/status | jq

# Get VQA drift summary
curl http://localhost:8000/v1/vqa/drift/summary | jq

# Get VQA data quality
curl http://localhost:8000/v1/vqa/data-quality | jq
```

### 5. Simulate Drift

**For YOLO:**
Send images with significantly different characteristics:

```bash
# Send very dark images (low brightness)
# Send very bright images (high brightness)
# Send images with different object distributions
# Send images with different detection confidence patterns
```

**For VQA:**
Send different types of questions and images:

```bash
# Different question types: what, where, who, how many
# Different question lengths (short vs long questions)
# Different image characteristics (bright vs dark)
# Different complexity images
```

### 6. Reset Reference Data

After testing or model updates:

```bash
# Reset YOLO reference
curl -X POST http://localhost:8000/v1/yolo/drift/reset-reference

# Reset VQA reference
curl -X POST http://localhost:8000/v1/vqa/drift/reset-reference
```

### 7. Check Service Health

```bash
# YOLO health check
curl http://localhost:8000/v1/yolo/health | jq

# VQA health check
curl http://localhost:8000/v1/vqa/health | jq

# Model information
curl http://localhost:8000/v1/yolo/model/info | jq
curl http://localhost:8000/v1/vqa/model/info | jq
```

## 📝 Monitoring Best Practices

### 1. **Establish Baseline**

**For Both YOLO and VQA:**

- Collect 100+ samples before drift detection activates
- Use representative production data from typical use cases
- Document normal ranges for all monitored features:
  - **YOLO**: brightness (50-220), confidence (0.3-0.9), detections (1-20)
  - **VQA**: brightness, question length, answer length, inference time

**VQA Specific:**

- Ensure diverse question types in baseline (what, where, who, how_many, etc.)
- Include variety of image types and complexities
- Document typical answer lengths for different question types

### 2. **Set Alert Thresholds**

- Start conservative, adjust based on false positive rate
- Critical alerts should require immediate action
- Warning alerts should trigger investigation
- Different thresholds for different services:
  - **YOLO**: Focus on visual and detection quality drift
  - **VQA**: Monitor question patterns and answer quality drift

### 3. **Regular Reviews**

- Review drift trends weekly for both services
- Investigate persistent warnings
- Update reference data after valid distribution changes
- Compare drift patterns between YOLO and VQA for shared features (brightness)

### 4. **Response Procedures**

**When YOLO Drift Alert Fires:**

1. Check dashboard for affected features (brightness, confidence, detections)
2. Review recent data changes:
   - New camera or lighting conditions
   - Different object types or distributions
   - Changes in image quality or resolution
3. Validate model performance on current data
4. Decide: retrain model OR update reference data

**When VQA Drift Alert Fires:**

1. Check affected features (question patterns, answer lengths, visual features)
2. Review recent changes:
   - New question types or patterns
   - Different user behavior
   - Changes in image characteristics
   - Performance degradation (inference time drift)
3. Analyze question type distribution changes
4. Validate answer quality on current questions
5. Decide: retrain/fine-tune model OR update reference data

**When Performance Alert Fires:**

1. Check GPU memory and system resources
2. Review inference latency trends
3. Look for unusual request patterns
4. Scale infrastructure if needed

### 5. **Dashboard Customization**

- Add custom panels for your specific metrics
- Set up dashboard variables for filtering
- Create separate dashboards for different time ranges
- Export dashboards as JSON for version control

## 🐛 Troubleshooting

### Metrics Not Showing in Grafana

**Check Prometheus is scraping:**

```bash
curl http://localhost:9090/api/v1/targets
```

Look for `yolo_backend` job with state "UP"

**Check backend metrics endpoint:**

```bash
curl http://localhost:8000/metrics
```

Should return Prometheus format metrics

### Alerts Not Firing

**Verify alert rules loaded:**

```bash
docker compose logs prometheus | grep "alert_rules.yml"
```

**Check Prometheus alerts page:**
http://localhost:9090/alerts

### Dashboard Not Loading

**Check Grafana logs:**

```bash
docker compose logs grafana
```

**Verify datasource connection:**
Grafana → Configuration → Data Sources → Prometheus → Test

### No Drift Detection Results

**Need more data:**

- Send 100+ requests to establish reference for each service
- Send 50+ more requests to trigger detection

**Check detector status:**

```bash
# YOLO detector
curl http://localhost:8000/v1/yolo/drift/summary | jq

# VQA detector
curl http://localhost:8000/v1/vqa/drift/summary | jq
```

**Verify data collection:**

```bash
# Check if samples are being added
curl http://localhost:8000/v1/yolo/drift/status | jq '.reference_samples, .current_samples'
curl http://localhost:8000/v1/vqa/drift/status | jq '.reference_samples, .current_samples'
```

### VQA-Specific Issues

**Question type not detected:**

- Check if question follows expected patterns
- Review question type classification logic
- Verify question preprocessing

**High inference time drift:**

- Check GPU memory usage
- Review batch processing if applicable
- Verify BLIP model is loaded correctly

## 🔄 Maintenance

### Update Alert Rules

1. Edit `platform/monitor/prometheus/alert_rules.yml`

2. Reload Prometheus config:

```bash
curl -X POST http://localhost:9090/-/reload
```

### Backup Data

```bash
# Backup Prometheus data
docker run --rm -v monitor_prometheus-data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data

# Backup Grafana data
docker run --rm -v monitor_grafana-data:/data -v $(pwd):/backup alpine tar czf /backup/grafana-backup.tar.gz /data
```

### Clean Up Old Data

Prometheus retains data for 15 days by default. To change:

```yaml
# In docker-compose.yml, add to prometheus command:
- "--storage.tsdb.retention.time=30d"
```

## 📚 Next Steps

### For Both Services

1. **Integrate with CI/CD**: 
   - Trigger YOLO retraining when visual drift exceeds thresholds
   - Trigger VQA fine-tuning when question/answer patterns drift

2. **A/B Testing**: 
   - Compare drift between YOLO model versions
   - Compare BLIP variants or fine-tuned VQA models

3. **Custom Evidently Reports**: 
   - Generate detailed HTML reports periodically for stakeholders
   - Schedule weekly drift analysis reports

4. **Advanced Alerting**: 
   - Integrate with incident management (PagerDuty, Opsgenie)
   - Set up Slack/Teams notifications for different alert severities

5. **Model Performance Tracking**: 
   - Add accuracy/F1 metrics alongside drift (YOLO)
   - Add BLEU/ROUGE scores for VQA answer quality
   - Track user feedback metrics

### VQA-Specific Enhancements

6. **Question Understanding Analysis**:
   - Track question complexity trends
   - Monitor question type distribution changes
   - Analyze correlation between question types and drift

7. **Answer Quality Monitoring**:
   - Implement answer diversity metrics
   - Track repetitive answer patterns
   - Monitor answer relevance (if ground truth available)

8. **Multi-modal Drift Analysis**:
   - Correlate visual drift with textual drift
   - Identify if drift is image-driven or question-driven

### YOLO-Specific Enhancements

9. **Per-Class Drift Monitoring**:
   - Track drift for specific object classes
   - Monitor class distribution changes

10. **Spatial Drift Analysis**:
    - Monitor bounding box distribution changes
    - Track object size and location patterns

### Integration Ideas

11. **Cross-Service Analysis**:
    - Compare image brightness drift between YOLO and VQA
    - Identify system-wide issues affecting both services

12. **Automated Response**:
    - Auto-scale resources on high inference load
    - Auto-reset reference after scheduled model updates
    - Auto-collect drifted samples for retraining

## 🔗 Useful Links

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
