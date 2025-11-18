# Monitoring Architecture Diagram

## Complete System Flow

```plaintext
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLIENT REQUESTS                                 │
└────────────────────┬────────────────────────┬────────────────────────────┘
                     │                        │
                     ↓                        ↓
       ┌─────────────────────────┐ ┌─────────────────────────┐
       │   Frontend (Streamlit)  │ │   VQA Client/Test       │
       │   Port: 8501            │ │   Script                │
       └────────────┬────────────┘ └────────────┬────────────┘
                    │                           │
                    ↓ POST /v1/yolo/detect/    ↓ POST /v1/vqa/answer/
       ┌────────────────────────────────────────────────────────────┐
       │           ML Backend (FastAPI) - Port: 8000                │
       │                                                            │
       │  ┌─────────────────────┐    ┌─────────────────────┐      │
       │  │  YOLO Service       │    │  VQA Service        │      │
       │  │  /v1/yolo/*         │    │  /v1/vqa/*          │      │
       │  │                     │    │                     │      │
       │  │ • YOLO11x Model     │    │ • BLIP-VQA Model    │      │
       │  │ • Object Detection  │    │ • Question Answer   │      │
       │  └──────────┬──────────┘    └──────────┬──────────┘      │
       │             │                          │                 │
       │  ┌──────────▼──────────┐    ┌──────────▼──────────┐      │
       │  │ EvidentlyDrift      │    │ EvidentlyVQADrift   │      │
       │  │ Detector (YOLO)     │    │ Detector (VQA)      │      │
       │  │                     │    │                     │      │
       │  │ • Brightness        │    │ • Brightness        │      │
       │  │ • Confidence        │    │ • Answer Length     │      │
       │  │ • Detections        │    │ • Question Length   │      │
       │  │ • Embeddings        │    │ • Question Type     │      │
       │  └──────────┬──────────┘    └──────────┬──────────┘      │
       │             │                          │                 │
       │             └──────────┬───────────────┘                 │
       │                        │                                 │
       │              ┌─────────▼─────────┐                       │
       │              │ Metrics Export    │                       │
       │              │ /metrics          │                       │
       │              │                   │                       │
       │              │ YOLO + VQA metrics│                       │
       │              └───────────────────┘                       │
       └────────────────────────┬──────────────────────────────────┘
                                │
                                ↓ Scrape every 15s
                   ┌─────────────────────────┐
                   │   Prometheus            │
                   │   Port: 9090            │
                   │                         │
                   │  ┌──────────────────┐   │
                   │  │ Time Series DB   │   │
                   │  └────────┬─────────┘   │
                   │           │             │
                   │  ┌────────▼─────────┐   │
                   │  │ Alert Rules      │   │
                   │  │ Evaluation       │   │
                   │  └────────┬─────────┘   │
                   └───────────┼─────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ↓                             ↓
   ┌────────────────────┐         ┌────────────────────┐
   │   Alertmanager     │         │     Grafana        │
   │   Port: 9093       │         │     Port: 3000     │
   │                    │         │                    │
   │  ┌──────────────┐  │         │  ┌──────────────┐  │
   │  │ Alert Router │  │         │  │  Dashboards  │  │
   │  └──────┬───────┘  │         │  │              │  │
   │         │          │         │  │ • YOLO (10)  │  │
   │  ┌──────▼───────┐  │         │  │ • VQA (10)   │  │
   │  │ Receivers    │  │         │  └──────────────┘  │
   │  │ • Slack      │  │         │                    │
   │  │ • Email      │  │         │  ┌──────────────┐  │
   │  │ • PagerDuty  │  │         │  │  Alerts View │  │
   │  │ • Webhook    │  │         │  └──────────────┘  │
   │  └──────────────┘  │         │                    │
   └────────────────────┘         │  ┌──────────────┐  │
                                  │  │  Queries     │  │
                                  │  │  (PromQL)    │  │
                                  │  └──────────────┘  │
                                  └────────────────────┘
                                            │
                                            ↓
                                  ┌────────────────────┐
                                  │   Loki (Logs)      │
                                  │   Port: 3100       │
                                  └────────────────────┘
```

## Metrics Flow

### YOLO Service Flow

```plaintext
YOLO Inference Request (/v1/yolo/detect)
    │
    ├─► Process Image with YOLO11x
    │       │
    │       ├─► Extract Features (brightness, detections, confidence)
    │       │
    │       └─► Extract Embeddings (CLIP)
    │
    ├─► Evidently Drift Detection (YOLO)
    │       │
    │       ├─► Add Sample to Window (100 ref, 50 current)
    │       │
    │       ├─► Compare with Reference (statistical tests)
    │       │
    │       └─► Calculate Drift Scores
    │               │
    │               ├─► Dataset Drift (binary)
    │               ├─► Drift Share (0-1)
    │               ├─► Feature Drifts (per feature)
    │               └─► Num Drifted Features
    │
    └─► Update YOLO Prometheus Metrics
            │
            ├─► evidently_dataset_drift = 0 or 1
            ├─► evidently_drift_share = 0.XX
            ├─► evidently_num_drifted_features = N
            ├─► evidently_brightness_drift_score = 0.XX
            ├─► evidently_confidence_drift_score = 0.XX
            ├─► evidently_detections_drift_score = 0.XX
            ├─► yolo_inference_count_total += 1
            ├─► yolo_inference_latency_seconds.observe(duration)
            ├─► drift_image_brightness = XX.XX
            ├─► drift_embedding_distance = 0.XX
            └─► process_vram_memory_GB = X.XX
```

### VQA Service Flow

```plaintext
VQA Inference Request (/v1/vqa/answer)
    │
    ├─► Process Image + Question with BLIP
    │       │
    │       ├─► Extract Visual Features (brightness)
    │       │
    │       ├─► Extract Text Features (question, answer)
    │       │
    │       └─► Generate Answer
    │
    ├─► Evidently VQA Drift Detection
    │       │
    │       ├─► Add Sample to Window (100 ref, 50 current)
    │       │
    │       ├─► Compare with Reference (statistical tests)
    │       │
    │       └─► Calculate Drift Scores
    │               │
    │               ├─► Dataset Drift (binary)
    │               ├─► Drift Share (0-1)
    │               ├─► Feature Drifts (per feature)
    │               └─► Num Drifted Features
    │
    └─► Update VQA Prometheus Metrics
            │
            ├─► vqa_evidently_dataset_drift = 0 or 1
            ├─► vqa_evidently_drift_share = 0.XX
            ├─► vqa_evidently_num_drifted_features = N
            ├─► vqa_evidently_brightness_drift_score = 0.XX
            ├─► vqa_evidently_answer_length_drift_score = 0.XX
            ├─► vqa_evidently_question_length_drift_score = 0.XX
            ├─► vqa_evidently_question_type_drift_score = 0.XX
            ├─► vqa_inference_count_total += 1
            ├─► vqa_inference_latency_seconds.observe(duration)
            └─► vqa_drift_image_brightness = XX.XX
                    │
                    ↓
            Prometheus Scrapes /metrics (both services)
                    │
                    ↓
            Evaluate Alert Rules
                    │
                    ├─► If drift_share > 0.3 → Fire Alert
                    ├─► If YOLO latency > 2s → Fire Alert
                    ├─► If VQA latency > 5s → Fire Alert
                    └─► If service down → Fire Alert
                            │
                            ↓
                    Send to Alertmanager
                            │
                            ├─► Route by severity/component
                            ├─► Group related alerts
                            ├─► Apply silences
                            └─► Send notifications
                                    │
                                    ├─► Slack
                                    ├─► Email
                                    └─► Webhook
```

## Alert Severity Levels

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                        CRITICAL                              │
│  • Service Down (up == 0)                                   │
│  • YOLO Inference Latency > 5s                              │
│  • VQA Inference Latency > 8s                               │
│  • Drift Share > 50% (both services)                        │
│                                                             │
│  Action: Immediate investigation required                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        WARNING                               │
│  • Dataset Drift Detected (YOLO or VQA)                     │
│  • Drift Share > 30% (either service)                       │
│  • High Feature Drift (> 0.7)                               │
│  • YOLO Latency > 2s or VQA Latency > 5s                    │
│  • GPU Memory > 10GB                                        │
│  • Low Inference Rate (either service)                      │
│  • VQA Question Pattern Shift                               │
│                                                             │
│  Action: Monitor and investigate if persists                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         INFO                                 │
│  • Abnormal Image Brightness                                │
│  • Data Quality Notices                                     │
│                                                             │
│  Action: Log for future reference                           │
└─────────────────────────────────────────────────────────────┘
```

## Dashboard Panel Layout

### YOLO Dashboard (10 Panels)

```plaintext
┌──────────────────────────────────────────────────────────────────┐
│                    YOLO ML Model Monitoring                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Dataset     │  Drift Share │  # Drifted   │  Inference Rate    │
│  Drift       │  (Gauge)     │  Features    │  (Time Series)     │
│  (Stat)      │              │  (TS)        │                    │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                                                                  │
│  Feature-Level Drift Scores (Time Series)                       │
│  ├─ Brightness Drift                                            │
│  ├─ Confidence Drift                                            │
│  └─ Detections Drift                                            │
│                                                                  │
├──────────────────────────────┬───────────────────────────────────┤
│  Inference Latency           │  Image Brightness                │
│  Percentiles (p50/p95/p99)   │  (Time Series)                   │
│  (Time Series)               │                                  │
├──────────────────────────────┼───────────────────────────────────┤
│  Embedding Drift             │  GPU Memory Usage                │
│  (Cosine Similarity)         │  (Time Series)                   │
│  (Time Series)               │                                  │
├──────────────────────────────┴───────────────────────────────────┤
│  Active Alerts (Table)                                          │
│  Alert Name │ Severity │ Component │ Description                │
└──────────────────────────────────────────────────────────────────┘
```

### VQA Dashboard (10 Panels)

```plaintext
┌──────────────────────────────────────────────────────────────────┐
│                    VQA ML Model Monitoring                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Dataset     │  Drift Share │  # Drifted   │  Inference Rate    │
│  Drift       │  (Gauge)     │  Features    │  (Time Series)     │
│  (Stat)      │              │  (TS)        │                    │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                                                                  │
│  Feature-Level Drift Scores (Time Series)                       │
│  ├─ Brightness Drift                                            │
│  ├─ Answer Length Drift                                         │
│  ├─ Question Length Drift                                       │
│  └─ Question Type Drift                                         │
│                                                                  │
├──────────────────────────────┬───────────────────────────────────┤
│  Inference Latency           │  Image Brightness                │
│  Percentiles (p50/p95/p99)   │  (Time Series)                   │
│  (Time Series)               │                                  │
├──────────────────────────────┼───────────────────────────────────┤
│  Question Type Distribution  │  Answer Length Distribution      │
│  (Pie Chart)                 │  (Histogram)                     │
│                              │                                  │
├──────────────────────────────┴───────────────────────────────────┤
│  Active Alerts (Table)                                          │
│  Alert Name │ Severity │ Component │ Description                │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow Timeline

### YOLO Service Timeline

```plaintext
t=0s     │ Backend starts, YOLO Evidently detector initialized
         │ Reference window = empty (needs 100 samples)
         │
t=15s    │ Prometheus first scrape (no metrics yet)
         │
t=30s    │ First YOLO inference request arrives
         │ ├─ YOLO11x detection
         │ ├─ Evidently adds sample (1/100)
         │ └─ Metrics updated (yolo_*)
         │
t=45s    │ Prometheus scrapes metrics
         │ └─ Stores time series data
         │
...      │ More YOLO requests processed...
         │
t=300s   │ 100th YOLO request processed
         │ ├─ Reference baseline established!
         │ └─ Drift detection now active
         │
t=315s   │ 101st-150th YOLO requests
         │ ├─ Current window = 50 samples
         │ ├─ Evidently compares with reference
         │ └─ Drift scores calculated
         │
t=330s   │ Prometheus evaluates alert rules
         │ ├─ evidently_drift_share = 0.35 (> 0.3 threshold)
         │ └─ Alert: EvidentlyHighDriftShare (pending)
         │
t=510s   │ Alert fires (after 3m duration)
         │ ├─ Sent to Alertmanager
         │ ├─ Routed to "drift-alerts" receiver
         │ └─ Notification sent (Slack/Email)
         │
t=600s   │ User checks YOLO Grafana dashboard
         │ ├─ Sees drift indicators
         │ ├─ Reviews feature-level scores (brightness, confidence, detections)
         │ └─ Investigates root cause
         │
t=900s   │ User resets YOLO reference data
         │ curl -X POST /v1/yolo/drift/reset-reference
         │ └─ New baseline established
         │
t=1200s  │ Drift score normalizes
         │ └─ Alert resolves
```

### VQA Service Timeline

```plaintext
t=0s     │ Backend starts, VQA Evidently detector initialized
         │ Reference window = empty (needs 100 samples)
         │
t=30s    │ First VQA inference request arrives
         │ ├─ BLIP VQA processing (image + question)
         │ ├─ Evidently adds sample (1/100)
         │ └─ Metrics updated (vqa_*)
         │
t=45s    │ Prometheus scrapes VQA metrics
         │ └─ Stores time series data
         │
...      │ More VQA requests processed...
         │
t=400s   │ 100th VQA request processed
         │ ├─ Reference baseline established!
         │ └─ Drift detection now active
         │
t=420s   │ 101st-150th VQA requests
         │ ├─ Current window = 50 samples
         │ ├─ Evidently compares with reference
         │ └─ Drift scores calculated (question, answer features)
         │
t=450s   │ Question pattern shift detected
         │ ├─ vqa_evidently_question_type_drift_score = 0.85
         │ └─ Alert: VQAQuestionPatternDrift (pending)
         │
t=630s   │ Alert fires (after 3m duration)
         │ └─ Notification sent
         │
t=720s   │ User checks VQA Grafana dashboard
         │ ├─ Reviews question type distribution
         │ ├─ Checks answer length drift
         │ └─ Identifies question pattern shift
         │
t=1000s  │ User resets VQA reference data
         │ curl -X POST /v1/vqa/drift/reset-reference
         │ └─ New baseline established
         │
t=1300s  │ Drift score normalizes
         │ └─ Alert resolves
```

## Port Summary

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Backend | 8000 | HTTP | YOLO + VQA API & Metrics |
| Frontend | 8501 | HTTP | Streamlit UI (YOLO) |
| Prometheus | 9090 | HTTP | Metrics & Queries |
| Alertmanager | 9093 | HTTP | Alert Management |
| Grafana | 3000 | HTTP | Dashboards (YOLO + VQA) |
| Loki | 3100 | HTTP | Log Aggregation |

## Network Topology

```plaintext
┌─────────────────────────────────────────────────────────┐
│                   aivn-network (Docker)                 │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ Backend  │◄───┤Prometheus│◄───┤ Grafana  │        │
│  │  :8000   │    │  :9090   │    │  :3000   │        │
│  │          │    └────┬─────┘    └──────────┘        │
│  │ YOLO API │         │                                │
│  │ VQA API  │         ├────► Alertmanager :9093        │
│  │ Metrics  │         │                                │
│  └─────┬────┘         │                                │
│        │              │                                │
│        └──────────────┴────► Loki :3100                │
│                                                         │
│  All containers can communicate via service names      │
│  (e.g., http://prometheus:9090)                        │
└─────────────────────────────────────────────────────────┘
         │
         ↓ Exposed ports
┌─────────────────────────────────────────────────────────┐
│                    Host Machine                         │
│  localhost:3000  → Grafana (YOLO + VQA dashboards)     │
│  localhost:9090  → Prometheus                          │
│  localhost:9093  → Alertmanager                        │
│  localhost:8000  → Backend API (/v1/yolo, /v1/vqa)    │
│  localhost:8501  → Frontend (YOLO Streamlit)           │
└─────────────────────────────────────────────────────────┘
```

## Service Endpoint Summary

### YOLO Endpoints
- `POST /v1/yolo/detect` - Object detection
- `GET /v1/yolo/drift/status` - Drift status
- `POST /v1/yolo/drift/reset-reference` - Reset baseline
- `GET /v1/yolo/model/info` - Model information

### VQA Endpoints
- `POST /v1/vqa/answer` - Visual question answering
- `GET /v1/vqa/drift/status` - Drift status
- `POST /v1/vqa/drift/reset-reference` - Reset baseline
- `GET /v1/vqa/model/info` - Model information

### Common Endpoints
- `GET /metrics` - Prometheus metrics (both services)
- `GET /health` - Health check
- `GET /` - API documentation
