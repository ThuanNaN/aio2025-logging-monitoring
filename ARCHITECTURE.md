# Monitoring Architecture Diagram

## Complete System Flow

```plaintext
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLIENT REQUESTS                                 │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ↓
                   ┌─────────────────────────┐
                   │   Frontend (Streamlit)  │
                   │   Port: 8501            │
                   └────────────┬────────────┘
                                │
                                ↓ HTTP POST /v1/yolo/detect/
                   ┌─────────────────────────┐
                   │   YOLO Backend (FastAPI)│
                   │   Port: 8000            │
                   │                         │
                   │  ┌──────────────────┐   │
                   │  │ YOLO Detection   │   │
                   │  └────────┬─────────┘   │
                   │           │             │
                   │  ┌────────▼─────────┐   │
                   │  │ Evidently        │   │
                   │  │ Drift Detector   │   │
                   │  │                  │   │
                   │  │ • Brightness     │   │
                   │  │ • Confidence     │   │
                   │  │ • Detections     │   │
                   │  │ • Embeddings     │   │
                   │  └────────┬─────────┘   │
                   │           │             │
                   │  ┌────────▼─────────┐   │
                   │  │ Metrics Export   │   │
                   │  │ /metrics         │   │
                   │  └──────────────────┘   │
                   └────────────┬────────────┘
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
   │  └──────┬───────┘  │         │  └──────────────┘  │
   │         │          │         │                    │
   │  ┌──────▼───────┐  │         │  ┌──────────────┐  │
   │  │ Receivers    │  │         │  │  Alerts View │  │
   │  │ • Slack      │  │         │  └──────────────┘  │
   │  │ • Email      │  │         │                    │
   │  │ • PagerDuty  │  │         │  ┌──────────────┐  │
   │  │ • Webhook    │  │         │  │  Queries     │  │
   │  └──────────────┘  │         │  │  (PromQL)    │  │
   └────────────────────┘         │  └──────────────┘  │
                                  └────────────────────┘
                                            │
                                            ↓
                                  ┌────────────────────┐
                                  │   Loki (Logs)      │
                                  │   Port: 3100       │
                                  └────────────────────┘
```

## Metrics Flow

```plaintext
YOLO Inference Request
    │
    ├─► Process Image
    │       │
    │       ├─► Extract Features (brightness, detections, confidence)
    │       │
    │       └─► Extract Embeddings
    │
    ├─► Evidently Drift Detection
    │       │
    │       ├─► Add Sample to Window
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
    └─► Update Prometheus Metrics
            │
            ├─► evidently_dataset_drift = 0 or 1
            ├─► evidently_drift_share = 0.XX
            ├─► evidently_num_drifted_features = N
            ├─► evidently_brightness_drift_score = 0.XX
            ├─► evidently_confidence_drift_score = 0.XX
            ├─► evidently_detections_drift_score = 0.XX
            ├─► inference_count_total += 1
            ├─► inference_latency_seconds.observe(duration)
            ├─► drift_image_brightness = XX.XX
            ├─► drift_embedding_distance = 0.XX
            └─► process_vram_memory_GB = X.XX
                    │
                    ↓
            Prometheus Scrapes /metrics
                    │
                    ↓
            Evaluate Alert Rules
                    │
                    ├─► If drift_share > 0.3 → Fire Alert
                    ├─► If latency > 2s → Fire Alert
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
│  • Inference Latency > 5s                                   │
│  • Drift Share > 50%                                        │
│                                                             │
│  Action: Immediate investigation required                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        WARNING                               │
│  • Dataset Drift Detected                                   │
│  • Drift Share > 30%                                        │
│  • High Feature Drift (> 0.7)                               │
│  • Inference Latency > 2s                                   │
│  • GPU Memory > 10GB                                        │
│  • Low Inference Rate                                       │
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

## Data Flow Timeline

```plaintext
t=0s     │ Backend starts, Evidently detector initialized
         │ Reference window = empty (needs 100 samples)
         │
t=15s    │ Prometheus first scrape (no metrics yet)
         │
t=30s    │ First inference request arrives
         │ ├─ YOLO detection
         │ ├─ Evidently adds sample (1/100)
         │ └─ Metrics updated
         │
t=45s    │ Prometheus scrapes metrics
         │ └─ Stores time series data
         │
...      │ More requests processed...
         │
t=300s   │ 100th request processed
         │ ├─ Reference baseline established!
         │ └─ Drift detection now active
         │
t=315s   │ 101st-150th requests
         │ ├─ Current window = 50 samples
         │ ├─ Evidently compares with reference
         │ └─ Drift scores calculated
         │
t=330s   │ Prometheus evaluates alert rules
         │ ├─ drift_share = 0.35 (> 0.3 threshold)
         │ └─ Alert: EvidentlyHighDriftShare (pending)
         │
t=510s   │ Alert fires (after 3m duration)
         │ ├─ Sent to Alertmanager
         │ ├─ Routed to "drift-alerts" receiver
         │ └─ Notification sent (Slack/Email)
         │
t=600s   │ User checks Grafana dashboard
         │ ├─ Sees drift indicators
         │ ├─ Reviews feature-level scores
         │ └─ Investigates root cause
         │
t=900s   │ User resets reference data
         │ curl -X POST /v1/monitoring/drift/reset-reference
         │ └─ New baseline established
         │
t=1200s  │ Drift score normalizes
         │ └─ Alert resolves
```

## Port Summary

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Backend | 8000 | HTTP | API & Metrics |
| Frontend | 8501 | HTTP | Streamlit UI |
| Prometheus | 9090 | HTTP | Metrics & Queries |
| Alertmanager | 9093 | HTTP | Alert Management |
| Grafana | 3000 | HTTP | Dashboards |
| Loki | 3100 | HTTP | Log Aggregation |

## Network Topology

```plaintext
┌─────────────────────────────────────────────────────────┐
│                   aivn-network (Docker)                 │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ Backend  │◄───┤Prometheus│◄───┤ Grafana  │        │
│  │  :8000   │    │  :9090   │    │  :3000   │        │
│  └─────┬────┘    └────┬─────┘    └──────────┘        │
│        │              │                                │
│        │              ├────► Alertmanager :9093        │
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
│  localhost:3000  → Grafana                             │
│  localhost:9090  → Prometheus                          │
│  localhost:9093  → Alertmanager                        │
│  localhost:8000  → Backend API                         │
│  localhost:8501  → Frontend                            │
└─────────────────────────────────────────────────────────┘
```
