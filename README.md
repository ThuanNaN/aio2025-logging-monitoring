# ML Model Monitoring with Evidently, Prometheus & Grafana

## 1. Introduction

This project implements comprehensive monitoring and drift detection for two AI services:
- **YOLO Object Detection**: Real-time object detection with visual drift monitoring
- **VQA (BLIP)**: Visual Question Answering with multi-modal drift detection

### Technology Stack

- **Evidently**: Advanced statistical drift detection for both services
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Interactive visualization dashboards (separate dashboards for YOLO and VQA)
- **Alertmanager**: Alert routing and notifications
- **Loki**: Log aggregation
- **FastAPI**: Backend API server
- **Streamlit**: Frontend UI (optional)

## 2. Features

### YOLO Object Detection

✅ Real-time object detection with YOLO11x  
✅ Visual feature drift detection (brightness, confidence, detections)  
✅ Embedding-based drift monitoring (cosine similarity)  
✅ Evidently statistical drift analysis  
✅ Performance monitoring (latency, throughput)  
✅ Automated image capture and annotation  

### VQA (Visual Question Answering)

✅ Natural language question answering using BLIP  
✅ Multi-modal drift detection (visual + textual)  
✅ Question pattern analysis (type, length, complexity)  
✅ Answer quality monitoring  
✅ Inference time tracking  
✅ Question type classification (what, where, who, how_many, etc.)  

### Monitoring & Alerting

✅ Dataset-level and feature-level drift detection  
✅ Automated alerting with configurable thresholds  
✅ Interactive Grafana dashboards (10 panels each)  
✅ GPU resource monitoring (shared across services)  
✅ RESTful API endpoints for drift analysis  
✅ Reference dataset management  
✅ Data quality reports  

## 3. Quick Start

### Prerequisites

```bash
# Install Loki Docker driver
docker plugin install grafana/loki-docker-driver:3.3.2-arm64 --alias loki --grant-all-permissions

# Create network
docker network create aivn-network
```

### Start Services

```bash
# 1. Start monitoring stack
cd platform/monitor
docker compose up -d

# 2. Start backend
cd ../../backend
docker compose up -d

# 3. Start frontend (optional)
cd ../frontend
docker compose up -d
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Backend API**: http://localhost:8000/docs

## 4. Architecture

```
┌─────────────────────────────────────────────────┐
│         FastAPI Backend (Port 8000)             │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐         ┌──────────────┐     │
│  │ YOLO Service │         │  VQA Service │     │
│  │  /v1/yolo/*  │         │  /v1/vqa/*   │     │
│  └──────┬───────┘         └──────┬───────┘     │
│         │                        │             │
│         ↓                        ↓             │
│  ┌──────────────────────────────────────┐      │
│  │   Evidently Drift Detectors          │      │
│  │   • YOLO: brightness, confidence,    │      │
│  │     detections, embeddings           │      │
│  │   • VQA: brightness, contrast,       │      │
│  │     question/answer patterns         │      │
│  └──────────────┬───────────────────────┘      │
│                 │                              │
│                 ↓                              │
│       /metrics (Prometheus format)             │
└─────────────────┬──────────────────────────────┘
                  │
                  ↓ (scrapes every 15s)
          ┌───────────────┐
          │  Prometheus   │  Collects metrics
          │   Port 9090   │  Evaluates alert rules
          └───────┬───────┘
                  │
        ┌─────────┴──────────┐
        │                    │
        ↓                    ↓
  ┌─────────────┐    ┌──────────────┐
  │Alertmanager │    │   Grafana    │
  │  Port 9093  │    │  Port 3000   │
  │             │    │              │
  │ Routes      │    │ • YOLO       │
  │ alerts      │    │   Dashboard  │
  └─────────────┘    │ • VQA        │
                     │   Dashboard  │
                     └──────┬───────┘
                            │
                            ↓
                     ┌──────────────┐
                     │     Loki     │
                     │  Port 3100   │
                     │ Log          │
                     │ Aggregation  │
                     └──────────────┘
```

## 5. Monitoring Metrics

### YOLO Drift Metrics

- `evidently_dataset_drift`: Dataset-level drift (0/1)
- `evidently_drift_share`: Proportion of drifted features (0.0-1.0)
- `evidently_num_drifted_features`: Count of drifted features
- `evidently_brightness_drift_score`: Brightness drift score
- `evidently_confidence_drift_score`: Confidence drift score
- `evidently_detections_drift_score`: Detection count drift score

### VQA Drift Metrics

- `vqa_evidently_dataset_drift`: VQA dataset-level drift (0/1)
- `vqa_evidently_drift_share`: Proportion of VQA features drifting
- `vqa_evidently_num_drifted_features`: Count of drifted VQA features
- `vqa_evidently_brightness_drift_score`: Visual brightness drift
- `vqa_evidently_question_length_drift_score`: Question length drift
- `vqa_evidently_answer_length_drift_score`: Answer length drift
- `vqa_evidently_inference_time_drift_score`: Performance drift

### YOLO Performance Metrics

- `inference_count_total`: Total YOLO inference requests
- `inference_latency_seconds`: YOLO request latency histogram
- `drift_image_brightness`: Image brightness values
- `drift_embedding_distance`: Cosine similarity drift

### VQA Performance Metrics

- `vqa_inference_count_total`: Total VQA requests
- `vqa_inference_latency_seconds`: VQA latency histogram
- `vqa_question_length`: Question length distribution
- `vqa_answer_length`: Answer length distribution
- `vqa_question_type_total`: Question type counters

### System Metrics (Shared)

- `process_vram_memory_GB`: GPU memory usage
- `up`: Service availability

## 6. API Endpoints

### YOLO Object Detection

**Detection:**
```bash
POST /v1/yolo/detect/
# Upload image for object detection with drift analysis
```

**Monitoring:**
```bash
GET /v1/yolo/drift/status        # Current drift status
GET /v1/yolo/drift/summary       # Detailed drift summary
POST /v1/yolo/drift/reset-reference  # Reset baseline
GET /v1/yolo/data-quality        # Data quality metrics
GET /v1/yolo/model/info          # Model information
GET /v1/yolo/health              # Health check
```

### VQA (Visual Question Answering)

**Question Answering:**
```bash
POST /v1/vqa/answer
# Upload image and question for answer generation
# Parameters: image, question, max_length, num_beams
```

**Monitoring:**
```bash
GET /v1/vqa/drift/status         # Current VQA drift status
GET /v1/vqa/drift/summary        # Detailed VQA drift summary
POST /v1/vqa/drift/reset-reference   # Reset VQA baseline
GET /v1/vqa/data-quality         # VQA data quality metrics
GET /v1/vqa/model/info           # BLIP model information
GET /v1/vqa/health               # VQA health check
```

### General

```bash
GET /metrics                     # Prometheus metrics endpoint
GET /docs                        # OpenAPI documentation
GET /health                      # Overall health check
```

## 7. Alerting

### YOLO Alerts

- Dataset drift detection
- High drift share (>30%, >50%)
- Feature-specific drift (brightness, confidence, detections)
- High inference latency (p95 > 2s, p99 > 5s)
- Low inference rate
- Abnormal image brightness

### VQA Alerts

- VQA dataset drift detection
- High VQA drift share
- Question pattern drift (length, type distribution)
- Answer quality drift (length changes)
- Performance degradation (inference time)
- Low VQA request rate

### System Alerts

- GPU memory warnings (>10GB)
- Service availability (backend down)
- High system resource usage

See [MONITORING_SETUP_GUIDE.md](./MONITORING_SETUP_GUIDE.md) for complete alert configuration.

## 8. Documentation

- **[EVIDENTLY_README.md](./EVIDENTLY_README.md)**: Evidently integration for YOLO & VQA
- **[VQA_README.md](./VQA_README.md)**: VQA implementation and usage guide
- **[MONITORING_SETUP_GUIDE.md](./MONITORING_SETUP_GUIDE.md)**: Complete monitoring setup
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)**: Common commands and queries
- **[ARCHITECTURE.md](./ARCHITECTURE.md)**: System architecture details
- **[Platform README](./platform/README.md)**: Platform-specific documentation

## 9. Testing

### Quick Test - YOLO

```bash
# Send YOLO test request
curl -X POST "http://localhost:8000/v1/yolo/detect/" \
  -F "file=@path/to/image.jpg"

# Check YOLO drift status
curl http://localhost:8000/v1/yolo/drift/status | jq

# View YOLO metrics
curl http://localhost:8000/metrics | grep "evidently_"
```

### Quick Test - VQA

```bash
# Send VQA test request
curl -X POST "http://localhost:8000/v1/vqa/answer" \
  -F "image=@path/to/image.jpg" \
  -F "question=What is in this image?"

# Check VQA drift status
curl http://localhost:8000/v1/vqa/drift/status | jq

# View VQA metrics
curl http://localhost:8000/metrics | grep "vqa_"
```

### Testing Scripts

**1. test_api.py** - YOLO testing client
```bash
# Single image detection
python test_api.py --image path/to/image.jpg

# Batch processing
python test_api.py --dir path/to/images --delay 1.0

# Load test (establish baseline + drift detection)
python test_api.py --image test.jpg --repeat 200 --delay 0.5

# Check drift status
python test_api.py --drift-status

# Reset reference
python test_api.py --reset-reference
```

**2. test_vqa_api.py** - VQA testing client
```bash
# Single question
python test_vqa_api.py --image image.jpg --question "What is this?"

# Interactive mode
python test_vqa_api.py --interactive

# Load test with multiple questions
python test_vqa_api.py --image test.jpg --repeat 200

# Different question types
python test_vqa_api.py --image img.jpg \
  --question "What is in this image?" \
  --question "Where is the object?" \
  --question "How many people?"

# Check VQA drift
python test_vqa_api.py --drift-status
```

**3. test_api.sh** - Bash script for quick tests
```bash
chmod +x test_api.sh

./test_api.sh --health
./test_api.sh -i image.jpg
./test_api.sh -d images/ --delay 1.0
```

**4. Comprehensive Test Suite**
```bash
# Run full test suite for both services
./run_comprehensive_tests.sh

# Results saved to test_results/
```

See [VQA_README.md](./VQA_README.md) for detailed VQA testing instructions.

## 10. Grafana Dashboards

### YOLO Dashboard (10 Panels)

1. Dataset Drift Status (stat panel)
2. Drift Share (gauge)
3. Number of Drifted Features (time series)
4. Inference Rate (time series)
5. Feature-Level Drift Scores (time series)
6. Inference Latency Percentiles (time series)
7. Image Brightness (time series)
8. Embedding Drift (time series)
9. GPU Memory Usage (time series)
10. Active Alerts (table)

**Import from:** `platform/monitor/grafana/yolo-evidently-dashboard.json`

### VQA Dashboard (10 Panels)

1. VQA Dataset Drift Status (stat panel)
2. VQA Drift Share (gauge)
3. VQA Drifted Features Count (time series)
4. VQA Inference Rate (time series)
5. VQA Feature-Level Drift Scores (time series)
6. VQA Inference Latency (time series)
7. Question Type Distribution (bar/time series)
8. Question Length Trend (time series)
9. Answer Length Trend (time series)
10. VQA Active Alerts (table)

**Import from:** `platform/monitor/grafana/vqa-evidently-dashboard.json`

Both dashboards are auto-provisioned when Grafana starts.

## 11. Troubleshooting

### Metrics not appearing
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify backend metrics endpoint
curl http://localhost:8000/metrics

# Check specific service metrics
curl http://localhost:8000/metrics | grep "evidently_"  # YOLO
curl http://localhost:8000/metrics | grep "vqa_"       # VQA
```

### Drift not detecting

**For YOLO:**
- Need 100+ samples for reference baseline
- Need 50+ additional samples for detection window
- Check: `curl http://localhost:8000/v1/yolo/drift/summary | jq`

**For VQA:**
- Need 100+ diverse questions for reference
- Include variety of question types
- Check: `curl http://localhost:8000/v1/vqa/drift/summary | jq`

### Service health checks
```bash
# Check overall backend health
curl http://localhost:8000/health | jq

# Check individual services
curl http://localhost:8000/v1/yolo/health | jq
curl http://localhost:8000/v1/vqa/health | jq

# Check model loading
curl http://localhost:8000/v1/yolo/model/info | jq
curl http://localhost:8000/v1/vqa/model/info | jq
```

### Alerts not firing
```bash
# Check alert rules loaded
docker compose -f platform/monitor/docker-compose.yml logs prometheus | grep alert

# View alerts in Prometheus
# Go to http://localhost:9090/alerts

# Check Alertmanager
curl http://localhost:9093/api/v2/alerts | jq
```

### GPU/Memory Issues
```bash
# Check GPU availability
curl http://localhost:8000/metrics | grep vram

# Monitor GPU usage
watch -n 1 'curl -s http://localhost:8000/metrics | grep vram'
```

## 12. Production Recommendations

1. **Secure Grafana**: Change default admin password
2. **Configure Notifications**: Add Slack/email alerts in Alertmanager
3. **Set Retention**: Configure Prometheus data retention
4. **Backup Dashboards**: Export and version control dashboard JSON
5. **Monitor Resources**: Set up alerts for Prometheus/Grafana resource usage
6. **Regular Reviews**: Review drift trends weekly, update thresholds

## 13. Test Cases

### System Tests

- Stop backend container manually → Service down alert
- Missing packages → Import errors in logs
- Missing environment variables → Configuration errors
- Restart services → Recovery and reconnection

### YOLO Drift Tests

- Send images with varying brightness → Brightness drift alert
- Send images with different object counts → Detection drift
- Send images with low confidence detections → Confidence drift
- Send burst of requests → Performance metrics spike
- Stop sending requests → Low inference rate alert

### VQA Drift Tests

- Send only "what" questions → Question type distribution drift
- Send very long questions → Question length drift
- Send very short questions → Question length drift  
- Send complex images → Inference time drift
- Change image characteristics → Visual feature drift
- Vary question patterns → Question pattern drift

### Performance Tests

- Concurrent YOLO and VQA requests → Resource sharing
- High-frequency requests → Throughput limits
- Large images → Latency increase
- Memory-intensive operations → GPU memory alerts

### Recovery Tests

- Prometheus restart → Metric continuity
- Grafana restart → Dashboard persistence
- Backend restart → Service recovery and metric reset
- Reference data reset → New baseline establishment

## 14. Key Differences: YOLO vs VQA Monitoring

| Aspect | YOLO | VQA |
|--------|------|-----|
| **Primary Features** | Visual (brightness, detections, confidence) | Multi-modal (visual + textual) |
| **Drift Detection** | Image and detection patterns | Question/answer patterns + visual |
| **Performance Metrics** | Detection latency, throughput | Inference time, question complexity |
| **Categorical Features** | None | Question types (what, where, who, etc.) |
| **Use Case** | Object detection drift | Question understanding drift |
| **Baseline Requirements** | 100+ diverse images | 100+ diverse questions and images |
| **Alert Focus** | Visual quality, detection accuracy | Question patterns, answer quality |

## 15. Related Projects

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [BLIP Documentation](https://huggingface.co/Salesforce/blip-vqa-base)
