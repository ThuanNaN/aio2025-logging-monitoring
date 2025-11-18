# Quick Reference: YOLO & VQA Monitoring & Alerting

Quick reference for monitoring both **YOLO Object Detection** and **VQA (Visual Question Answering)** services with Evidently, Prometheus, and Grafana.

## 🚀 Start/Stop Services

```bash
# Start backend (YOLO + VQA)
cd backend
docker compose up -d --build

# Start monitoring stack
cd platform/monitor
docker compose up -d

# Stop monitoring stack
docker compose down

# Stop backend
cd backend
docker compose down

# Restart specific service
cd platform/monitor
docker compose restart prometheus
docker compose restart grafana
docker compose restart alertmanager

# View logs
docker compose logs -f prometheus
docker compose logs -f grafana
docker compose logs -f alertmanager

# View backend logs
cd backend
docker compose logs -f
```

## 📊 Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
  - YOLO Dashboard: ML Monitoring Dashboards → YOLO ML Model Monitoring
  - VQA Dashboard: ML Monitoring Dashboards → VQA BLIP Model Monitoring
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Backend API**: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - Metrics: http://localhost:8000/metrics

## 🔍 Check Metrics

```bash
# View all available metrics
curl http://localhost:8000/metrics

# Check YOLO Evidently metrics
curl http://localhost:8000/metrics | grep "evidently_"

# Check VQA Evidently metrics
curl http://localhost:8000/metrics | grep "vqa_evidently_"

# Check all VQA metrics
curl http://localhost:8000/metrics | grep "vqa_"

# Query Prometheus directly - YOLO
curl 'http://localhost:9090/api/v1/query?query=evidently_dataset_drift'
curl 'http://localhost:9090/api/v1/query?query=evidently_drift_share'

# Query Prometheus directly - VQA
curl 'http://localhost:9090/api/v1/query?query=vqa_evidently_dataset_drift'
curl 'http://localhost:9090/api/v1/query?query=vqa_evidently_drift_share'
```

## 🚨 Alert Management

```bash
# View all alert rules
curl http://localhost:9090/api/v1/rules

# Check active alerts
curl http://localhost:9090/api/v1/alerts

# Reload Prometheus config (after editing alert_rules.yml)
curl -X POST http://localhost:9090/-/reload

# Silence an alert in Alertmanager
curl -X POST http://localhost:9093/api/v2/silences \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [{"name":"alertname","value":"EvidentlyDatasetDriftDetected","isRegex":false}],
    "startsAt":"2025-01-01T00:00:00Z",
    "endsAt":"2025-01-01T12:00:00Z",
    "comment":"Investigating drift",
    "createdBy":"admin"
  }'
```

## 🔬 Drift Detection API

### YOLO Object Detection

```bash
# Get YOLO drift status
curl http://localhost:8000/v1/yolo/drift/status | jq

# Get YOLO drift summary
curl http://localhost:8000/v1/yolo/drift/summary | jq

# Reset YOLO reference data
curl -X POST http://localhost:8000/v1/yolo/drift/reset-reference | jq

# Get YOLO data quality report
curl http://localhost:8000/v1/yolo/data-quality | jq

# Get YOLO model info
curl http://localhost:8000/v1/yolo/model/info | jq

# YOLO health check
curl http://localhost:8000/v1/yolo/health | jq
```

### VQA (Visual Question Answering)

```bash
# Get VQA drift status
curl http://localhost:8000/v1/vqa/drift/status | jq

# Get VQA drift summary
curl http://localhost:8000/v1/vqa/drift/summary | jq

# Reset VQA reference data
curl -X POST http://localhost:8000/v1/vqa/drift/reset-reference | jq

# Get VQA data quality report
curl http://localhost:8000/v1/vqa/data-quality | jq

# Get VQA model info
curl http://localhost:8000/v1/vqa/model/info | jq

# VQA health check
curl http://localhost:8000/v1/vqa/health | jq
```

## 🧪 Testing

### YOLO Object Detection

```bash
# Send single YOLO inference request
curl -X POST "http://localhost:8000/v1/yolo/detect/" \
  -F "file=@image.jpg" | jq

# Send multiple YOLO requests (load test)
for i in {1..200}; do
  curl -X POST "http://localhost:8000/v1/yolo/detect/" \
    -F "file=@image.jpg"
  sleep 0.5
done

# Check YOLO metrics are being scraped
curl http://localhost:9090/api/v1/targets | \
  jq '.data.activeTargets[] | select(.job=="yolo_backend")'
```

### VQA (Visual Question Answering)

```bash
# Send single VQA request
curl -X POST "http://localhost:8000/v1/vqa/answer" \
  -F "image=@image.jpg" \
  -F "question=What is in this image?" \
  -F "max_length=50" \
  -F "num_beams=5" | jq

# Send multiple VQA requests (load test)
for i in {1..200}; do
  curl -X POST "http://localhost:8000/v1/vqa/answer" \
    -F "image=@image.jpg" \
    -F "question=What is in this image?" \
    -F "max_length=50" \
    -F "num_beams=5"
  sleep 1
done

# Test different question types
for question in "What is this?" "Where is the object?" "How many people?" "Who is this?"; do
  curl -X POST "http://localhost:8000/v1/vqa/answer" \
    -F "image=@image.jpg" \
    -F "question=$question" | jq '.answer'
done
```

### General Health Checks

```bash
# Check backend is healthy
curl http://localhost:8000/health | jq

# Check all services
curl http://localhost:8000/v1/yolo/health | jq
curl http://localhost:8000/v1/vqa/health | jq
```

## 📈 Useful Prometheus Queries

### YOLO Metrics

```promql
# Drift detection
evidently_dataset_drift
evidently_drift_share
evidently_num_drifted_features

# Feature drift scores
evidently_brightness_drift_score
evidently_confidence_drift_score
evidently_detections_drift_score

# Performance metrics
rate(inference_count_total[5m])
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))

# System metrics
process_vram_memory_GB
drift_image_brightness
drift_embedding_distance
```

### VQA Metrics

```promql
# VQA drift detection
vqa_evidently_dataset_drift
vqa_evidently_drift_share
vqa_evidently_num_drifted_features

# VQA feature drift scores
vqa_evidently_brightness_drift_score
vqa_evidently_question_length_drift_score
vqa_evidently_answer_length_drift_score
vqa_evidently_inference_time_drift_score

# VQA performance
rate(vqa_inference_count_total[5m])
histogram_quantile(0.95, rate(vqa_inference_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(vqa_inference_latency_seconds_bucket[5m]))

# VQA question metrics
vqa_question_length
vqa_answer_length
rate(vqa_question_type_total[5m])

# Question type distribution
sum by (question_type) (vqa_question_type_total)
```

### Combined Queries

```promql
# Service health
up{job="yolo_backend"}

# Total inference rate (YOLO + VQA)
rate(inference_count_total[5m]) + rate(vqa_inference_count_total[5m])

# Compare drift between services
evidently_drift_share
vqa_evidently_drift_share

# GPU memory (shared)
process_vram_memory_GB
```

## 🛠️ Maintenance

```bash
# Backup Prometheus data
docker run --rm \
  -v monitor_prometheus-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/prometheus-backup.tar.gz /data

# Backup Grafana dashboards
docker run --rm \
  -v monitor_grafana-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/grafana-backup.tar.gz /data

# Clean Docker volumes (WARNING: deletes all data)
docker compose down -v

# View disk usage
docker system df
du -sh platform/monitor/*
```

## 🐛 Troubleshooting

```bash
# Check backend is healthy
curl http://localhost:8000/health | jq

# Check individual services
curl http://localhost:8000/v1/yolo/health | jq
curl http://localhost:8000/v1/vqa/health | jq

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Test Prometheus query
curl 'http://localhost:9090/api/v1/query?query=up' | jq

# Check YOLO drift detector status
curl http://localhost:8000/v1/yolo/drift/summary | jq '.detector_stats'

# Check VQA drift detector status
curl http://localhost:8000/v1/vqa/drift/summary | jq '.detector_stats'

# Check Grafana datasources
docker compose exec grafana grafana-cli plugins ls

# Verify alert rules are valid
docker compose exec prometheus promtool check rules /etc/prometheus/alert_rules.yml

# Check Alertmanager config
docker compose exec alertmanager amtool check-config /etc/alertmanager/alertmanager.yml

# Check if models are loaded
curl http://localhost:8000/v1/yolo/model/info | jq
curl http://localhost:8000/v1/vqa/model/info | jq

# View backend container logs
cd backend
docker compose logs --tail=100 -f

# Check GPU availability
curl http://localhost:8000/metrics | grep vram
```

## 📝 Common Tasks

### Update Alert Thresholds

1. Edit `platform/monitor/prometheus/alert_rules.yml`
2. Change the `expr` condition:
   - YOLO: `evidently_drift_share > 0.5`
   - VQA: `vqa_evidently_drift_share > 0.5`
3. Reload Prometheus: `curl -X POST http://localhost:9090/-/reload`

### Add New Dashboard Panel

1. Login to Grafana (http://localhost:3000)
2. Open YOLO or VQA dashboard → Add → Visualization
3. Select Prometheus datasource
4. Enter query:
   - YOLO: `evidently_dataset_drift`
   - VQA: `vqa_evidently_dataset_drift`
5. Configure visualization
6. Save dashboard

### Export Dashboard

1. Open dashboard in Grafana
2. Click **Share** → **Export** → **Save to file**
3. Save to `platform/monitor/grafana/`
   - YOLO: `yolo-evidently-dashboard.json`
   - VQA: `vqa-evidently-dashboard.json`

### Import Dashboard

```bash
# Via UI
Grafana → Dashboards → Import → Upload JSON file

# Via API - YOLO
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @platform/monitor/grafana/yolo-evidently-dashboard.json

# Via API - VQA
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @platform/monitor/grafana/vqa-evidently-dashboard.json
```

### Reset Drift Detection

```bash
# Reset after model update or data distribution change
curl -X POST http://localhost:8000/v1/yolo/drift/reset-reference | jq
curl -X POST http://localhost:8000/v1/vqa/drift/reset-reference | jq

# Verify reset
curl http://localhost:8000/v1/yolo/drift/summary | jq '.detector_stats'
curl http://localhost:8000/v1/vqa/drift/summary | jq '.detector_stats'
```

## 🔔 Alert Notification Setup

### Slack

Edit `platform/monitor/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'drift-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#ml-alerts'
        title: 'Drift Alert: {{ .GroupLabels.alertname }}'
```

### Email

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
        headers:
          Subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
```

Restart Alertmanager after changes:
```bash
docker compose restart alertmanager
```

## 📊 Grafana Variables

Create dashboard variables for dynamic filtering:

1. Dashboard Settings → Variables → Add variable
2. Name: `time_range`
3. Type: Interval
4. Values: `5m,15m,1h,6h,24h`
5. Use in queries: `rate(inference_count_total[$time_range])`

## 🎯 Performance Optimization

```bash
# Increase Prometheus retention
# Edit docker-compose.yml, add to prometheus command:
- "--storage.tsdb.retention.time=30d"
- "--storage.tsdb.retention.size=10GB"

# Reduce scrape interval for high-volume metrics
# Edit prometheus.yml:
scrape_interval: 5s  # default is 15s

# Optimize Grafana queries
# Use recording rules in Prometheus for complex queries
# Edit alert_rules.yml, add:
- record: job:inference_rate:5m
  expr: rate(inference_count_total[5m])

- record: job:vqa_inference_rate:5m
  expr: rate(vqa_inference_count_total[5m])
```

## 🎨 VQA-Specific Quick Tips

```bash
# Check question type distribution
curl http://localhost:8000/v1/vqa/drift/summary | jq '.detector_stats.question_type_distribution'

# Monitor average question/answer lengths
curl http://localhost:8000/v1/vqa/drift/summary | jq '.detector_stats | {avg_question_length, avg_answer_length, avg_inference_time}'

# Test different question types
for qtype in "What" "Where" "How many" "Who" "Why"; do
  echo "Testing: $qtype"
  curl -s -X POST "http://localhost:8000/v1/vqa/answer" \
    -F "image=@image.jpg" \
    -F "question=$qtype is in the image?" | jq '.answer'
done

# Check VQA feature drift details
curl 'http://localhost:9090/api/v1/query?query=vqa_evidently_question_length_drift_score' | jq
curl 'http://localhost:9090/api/v1/query?query=vqa_evidently_answer_length_drift_score' | jq
```

## 📚 Quick Links

- **API Documentation**: http://localhost:8000/docs
- **YOLO Endpoints**: http://localhost:8000/docs#/YOLO
- **VQA Endpoints**: http://localhost:8000/docs#/VQA
- **Metrics**: http://localhost:8000/metrics
- **Grafana YOLO**: http://localhost:3000/d/yolo-evidently
- **Grafana VQA**: http://localhost:3000/d/vqa-evidently
- **Prometheus**: http://localhost:9090/graph
- **Alertmanager**: http://localhost:9093/#/alerts

## 💡 Pro Tips

1. **Baseline Collection**: Send 100+ diverse requests before expecting drift detection
2. **Question Variety**: For VQA, use different question types to establish good baseline
3. **Alert Tuning**: Start with high thresholds, lower gradually based on false positives
4. **Reference Updates**: Reset reference after model updates or expected distribution changes
5. **Dashboard Refresh**: Set to 10s-30s for real-time monitoring, 5m for historical analysis
6. **Metrics Retention**: Keep 15-30 days for trend analysis
7. **Load Testing**: Use gradual ramp-up to avoid overwhelming the system
8. **GPU Monitoring**: Watch VRAM usage, especially when running both YOLO and VQA
