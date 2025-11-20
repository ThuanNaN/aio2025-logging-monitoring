# Testing Guide for ML API Drift Detection

## Overview
This guide provides comprehensive testing scenarios for YOLO and VQA services with drift detection capabilities.

## Image Dataset Structure

```
images/
├── baseline/                    # Baseline reference data
│   ├── vqa/                    # VQA baseline (300 images)
│   │   ├── general/            # 100 general images
│   │   ├── objects/            # 100 object-focused images
│   │   └── people/             # 100 people-focused images
│   └── yolo/                   # YOLO baseline (300 images)
│       ├── indoor/             # 100 indoor scenes
│       ├── normal/             # 100 normal scenes
│       └── outdoor/            # 100 outdoor scenes
│
├── drift_scenarios/            # Drift testing scenarios
│   ├── vqa/                    # VQA drift tests (250 images)
│   │   ├── brightness/         # 100 brightness variations
│   │   │   ├── bright/         # 50 bright images
│   │   │   └── dark/           # 50 dark images
│   │   └── complexity/         # 150 complexity variations
│   │       ├── abstract/       # 50 abstract images
│   │       ├── complex/        # 50 complex scenes
│   │       └── simple/         # 50 simple scenes
│   └── yolo/                   # YOLO drift tests (450 images)
│       ├── brightness/         # 150 brightness variations
│       │   ├── bright/         # 50 bright images
│       │   ├── dark/           # 50 dark images
│       │   └── high_contrast/  # 50 high contrast images
│       ├── confidence/         # 150 detection confidence tests
│       │   ├── low_quality/    # 50 low quality images
│       │   ├── occluded/       # 50 occluded objects
│       │   └── unusual_angles/ # 50 unusual angle views
│       └── object_density/     # 150 object density tests
│           ├── crowded/        # 50 crowded scenes
│           ├── few_objects/    # 50 sparse scenes
│           └── many_objects/   # 50 dense scenes
│
└── load_test/                  # Performance testing
    ├── identical/              # 10 identical copies
    └── similar/                # 20 similar images
```

## Testing Scenarios

### 1. Health Checks

```bash
# Check YOLO service health
python test_api.py --service yolo --health

# Check VQA service health
python test_api.py --service vqa --health
```

### 2. Baseline Establishment (Reference Data)

**YOLO Baseline:**
```bash
# Establish baseline with indoor scenes
python test_api.py --service yolo --dir images/baseline/yolo/indoor --delay 0.5

# Add outdoor scenes to baseline
python test_api.py --service yolo --dir images/baseline/yolo/outdoor --delay 0.5

# Add normal scenes to complete baseline
python test_api.py --service yolo --dir images/baseline/yolo/normal --delay 0.5
```

**VQA Baseline:**
```bash
# Establish baseline with general images
python test_api.py --service vqa --dir images/baseline/vqa/general --delay 0.5

# Add object-focused images
python test_api.py --service vqa --dir images/baseline/vqa/objects --delay 0.5

# Add people-focused images
python test_api.py --service vqa --dir images/baseline/vqa/people --delay 0.5
```

### 3. Drift Detection Tests

#### YOLO Drift Tests

**Brightness Drift:**
```bash
# Test with bright images (should trigger drift)
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/brightness/bright \
  --delay 0.5 --check-drift 10 --output results_yolo_bright.json

# Test with dark images
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/brightness/dark \
  --delay 0.5 --check-drift 10

# Test with high contrast images
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/brightness/high_contrast \
  --delay 0.5 --check-drift 10
```

**Detection Confidence Drift:**
```bash
# Test with low quality images (lower confidence)
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/confidence/low_quality \
  --delay 0.5 --check-drift 10

# Test with occluded objects
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/confidence/occluded \
  --delay 0.5 --check-drift 10

# Test with unusual angles
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/confidence/unusual_angles \
  --delay 0.5 --check-drift 10
```

**Object Density Drift:**
```bash
# Test with crowded scenes
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/object_density/crowded \
  --delay 0.5 --check-drift 10

# Test with sparse scenes
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/object_density/few_objects \
  --delay 0.5 --check-drift 10

# Test with dense scenes
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/object_density/many_objects \
  --delay 0.5 --check-drift 10
```

#### VQA Drift Tests

**Brightness Drift:**
```bash
# Test with bright images
python test_api.py --service vqa \
  --dir images/drift_scenarios/vqa/brightness/bright \
  --delay 0.5 --check-drift 10 --output results_vqa_bright.json

# Test with dark images
python test_api.py --service vqa \
  --dir images/drift_scenarios/vqa/brightness/dark \
  --delay 0.5 --check-drift 10
```

**Complexity Drift:**
```bash
# Test with abstract images
python test_api.py --service vqa \
  --dir images/drift_scenarios/vqa/complexity/abstract \
  --delay 0.5 --check-drift 10

# Test with complex scenes
python test_api.py --service vqa \
  --dir images/drift_scenarios/vqa/complexity/complex \
  --delay 0.5 --check-drift 10

# Test with simple scenes
python test_api.py --service vqa \
  --dir images/drift_scenarios/vqa/complexity/simple \
  --delay 0.5 --check-drift 10
```

### 4. Load Testing

**Identical Images (No drift expected):**
```bash
# YOLO load test with identical images
python test_api.py --service yolo \
  --dir images/load_test/identical \
  --delay 0.1 --check-drift 5 --output load_yolo_identical.json

# VQA load test with identical images
python test_api.py --service vqa \
  --dir images/load_test/identical \
  --delay 0.1 --check-drift 5 --output load_vqa_identical.json
```

**Similar Images (Minimal drift):**
```bash
# YOLO load test with similar images
python test_api.py --service yolo \
  --dir images/load_test/similar \
  --delay 0.1 --check-drift 5 --output load_yolo_similar.json

# VQA load test with similar images
python test_api.py --service vqa \
  --dir images/load_test/similar \
  --delay 0.1 --check-drift 5 --output load_vqa_similar.json
```

**Repeat Testing:**
```bash
# High volume test - 200 requests
python test_api.py --service yolo \
  --image images/load_test/identical/identical_001.jpg \
  --repeat 200 --delay 0.2 --check-drift 20

# Stress test - 500 requests
python test_api.py --service yolo \
  --image images/load_test/identical/identical_001.jpg \
  --repeat 500 --delay 0.1 --check-drift 50
```

### 5. Drift Status Monitoring

```bash
# Check current drift status
python test_api.py --service yolo --drift-status
python test_api.py --service vqa --drift-status

# Reset reference baseline
python test_api.py --service yolo --reset-reference
python test_api.py --service vqa --reset-reference
```

### 6. Custom Question Testing (VQA)

```bash
# Single custom question
python test_api.py --service vqa \
  --image images/baseline/vqa/general/image001.jpg \
  --question "What objects are in this image?"

# Multiple questions on same image
python test_api.py --service vqa \
  --image images/baseline/vqa/people/image001.jpg \
  --questions "How many people are there?" "What are they doing?" "Where is this?"

# Multiple questions on directory
python test_api.py --service vqa \
  --dir images/baseline/vqa/objects \
  --questions "What is the main object?" "What color is it?" "Where is it located?"
```

## Complete Testing Workflow

### Initial Setup
```bash
# 1. Start services
cd platform/monitor && docker-compose up -d
cd backend && docker-compose up -d

# 2. Verify health
python test_api.py --service yolo --health
python test_api.py --service vqa --health
```

### Baseline Phase (100-300 samples)
```bash
# 3. Establish YOLO baseline
python test_api.py --service yolo --dir images/baseline/yolo --delay 0.5

# 4. Establish VQA baseline
python test_api.py --service vqa --dir images/baseline/vqa --delay 0.5

# 5. Verify baseline established
python test_api.py --service yolo --drift-status
python test_api.py --service vqa --drift-status
```

### Drift Testing Phase
```bash
# 6. Test brightness drift (YOLO)
python test_api.py --service yolo \
  --dir images/drift_scenarios/yolo/brightness/bright \
  --delay 0.5 --check-drift 10

# 7. Check drift status
python test_api.py --service yolo --drift-status

# 8. Reset if needed
python test_api.py --service yolo --reset-reference
```

### Performance Testing Phase
```bash
# 9. Load test with identical images
python test_api.py --service yolo \
  --dir images/load_test/identical \
  --delay 0.1 --output load_test_results.json

# 10. Repeat test for high volume
python test_api.py --service yolo \
  --image images/load_test/identical/identical_001.jpg \
  --repeat 200 --delay 0.2
```

## Expected Behaviors

### Baseline Phase
- **Sample Count**: Need 100+ samples for reference
- **Status**: "insufficient_data" until 100 samples
- **Drift**: No drift detection until baseline established

### Normal Operation
- **Drift Share**: < 10% typically
- **Dataset Drift**: False
- **Sample Count**: Maintains current window (50-100 samples)

### Drift Detection
- **Brightness Changes**: Should trigger drift in brightness feature
- **Object Density Changes**: Should trigger drift in detection count
- **Confidence Changes**: Should trigger drift in avg_confidence
- **Drift Share**: > 30% indicates significant drift
- **Dataset Drift**: True when multiple features drift

### Load Testing
- **Identical Images**: No drift, consistent metrics
- **Similar Images**: Minimal drift, stable metrics
- **High Volume**: Performance degrades gracefully

## Monitoring

### Check Grafana Dashboards
- YOLO Dashboard: http://localhost:3000/d/yolo-evidently
- VQA Dashboard: http://localhost:3000/d/vqa-evidently
- Prometheus Metrics: http://localhost:9090

### Key Metrics to Monitor
- **inference_duration_seconds**: Response time
- **drift_share**: Percentage of drifted features
- **reference_samples_total**: Baseline size
- **current_samples_total**: Current window size
- **dataset_drift**: Overall drift status

## Troubleshooting

### "insufficient_data" Status
- Need 100+ samples for reference baseline
- Continue submitting baseline images

### No Drift Detected
- Ensure baseline is from different distribution
- Try more extreme drift scenarios (very bright/dark)
- Check if drift threshold is appropriate

### Service Unavailable
```bash
# Check service status
docker ps

# View logs
docker logs backend-app-1
docker logs frontend-app-1

# Restart services
cd backend && docker-compose restart
```

## Best Practices

1. **Always establish baseline first** (100-300 samples)
2. **Use appropriate delay** (0.5s for testing, 0.1s for load tests)
3. **Monitor drift regularly** (check every 10-20 requests)
4. **Save results to JSON** for later analysis
5. **Reset reference** when switching test scenarios
6. **Use verbose mode** for debugging individual requests
7. **Check Grafana** for visual drift patterns
