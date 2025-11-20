# Image Setup Guide for Drift Testing

This guide explains how to organize images to effectively test drift detection for both YOLO object detection and VQA services.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Image Categories for Drift Testing](#image-categories-for-drift-testing)
3. [YOLO Drift Testing Setup](#yolo-drift-testing-setup)
4. [VQA Drift Testing Setup](#vqa-drift-testing-setup)
5. [Sample Dataset Organization](#sample-dataset-organization)
6. [Testing Scenarios](#testing-scenarios)
7. [Quick Start Examples](#quick-start-examples)

## Directory Structure

Create the following directory structure for organized drift testing:

```plaintext
test_images/
├── baseline/                    # Reference/baseline images
│   ├── yolo/
│   │   ├── normal/             # Normal lighting, standard objects
│   │   ├── indoor/             # Indoor scenes
│   │   └── outdoor/            # Outdoor scenes
│   └── vqa/
│       ├── general/            # General purpose images
│       ├── people/             # Images with people
│       └── objects/            # Images with common objects
│
├── drift_scenarios/            # Images designed to trigger drift
│   ├── yolo/
│   │   ├── brightness/
│   │   │   ├── dark/          # Very dark images (brightness < 50)
│   │   │   ├── bright/        # Very bright images (brightness > 220)
│   │   │   └── high_contrast/ # High contrast images
│   │   ├── object_density/
│   │   │   ├── few_objects/   # Images with 0-2 objects
│   │   │   ├── many_objects/  # Images with 10+ objects
│   │   │   └── crowded/       # Very crowded scenes (20+ objects)
│   │   ├── confidence/
│   │   │   ├── low_quality/   # Blurry, low resolution
│   │   │   ├── occluded/      # Heavily occluded objects
│   │   │   └── unusual_angles/ # Unusual viewpoints
│   │   └── new_classes/
│   │       └── rare_objects/  # Uncommon object types
│   └── vqa/
│       ├── brightness/
│       │   ├── dark/          # Dark images
│       │   └── bright/        # Bright images
│       ├── complexity/
│       │   ├── simple/        # Single object, simple scenes
│       │   ├── complex/       # Multiple objects, complex scenes
│       │   └── abstract/      # Abstract or artistic images
│       └── domains/
│           ├── medical/       # Medical imagery (domain shift)
│           ├── aerial/        # Aerial/satellite images
│           └── technical/     # Technical diagrams, charts
│
└── load_test/                  # Images for load/stress testing
    ├── identical/              # Same image repeated
    └── similar/                # Very similar images
```

## Image Categories for Drift Testing

### 1. Baseline Images (100+ images)

**Purpose:** Establish the reference distribution

**Characteristics:**

- Diverse but representative of normal operation
- Balanced brightness (80-180)
- Mix of indoor/outdoor scenes
- Various object counts (3-8 objects typical)
- Good image quality
- Standard resolution (640x480 to 1920x1080)

**YOLO Baseline:**

```plaintext
- Street scenes with cars, pedestrians, traffic signs
- Indoor scenes with furniture, people, electronics
- Outdoor scenes with nature, buildings, animals
- Various lighting conditions (but not extreme)
- Mix of object densities (2-10 objects)
```

**VQA Baseline:**

```plaintext
- Everyday scenes (living rooms, kitchens, streets)
- People performing common actions
- Common objects in typical contexts
- Clear, well-lit images
- Variety of subjects for diverse questions
```

### 2. Drift-Inducing Images (50+ per category)

**Purpose:** Test specific drift detection scenarios

#### Brightness Drift

- **Dark images:** brightness < 50 (night scenes, poorly lit interiors)
- **Bright images:** brightness > 220 (overexposed, snow scenes, white backgrounds)
- **High contrast:** Mix of very dark and very bright regions

#### Object Density Drift (YOLO)

- **Few objects:** 0-2 objects (minimalist scenes, single subject portraits)
- **Many objects:** 10-20 objects (crowded markets, shelves, traffic)
- **Extreme density:** 20+ objects (stadium crowds, dense forests)

#### Confidence Drift (YOLO)

- **Low quality:** Blurry, pixelated, low resolution
- **Occlusion:** Partially hidden objects
- **Unusual angles:** Bird's eye view, extreme perspectives
- **Poor lighting:** Shadows, backlighting

#### Question Pattern Drift (VQA)

- Images requiring specific question types:
  - **Counting:** Multiple similar objects ("How many cars?")
  - **Location:** Objects in specific positions ("Where is the cat?")
  - **Identity:** People or specific objects ("Who is this?")
  - **Description:** Complex scenes ("What is happening?")

#### Domain Drift

- **New visual domains:** Medical images, aerial photos, technical drawings
- **Different art styles:** Cartoons, paintings, sketches
- **Unusual contexts:** Underwater, space, microscopic

## YOLO Drift Testing Setup

### Step 1: Collect Baseline Images (100-150 images)

```bash
# Create baseline directory
mkdir -p test_images/baseline/yolo/{normal,indoor,outdoor}

# Distribute images:
# - 40-50 images: Indoor scenes (furniture, people, electronics)
# - 40-50 images: Outdoor scenes (streets, parks, buildings)
# - 20-30 images: Normal mixed scenes
```

**Baseline Criteria:**

- Brightness: 80-180 (mean pixel value)
- Object count: 2-8 objects per image
- Confidence: 0.5-0.9 average
- Image quality: Good (sharp, well-lit)

### Step 2: Prepare Drift Test Sets (50 images each)

**Brightness Drift Test:**

```bash
mkdir -p test_images/drift_scenarios/yolo/brightness/{dark,bright,high_contrast}

# Dark images (target brightness < 50)
# - Night photography
# - Poorly lit interiors
# - Silhouettes

# Bright images (target brightness > 220)
# - Overexposed photos
# - Snow/beach scenes
# - White backgrounds

# High contrast (mixed brightness)
# - Sunset/sunrise
# - Strong shadows
# - Indoor/outdoor transitions
```

**Object Density Drift Test:**

```bash
mkdir -p test_images/drift_scenarios/yolo/object_density/{few_objects,many_objects,crowded}

# Few objects (0-2 objects)
# - Minimalist scenes
# - Single person portraits
# - Empty rooms

# Many objects (10-20 objects)
# - Grocery store shelves
# - Busy streets
# - Parking lots

# Crowded (20+ objects)
# - Stadium crowds
# - Dense traffic
# - Market scenes
```

**Confidence Drift Test:**

```bash
mkdir -p test_images/drift_scenarios/yolo/confidence/{low_quality,occluded,unusual_angles}

# Low quality images
# - Intentionally blurred
# - Low resolution (< 480p)
# - Compressed/artifacts

# Occluded objects
# - Partially hidden objects
# - Objects behind obstacles
# - Overlapping objects

# Unusual angles
# - Top-down views
# - Extreme perspectives
# - Fisheye lens
```

### Step 3: Verification Script

```bash
# Create a verification script
cat > verify_yolo_images.sh << 'EOF'
#!/bin/bash

echo "Verifying YOLO test images..."

BASELINE="test_images/baseline/yolo"
DRIFT="test_images/drift_scenarios/yolo"

echo "📊 Baseline Images:"
echo "  Normal: $(find $BASELINE/normal -type f | wc -l)"
echo "  Indoor: $(find $BASELINE/indoor -type f | wc -l)"
echo "  Outdoor: $(find $BASELINE/outdoor -type f | wc -l)"

echo -e "\n🔄 Drift Test Images:"
echo "  Dark: $(find $DRIFT/brightness/dark -type f 2>/dev/null | wc -l)"
echo "  Bright: $(find $DRIFT/brightness/bright -type f 2>/dev/null | wc -l)"
echo "  Few objects: $(find $DRIFT/object_density/few_objects -type f 2>/dev/null | wc -l)"
echo "  Many objects: $(find $DRIFT/object_density/many_objects -type f 2>/dev/null | wc -l)"

TOTAL_BASELINE=$(find $BASELINE -type f | wc -l)
TOTAL_DRIFT=$(find $DRIFT -type f 2>/dev/null | wc -l)

echo -e "\n✅ Total Baseline: $TOTAL_BASELINE (need 100+)"
echo "🔄 Total Drift Tests: $TOTAL_DRIFT (need 50+ per scenario)"

if [ $TOTAL_BASELINE -lt 100 ]; then
    echo "⚠️  Warning: Need more baseline images!"
fi
EOF

chmod +x verify_yolo_images.sh
```

## VQA Drift Testing Setup

### Step 1: Collect Baseline Images (100-150 images)

```bash
# Create baseline directory
mkdir -p test_images/baseline/vqa/{general,people,objects}

# Distribute images:
# - 40-50 images: General scenes (mixed content)
# - 30-40 images: People (various activities)
# - 30-40 images: Objects (various items)
```

**Baseline Criteria:**

- Brightness: 80-180 (mean pixel value)
- Variety: Different subjects, contexts, compositions
- Question-friendly: Clear subjects that can answer "what, where, who, how many"
- Quality: Clear, well-composed images

### Step 2: Prepare VQA-Specific Drift Sets

**Visual Feature Drift:**

```bash
mkdir -p test_images/drift_scenarios/vqa/brightness/{dark,bright}

# Similar to YOLO but focus on how it affects question answering
```

**Complexity Drift:**

```bash
mkdir -p test_images/drift_scenarios/vqa/complexity/{simple,complex,abstract}

# Simple (1-2 subjects)
# - Single object on plain background
# - Close-up portraits
# - Minimalist compositions

# Complex (5+ subjects, interactions)
# - Crowded scenes
# - Multiple people interacting
# - Rich environments

# Abstract
# - Abstract art
# - Patterns
# - Ambiguous scenes
```

**Domain Drift:**

```bash
mkdir -p test_images/drift_scenarios/vqa/domains/{medical,aerial,technical}

# Medical images (X-rays, MRI, surgical)
# Aerial/satellite imagery
# Technical diagrams, charts, screenshots
```

### Step 3: Question Templates for Testing

Create question lists for different test scenarios:

```bash
# Create question templates
cat > vqa_questions.txt << 'EOF'
# General questions (baseline)
What is in this image?
What objects can you see?
Describe this image
What is the main subject?

# Counting questions (test numeric understanding)
How many people are there?
How many objects can you count?
How many cars are visible?

# Location questions
Where is this photo taken?
Where is the person located?
What is in the background?

# Identity questions
What is this object?
What type of animal is this?
What color is the object?

# Action questions
What is the person doing?
What is happening in this picture?
What activity is shown?

# Complex questions
Why might this scene be unusual?
What time of day is it?
What is the relationship between the objects?
EOF
```

## Sample Dataset Organization

### Recommended Sources

1. **Public Datasets:**
   - COCO Dataset: http://cocodataset.org/
   - Open Images: https://storage.googleapis.com/openimages/web/index.html
   - ImageNet: https://www.image-net.org/
   - Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/

2. **Free Stock Photos:**
   - Unsplash: https://unsplash.com/
   - Pexels: https://www.pexels.com/
   - Pixabay: https://pixabay.com/

3. **Specialized Images:**
   - Medical: NIH Clinical Center datasets
   - Aerial: Landsat/Sentinel satellite imagery
   - Technical: Google Quick Draw dataset

### Download Helper Script

```bash
cat > download_test_images.sh << 'EOF'
#!/bin/bash

# Example script to organize downloaded images
# Assumes you've downloaded images to a 'downloads' folder

DOWNLOADS="downloads"
BASELINE="test_images/baseline"
DRIFT="test_images/drift_scenarios"

echo "Organizing test images..."

# Create directory structure
mkdir -p $BASELINE/{yolo,vqa}/{normal,indoor,outdoor,general,people,objects}
mkdir -p $DRIFT/{yolo,vqa}/brightness/{dark,bright}

# Move images based on naming convention or manual organization
# Example: Move dark images
find $DOWNLOADS -name "*night*" -o -name "*dark*" | while read img; do
    cp "$img" "$DRIFT/yolo/brightness/dark/"
    cp "$img" "$DRIFT/vqa/brightness/dark/"
done

# Move bright images
find $DOWNLOADS -name "*bright*" -o -name "*snow*" -o -name "*beach*" | while read img; do
    cp "$img" "$DRIFT/yolo/brightness/bright/"
    cp "$img" "$DRIFT/vqa/brightness/bright/"
done

echo "✅ Images organized!"
echo "Please review and manually organize remaining images."
EOF

chmod +x download_test_images.sh
```

## Testing Scenarios

### Scenario 1: Establish Baseline (Day 1)

```bash
# Run baseline establishment
python test_api.py --service yolo \
  --dir test_images/baseline/yolo \
  --delay 0.5 \
  --check-drift 25 \
  --output results/yolo_baseline.json

python test_api.py --service vqa \
  --dir test_images/baseline/vqa \
  --delay 1.0 \
  --check-drift 25 \
  --output results/vqa_baseline.json
```

**Expected Results:**

- Reference window fills (100 samples)
- No drift detected (baseline establishment)
- Metrics stabilize

### Scenario 2: Brightness Drift Test

```bash
# Test brightness drift
python test_api.py --service yolo \
  --dir test_images/drift_scenarios/yolo/brightness/dark \
  --delay 0.5 \
  --check-drift 10 \
  --output results/yolo_brightness_drift.json
```

**Expected Results:**

- `evidently_brightness_drift_score` increases
- `evidently_dataset_drift` may trigger (>30% features drifting)
- Alert: `BrightnessDriftHigh`

### Scenario 3: Object Density Drift Test (YOLO)

```bash
# Test with many objects
python test_api.py --service yolo \
  --dir test_images/drift_scenarios/yolo/object_density/many_objects \
  --delay 0.5 \
  --check-drift 10 \
  --output results/yolo_density_drift.json
```

**Expected Results:**

- `evidently_detections_drift_score` increases
- Average object count changes significantly
- Possible confidence drift (crowded scenes harder to detect)

### Scenario 4: VQA Question Pattern Drift

```bash
# Test with specific question types
python test_api.py --service vqa \
  --dir test_images/drift_scenarios/vqa/complexity/complex \
  --questions "How many people?" "How many objects?" "How many cars?" \
  --delay 1.0 \
  --check-drift 10 \
  --output results/vqa_question_drift.json
```

**Expected Results:**

- `vqa_evidently_question_length_drift_score` may increase
- Question type distribution shifts (all "how_many")
- Answer length may change

### Scenario 5: Domain Shift Test

```bash
# Test with out-of-domain images
python test_api.py --service yolo \
  --dir test_images/drift_scenarios/yolo/brightness/dark \
  --delay 0.5 \
  --verbose \
  --output results/yolo_domain_shift.json
```

**Expected Results:**

- Multiple drift indicators trigger
- Confidence drops significantly
- Detection count changes
- Strong dataset drift signal

### Scenario 6: Recovery Test

```bash
# Return to baseline
python test_api.py --service yolo --reset-reference
python test_api.py --service vqa --reset-reference

# Send baseline images again
python test_api.py --service yolo \
  --dir test_images/baseline/yolo/normal \
  --delay 0.5
```

**Expected Results:**

- New reference established
- Drift scores decrease
- System stabilizes

## Quick Start Examples

### Minimal Setup (for quick testing)

```bash
# Create minimal structure
mkdir -p test_images/{baseline,drift}

# Add 10-20 baseline images
cp ~/Pictures/sample_*.jpg test_images/baseline/

# Add 10 drift images (dark or bright)
cp ~/Pictures/dark_*.jpg test_images/drift/

# Test baseline
python test_api.py --service yolo \
  --dir test_images/baseline \
  --repeat 10 \
  --delay 0.3

# Test drift
python test_api.py --service yolo \
  --dir test_images/drift \
  --repeat 5 \
  --delay 0.3
```

### Full Production Setup

```bash
# 1. Create complete directory structure
./setup_image_directories.sh

# 2. Download and organize 200+ images
./download_test_images.sh

# 3. Verify setup
./verify_yolo_images.sh
./verify_vqa_images.sh

# 4. Establish baseline (100 images)
python test_api.py --service yolo --dir test_images/baseline/yolo --delay 0.5

# 5. Run drift tests
for scenario in brightness object_density confidence; do
    python test_api.py --service yolo \
      --dir test_images/drift_scenarios/yolo/$scenario \
      --delay 0.5 \
      --output results/yolo_${scenario}_drift.json
done

# 6. Check Grafana dashboards
open http://localhost:3000
```

## Best Practices

### Image Quality

- ✅ Use common formats (JPG, PNG)
- ✅ Keep resolution consistent (640-1920px wide)
- ✅ Ensure images are properly oriented
- ❌ Avoid corrupted or partial images
- ❌ Don't mix different aspect ratios excessively

### Baseline Selection

- ✅ Representative of production data
- ✅ Balanced across categories
- ✅ Good quality, clear subjects
- ❌ No extreme cases in baseline
- ❌ Avoid test/drift images in baseline

### Drift Testing

- ✅ Test one drift type at a time
- ✅ Document image sources and characteristics
- ✅ Keep drift images separate from baseline
- ✅ Start with subtle drift, increase gradually
- ❌ Don't mix multiple drift types
- ❌ Avoid testing with corrupted images

### Monitoring

- ✅ Check Grafana dashboards during tests
- ✅ Save JSON results for analysis
- ✅ Reset reference between different test scenarios
- ✅ Document drift thresholds that trigger alerts
- ❌ Don't run tests too fast (respect delay)

## Troubleshooting

### "Not enough data for drift detection"

**Solution:** Need 100 baseline + 50 current samples

```bash
python test_api.py --service yolo --drift-status
# Check reference_size and current_size
```

### Drift not detecting despite obvious differences

**Solution:** May need more samples or stronger drift
```bash
# Increase number of drift images
# Use more extreme examples
# Check if features are actually different
```

### False positives (drift detected on similar images)

**Solution:** Baseline may not be diverse enough
```bash
# Add more variety to baseline
# Check baseline statistics
python test_api.py --service yolo --dir test_images/baseline --verbose
```

### Service errors or timeouts

**Solution:** Check backend logs and resource usage
```bash
cd backend
docker compose logs -f
# Check GPU memory, reduce request rate
```

## Next Steps

1. **Automate Testing:**
   - Create scripts for systematic drift testing
   - Schedule periodic drift checks
   - Integrate with CI/CD pipeline

2. **Expand Dataset:**
   - Add more drift scenarios
   - Test edge cases
   - Include production-like data

3. **Analyze Results:**
   - Compare drift scores across scenarios
   - Tune alert thresholds
   - Document drift patterns

4. **Production Deployment:**
   - Establish production baseline
   - Set up continuous monitoring
   - Configure automated retraining triggers

## Additional Resources

- [EVIDENTLY_README.md](./EVIDENTLY_README.md) - Evidently drift detection details
- [VQA_README.md](./VQA_README.md) - VQA-specific guidance
- [MONITORING_SETUP_GUIDE.md](./MONITORING_SETUP_GUIDE.md) - Complete monitoring setup
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Command reference
