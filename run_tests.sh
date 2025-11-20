#!/bin/bash

# Automated Testing Script for YOLO and VQA Drift Detection
# Usage: ./run_tests.sh [test_type] [service]
# Example: ./run_tests.sh baseline yolo

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="./images"
RESULTS_DIR="./test_results"
API_URL="http://localhost:8000"
DELAY=0.5
CHECK_DRIFT=10

# Create results directory
mkdir -p "$RESULTS_DIR"

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

check_health() {
    local service=$1
    print_info "Checking $service service health..."
    if python3 test_api.py --service "$service" --health --url "$API_URL"; then
        print_success "$service service is healthy"
        return 0
    else
        print_error "$service service is not responding"
        return 1
    fi
}

# Test functions
test_baseline_yolo() {
    print_header "YOLO - Baseline Establishment"
    
    print_info "Step 1/4: Resetting reference data..."
    python3 test_api.py --service yolo --reset-reference --url "$API_URL"
    
    print_info "Step 2/4: Submitting indoor baseline (100 images)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/baseline/yolo/indoor" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_yolo_indoor.json" \
        --url "$API_URL"
    
    print_info "Step 3/4: Submitting outdoor baseline (100 images)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/baseline/yolo/outdoor" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_yolo_outdoor.json" \
        --url "$API_URL"
    
    print_info "Step 4/4: Submitting normal baseline (100 images)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/baseline/yolo/normal" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_yolo_normal.json" \
        --url "$API_URL"
    
    print_success "YOLO baseline established (300 samples)"
    python3 test_api.py --service yolo --drift-status --url "$API_URL"
}

test_baseline_vqa() {
    print_header "VQA - Baseline Establishment"
    
    print_info "Step 1/4: Resetting reference data..."
    python3 test_api.py --service vqa --reset-reference --url "$API_URL"
    
    print_info "Step 2/4: Submitting general baseline (100 images)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/baseline/vqa/general" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_vqa_general.json" \
        --url "$API_URL"
    
    print_info "Step 3/4: Submitting objects baseline (100 images)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/baseline/vqa/objects" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_vqa_objects.json" \
        --url "$API_URL"
    
    print_info "Step 4/4: Submitting people baseline (100 images)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/baseline/vqa/people" \
        --delay "$DELAY" --check-drift 20 \
        --output "$RESULTS_DIR/baseline_vqa_people.json" \
        --url "$API_URL"
    
    print_success "VQA baseline established (300 samples)"
    python3 test_api.py --service vqa --drift-status --url "$API_URL"
}

test_drift_yolo_brightness() {
    print_header "YOLO - Brightness Drift Detection"
    
    print_info "Testing bright images (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/brightness/bright" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_bright.json" \
        --url "$API_URL"
    
    print_info "Testing dark images (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/brightness/dark" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_dark.json" \
        --url "$API_URL"
    
    print_info "Testing high contrast images (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/brightness/high_contrast" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_contrast.json" \
        --url "$API_URL"
    
    print_success "YOLO brightness drift tests completed"
    python3 test_api.py --service yolo --drift-status --url "$API_URL"
}

test_drift_yolo_confidence() {
    print_header "YOLO - Confidence Drift Detection"
    
    print_info "Testing low quality images (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/confidence/low_quality" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_low_quality.json" \
        --url "$API_URL"
    
    print_info "Testing occluded objects (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/confidence/occluded" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_occluded.json" \
        --url "$API_URL"
    
    print_info "Testing unusual angles (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/confidence/unusual_angles" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_angles.json" \
        --url "$API_URL"
    
    print_success "YOLO confidence drift tests completed"
    python3 test_api.py --service yolo --drift-status --url "$API_URL"
}

test_drift_yolo_density() {
    print_header "YOLO - Object Density Drift Detection"
    
    print_info "Testing crowded scenes (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/object_density/crowded" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_crowded.json" \
        --url "$API_URL"
    
    print_info "Testing sparse scenes (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/object_density/few_objects" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_few.json" \
        --url "$API_URL"
    
    print_info "Testing dense scenes (50 samples)..."
    python3 test_api.py --service yolo \
        --dir "$BASE_DIR/drift_scenarios/yolo/object_density/many_objects" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_yolo_many.json" \
        --url "$API_URL"
    
    print_success "YOLO density drift tests completed"
    python3 test_api.py --service yolo --drift-status --url "$API_URL"
}

test_drift_vqa_brightness() {
    print_header "VQA - Brightness Drift Detection"
    
    print_info "Testing bright images (50 samples)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/drift_scenarios/vqa/brightness/bright" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_vqa_bright.json" \
        --url "$API_URL"
    
    print_info "Testing dark images (50 samples)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/drift_scenarios/vqa/brightness/dark" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_vqa_dark.json" \
        --url "$API_URL"
    
    print_success "VQA brightness drift tests completed"
    python3 test_api.py --service vqa --drift-status --url "$API_URL"
}

test_drift_vqa_complexity() {
    print_header "VQA - Complexity Drift Detection"
    
    print_info "Testing abstract images (50 samples)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/drift_scenarios/vqa/complexity/abstract" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_vqa_abstract.json" \
        --url "$API_URL"
    
    print_info "Testing complex scenes (50 samples)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/drift_scenarios/vqa/complexity/complex" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_vqa_complex.json" \
        --url "$API_URL"
    
    print_info "Testing simple scenes (50 samples)..."
    python3 test_api.py --service vqa \
        --dir "$BASE_DIR/drift_scenarios/vqa/complexity/simple" \
        --delay "$DELAY" --check-drift "$CHECK_DRIFT" \
        --output "$RESULTS_DIR/drift_vqa_simple.json" \
        --url "$API_URL"
    
    print_success "VQA complexity drift tests completed"
    python3 test_api.py --service vqa --drift-status --url "$API_URL"
}

test_load_identical() {
    local service=$1
    print_header "$service - Load Test (Identical Images)"
    
    print_info "Testing with 10 identical images..."
    python3 test_api.py --service "$service" \
        --dir "$BASE_DIR/load_test/identical" \
        --delay 0.1 --check-drift 5 \
        --output "$RESULTS_DIR/load_${service}_identical.json" \
        --url "$API_URL"
    
    print_success "$service identical images load test completed"
    python3 test_api.py --service "$service" --drift-status --url "$API_URL"
}

test_load_similar() {
    local service=$1
    print_header "$service - Load Test (Similar Images)"
    
    print_info "Testing with 20 similar images..."
    python3 test_api.py --service "$service" \
        --dir "$BASE_DIR/load_test/similar" \
        --delay 0.1 --check-drift 5 \
        --output "$RESULTS_DIR/load_${service}_similar.json" \
        --url "$API_URL"
    
    print_success "$service similar images load test completed"
    python3 test_api.py --service "$service" --drift-status --url "$API_URL"
}

test_load_repeat() {
    local service=$1
    local count=${2:-200}
    print_header "$service - Repeat Load Test ($count requests)"
    
    local image_file="$BASE_DIR/load_test/identical/identical_001.jpg"
    
    print_info "Submitting same image $count times..."
    python3 test_api.py --service "$service" \
        --image "$image_file" \
        --repeat "$count" \
        --delay 0.2 --check-drift 20 \
        --output "$RESULTS_DIR/load_${service}_repeat_${count}.json" \
        --url "$API_URL"
    
    print_success "$service repeat load test completed"
    python3 test_api.py --service "$service" --drift-status --url "$API_URL"
}

test_all_yolo() {
    print_header "Running All YOLO Tests"
    
    check_health yolo || exit 1
    
    test_baseline_yolo
    sleep 2
    
    test_drift_yolo_brightness
    sleep 2
    
    test_drift_yolo_confidence
    sleep 2
    
    test_drift_yolo_density
    sleep 2
    
    test_load_identical yolo
    sleep 2
    
    test_load_similar yolo
    
    print_success "All YOLO tests completed!"
}

test_all_vqa() {
    print_header "Running All VQA Tests"
    
    check_health vqa || exit 1
    
    test_baseline_vqa
    sleep 2
    
    test_drift_vqa_brightness
    sleep 2
    
    test_drift_vqa_complexity
    sleep 2
    
    test_load_identical vqa
    sleep 2
    
    test_load_similar vqa
    
    print_success "All VQA tests completed!"
}

# Main script
main() {
    local test_type=${1:-help}
    local service=${2:-yolo}
    
    case "$test_type" in
        baseline)
            if [ "$service" = "yolo" ]; then
                test_baseline_yolo
            elif [ "$service" = "vqa" ]; then
                test_baseline_vqa
            else
                print_error "Invalid service: $service (use yolo or vqa)"
                exit 1
            fi
            ;;
        
        drift-brightness)
            if [ "$service" = "yolo" ]; then
                test_drift_yolo_brightness
            elif [ "$service" = "vqa" ]; then
                test_drift_vqa_brightness
            else
                print_error "Invalid service: $service"
                exit 1
            fi
            ;;
        
        drift-confidence)
            if [ "$service" = "yolo" ]; then
                test_drift_yolo_confidence
            else
                print_error "Confidence tests only available for YOLO"
                exit 1
            fi
            ;;
        
        drift-density)
            if [ "$service" = "yolo" ]; then
                test_drift_yolo_density
            else
                print_error "Density tests only available for YOLO"
                exit 1
            fi
            ;;
        
        drift-complexity)
            if [ "$service" = "vqa" ]; then
                test_drift_vqa_complexity
            else
                print_error "Complexity tests only available for VQA"
                exit 1
            fi
            ;;
        
        load-identical)
            test_load_identical "$service"
            ;;
        
        load-similar)
            test_load_similar "$service"
            ;;
        
        load-repeat)
            test_load_repeat "$service" "${3:-200}"
            ;;
        
        all)
            if [ "$service" = "yolo" ]; then
                test_all_yolo
            elif [ "$service" = "vqa" ]; then
                test_all_vqa
            else
                print_error "Invalid service: $service"
                exit 1
            fi
            ;;
        
        health)
            check_health yolo
            check_health vqa
            ;;
        
        help|*)
            echo -e "${BLUE}Automated Testing Script for ML API Drift Detection${NC}"
            echo ""
            echo "Usage: $0 [test_type] [service] [options]"
            echo ""
            echo "Test Types:"
            echo "  baseline              - Establish baseline reference data"
            echo "  drift-brightness      - Test brightness drift detection"
            echo "  drift-confidence      - Test confidence drift (YOLO only)"
            echo "  drift-density         - Test object density drift (YOLO only)"
            echo "  drift-complexity      - Test complexity drift (VQA only)"
            echo "  load-identical        - Load test with identical images"
            echo "  load-similar          - Load test with similar images"
            echo "  load-repeat [count]   - Repeat test (default: 200)"
            echo "  all                   - Run all tests for service"
            echo "  health                - Check service health"
            echo ""
            echo "Services:"
            echo "  yolo                  - YOLO object detection"
            echo "  vqa                   - Visual Question Answering"
            echo ""
            echo "Examples:"
            echo "  $0 baseline yolo              # Establish YOLO baseline"
            echo "  $0 drift-brightness vqa       # Test VQA brightness drift"
            echo "  $0 load-repeat yolo 500       # YOLO repeat test 500 times"
            echo "  $0 all yolo                   # Run all YOLO tests"
            echo "  $0 health                     # Check both services"
            echo ""
            ;;
    esac
}

# Run main function
main "$@"
