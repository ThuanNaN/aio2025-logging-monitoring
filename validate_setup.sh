#!/bin/bash

# Validation script to verify test infrastructure is ready
# Run this to check if everything is properly set up

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Infrastructure Validation${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check 1: Test scripts exist
echo -e "${YELLOW}[1/7] Checking test scripts...${NC}"
if [ -f "test_api.py" ] && [ -f "run_tests.sh" ]; then
    echo -e "${GREEN}✅ Test scripts found${NC}"
else
    echo -e "${RED}❌ Test scripts missing${NC}"
    exit 1
fi

# Check 2: Documentation exists
echo -e "${YELLOW}[2/7] Checking documentation...${NC}"
if [ -f "TEST_GUIDE.md" ] && [ -f "QUICK_TEST_REFERENCE.md" ]; then
    echo -e "${GREEN}✅ Documentation found${NC}"
else
    echo -e "${RED}❌ Documentation missing${NC}"
    exit 1
fi

# Check 3: Image directories exist
echo -e "${YELLOW}[3/7] Checking image directories...${NC}"
baseline_count=0
drift_count=0
load_count=0

if [ -d "images/baseline/yolo" ]; then ((baseline_count++)); fi
if [ -d "images/baseline/vqa" ]; then ((baseline_count++)); fi
if [ -d "images/drift_scenarios/yolo" ]; then ((drift_count++)); fi
if [ -d "images/drift_scenarios/vqa" ]; then ((drift_count++)); fi
if [ -d "images/load_test/identical" ]; then ((load_count++)); fi
if [ -d "images/load_test/similar" ]; then ((load_count++)); fi

if [ $baseline_count -eq 2 ] && [ $drift_count -eq 2 ] && [ $load_count -eq 2 ]; then
    echo -e "${GREEN}✅ All image directories found${NC}"
else
    echo -e "${RED}❌ Some image directories missing${NC}"
    echo "   Baseline: $baseline_count/2, Drift: $drift_count/2, Load: $load_count/2"
    exit 1
fi

# Check 4: Count images
echo -e "${YELLOW}[4/7] Counting images...${NC}"
yolo_baseline=$(find images/baseline/yolo -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
vqa_baseline=$(find images/baseline/vqa -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
yolo_drift=$(find images/drift_scenarios/yolo -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
vqa_drift=$(find images/drift_scenarios/vqa -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
load_test=$(find images/load_test -type f \( -iname "*.jpg" -o -iname "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
total=$((yolo_baseline + vqa_baseline + yolo_drift + vqa_drift + load_test))

echo -e "   YOLO baseline: $yolo_baseline images"
echo -e "   VQA baseline: $vqa_baseline images"
echo -e "   YOLO drift: $yolo_drift images"
echo -e "   VQA drift: $vqa_drift images"
echo -e "   Load test: $load_test images"
echo -e "   ${GREEN}Total: $total images${NC}"

if [ $total -lt 100 ]; then
    echo -e "${YELLOW}⚠️  Warning: Low image count (need 1000+ for full testing)${NC}"
fi

# Check 5: Python availability
echo -e "${YELLOW}[5/7] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo -e "${GREEN}✅ Python available: $python_version${NC}"
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Check 6: Required Python packages
echo -e "${YELLOW}[6/7] Checking Python packages...${NC}"
missing_packages=()

for package in requests; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=($package)
    fi
done

if [ ${#missing_packages[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ Required packages installed${NC}"
else
    echo -e "${YELLOW}⚠️  Missing packages: ${missing_packages[*]}${NC}"
    echo -e "   Install with: pip3 install ${missing_packages[*]}"
fi

# Check 7: Docker services (optional)
echo -e "${YELLOW}[7/7] Checking Docker services...${NC}"
if command -v docker &> /dev/null; then
    backend_running=$(docker ps --filter "name=backend" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')
    
    if [ $backend_running -gt 0 ]; then
        echo -e "${GREEN}✅ Backend services running${NC}"
        
        # Try to ping the API
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is accessible at http://localhost:8000${NC}"
        else
            echo -e "${YELLOW}⚠️  Backend running but API not responding${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Backend services not running${NC}"
        echo -e "   Start with: cd backend && docker-compose up -d"
    fi
else
    echo -e "${YELLOW}⚠️  Docker not found (services check skipped)${NC}"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}✅ Test infrastructure is ready!${NC}\n"

echo -e "Next steps:"
echo -e "  1. Ensure services are running: ${BLUE}./run_tests.sh health${NC}"
echo -e "  2. Run a quick test: ${BLUE}python3 test_api.py --service yolo --dir images/load_test/identical${NC}"
echo -e "  3. Establish baseline: ${BLUE}./run_tests.sh baseline yolo${NC}"
echo -e "  4. Test drift detection: ${BLUE}./run_tests.sh drift-brightness yolo${NC}"
echo -e "\nDocumentation:"
echo -e "  - Full guide: ${BLUE}TEST_GUIDE.md${NC}"
echo -e "  - Quick reference: ${BLUE}QUICK_TEST_REFERENCE.md${NC}"
echo -e "  - Update summary: ${BLUE}UPDATE_SUMMARY.md${NC}\n"
