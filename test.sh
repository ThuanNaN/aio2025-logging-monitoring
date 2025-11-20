# YOLO examples
python test_api.py --service yolo --dir images/baseline/yolo/indoor --delay 0.5
python test_api.py --service yolo --reset-reference

python test_api.py --service yolo --dir images/drift_scenarios/yolo/brightness/bright
python test_api.py --service yolo --dir images/drift_scenarios/yolo/brightness/dark
python test_api.py --service yolo --dir images/drift_scenarios/yolo/object_density/crowded

python test_api.py --service yolo --drift-status

# VQA examples
python test_api.py --service vqa --dir images/baseline/vqa/general --delay 1.0
python test_api.py --service vqa --reset-reference

python test_api.py --service vqa --dir images/drift_scenarios/vqa/brightness/bright
python test_api.py --service vqa --dir images/drift_scenarios/vqa/brightness/dark
python test_api.py --service vqa --dir images/drift_scenarios/vqa/complexity/abstract
