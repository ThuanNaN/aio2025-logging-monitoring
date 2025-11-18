# YOLO examples
python test_api.py --service yolo --image img.jpg
python test_api.py --service yolo --dir images/ --delay 1.0

# VQA examples
python test_api.py --service vqa --image img.jpg --question "What is this?"
python test_api.py --service vqa --dir images/ --delay 1.0
python test_api.py --service vqa --image img.jpg --questions "What is this?" "Where is it?"

# Drift monitoring
python test_api.py --service yolo --drift-status
python test_api.py --service vqa --drift-status
python test_api.py --service yolo --reset-reference
python test_api.py --service vqa --reset-reference