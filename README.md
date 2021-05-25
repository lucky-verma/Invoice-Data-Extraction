## [Approach 1]

**if: CNN classifier**
1. get the bbox.
2. crop the bbox with OpenCV.
3. run OCR for that region.

**else:**
1. Get the important keywords coordinates.
2. Run OCR for the increased (let say 20% of width) bbox area to get the numerical data. 

### RUN

``python train.py --img 1200 --batch 4 --epochs 500 --data train/data.yaml --cfg models/yolov5l.yaml --weights models/yolov5l.pt``