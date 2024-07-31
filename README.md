# Face Segmentation
A face segmentation tool based on YOLOv8 and SAM2

## Demo


https://github.com/user-attachments/assets/957bf1aa-0e53-412d-a59a-f29c82ef55bc


## How to use
- Create a Python3 virtual env
```
python3 -m venv ./venv
```
- Activate the environment creatred in the previous step
- cd to back folder
- install the requirements in requirements.txt file
```
pip install -r requirements
```
-  Code for inferring the pipeline on image and video
```
import cv2
from FaceSegmentHandler import FaceSegmentHandler

img = cv2.imread(r".\test_data\people.jpg")
fs = FaceSegmentHandler()
segmented_img = fs.segment_on_image(img)
cv2.imwrite("segmented_img.png", segmented_img)
fs.segment_on_video(r".\test_data\people.mp4")
```
