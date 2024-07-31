import cv2
from FaceSegmentHandler import FaceSegmentHandler

img = cv2.imread(r".\test_data\people.jpg")
fs = FaceSegmentHandler()
segmented_img = fs.segment_on_image(img)
cv2.imwrite("segmented_img.png", segmented_img)
fs.segment_on_video(r".\test_data\1.mp4", r'C:\ffmpeg\bin\ffmpeg.exe')