import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
import gdown
from ultralytics import YOLO
import cv2
import subprocess
from tqdm import tqdm
from glob import glob


class FaceSegmentHandler:
    def __init__(self):
        self._checkpints_root_path = os.path.join(os.getcwd(), "checkpoints")

        self._root_to_save_temp_frames = os.path.join(os.getcwd(), ".temp_frames")
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        os.makedirs(self._checkpints_root_path, exist_ok=True)


        self._sam2_model_path = os.path.join(self._checkpints_root_path, "sam2_hiera_tiny.pt")
        self._sam2_model_cfg = "sam2_hiera_t.yaml"
        self._yolo_model_path = os.path.join(self._checkpints_root_path, "yolov8n-face.pt")
        

        if not(os.path.isfile(self._sam2_model_path)):
            print("Downloading SAM2 Checkpoint ...")
            gdown.download(
                "https://drive.google.com/uc?id=1M4RCWg9JDPjNSuKEImzeU2OFXRJ88tcm",
                self._sam2_model_path,
                quiet=False
            )


        if not(os.path.isfile(self._yolo_model_path)):
            print("Downloading YOLOv8 Face Detection Checkpoint ...")
            gdown.download(
                "https://drive.google.com/uc?id=10-iUQGoAkTaeahs-jC0W05G4qaspTF3N",
                self._yolo_model_path,
                quiet=False
            )

        self._face_det_model = YOLO(self._yolo_model_path)
        self._sam2_predictor = SAM2ImagePredictor(build_sam2(self._sam2_model_cfg, self._sam2_model_path))

    def _detect_face(self, img):
        res = self._face_det_model.predict(img, conf=0.3, verbose=False, device=self._device)
        boxes = res[0].boxes.xyxy.to("cpu").numpy().astype(int)
        return boxes

    def _segment_face(self, img, box):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._sam2_predictor.set_image(img)
            masks, scores, _ = self._sam2_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=box[None, :],
                                multimask_output=False,
                                )

        return masks

    def segment_on_image(self, img):
        face_boxes = self._detect_face(img)
        masks = self._segment_face(img, face_boxes[0])
        segmented_img = self._apply_masks_to_rgb(img, masks)
        return segmented_img

    @staticmethod
    def _apply_masks_to_rgb(rgb_image, masks):
        if len(masks.shape) == 3:
            masks = masks[np.newaxis,:,:,:]
        if masks.shape[1] != 1 or masks.shape[2] != rgb_image.shape[0] or masks.shape[3] != rgb_image.shape[1]:
            raise ValueError("The masks and RGB image dimensions do not match")
        
        combined_mask = np.any(masks[:, 0], axis=0)
        combined_mask_3d = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
        masked_image = np.where(combined_mask_3d, rgb_image, 0)
        return masked_image
        

    def segment_on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        os.makedirs(self._root_to_save_temp_frames, exist_ok=True)
        counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            segmented_image = self.segment_on_image(frame)

            combined_image = np.concatenate((frame, segmented_image), axis=1)
            cv2.imwrite(os.path.join(self._root_to_save_temp_frames, f"{str(counter).zfill(10)}.png"), combined_image)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1

        cap.release()
        cv2.destroyAllWindows()


        self._write_frames_ffmpeg("output.mp4", fps, video_width*2,video_height)
    

    def _write_frames_ffmpeg(self, output_video_path, fps, video_width, video_height):
        ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'
        ffmpeg_command = [
            ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{video_width}x{video_height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]

        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Making the output video ...")
        for frame_p in tqdm(glob(os.path.join(self._root_to_save_temp_frames, "*.png"))):
            frame = cv2.imread(frame_p)
            process.stdin.write(frame.tobytes())

        process.stdin.close()


if __name__ == "__main__":
    
    img = cv2.imread(r".\test_data\people.jpg")
    fs = FaceSegmentHandler()
    segmented_img = fs.segment_on_image(img)
    cv2.imwrite("segmented_img.png", segmented_img)
    fs.segment_on_video(r".\test_data\1.mp4")
    


