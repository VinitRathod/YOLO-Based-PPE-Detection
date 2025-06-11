import cv2
import torch
import numpy as np
from pathlib import Path
import time
import winsound

import sys
import yolov5

# Set YOLOv5 path (UPDATE THIS)
yolov5_path = Path("D:/University Of Regina/3rd Semester/CS713/CS713 Project/safety_detection_app/yolov5").absolute()
sys.path.append(str(yolov5_path))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device


class SafetyGearDetector:
    def __init__(self, weights_path):
        # Initialize model
        self.device = select_device('0')
        self.model = attempt_load(weights_path, device=self.device)
        self.model.half()  # FP16

        # Class names and thresholds
        self.class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

        self.conf_thres = {
            'Hardhat': 0.9,
            'NO-Hardhat': 0.6,
            'Mask': 0.5,
            'NO-Mask': 0.5,
            'NO-Safety Vest': 0.5,
            'Person': 0.7,
            'Safety Cone': 0.8,
            'Safety Vest': 0.85,
            'machinery': 0.9,
            'vehicle': 0.7
        }

        self.default_conf = 0.25
        self.last_alert_time = 0
        self.alert_cooldown = 5  # Seconds between alerts

    def detect(self, frame):
        # Preprocess with letterbox
        img = letterbox(frame, new_shape=640, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() / 255.0
        img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, 0.1, 0.4)

        safety_violation = False
        current_time = time.time()

        # Post-process with class-specific thresholds
        for det in pred:
            if len(det):
                filtered_det = []
                for *xyxy, conf, cls in det:
                    class_name = self.class_names[int(cls)]
                    threshold = self.conf_thres.get(class_name, self.default_conf)
                    if conf >= threshold:
                        filtered_det.append([*xyxy, conf, cls])
                        if class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                            safety_violation = True

                if filtered_det:
                    filtered_det = torch.tensor(filtered_det).to(self.device)
                    filtered_det[:, :4] = scale_boxes(img.shape[2:], filtered_det[:, :4], frame.shape).round()

                    for *xyxy, conf, cls in filtered_det:
                        class_name = self.class_names[int(cls)]
                        confidence = float(conf)

                        if 'NO-' in class_name:
                            color = (0, 0, 255)
                            cv2.putText(frame, "!", (int(xyxy[2]) + 5, int(xyxy[1]) + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            color = (0, 255, 0)

                        cv2.rectangle(frame,
                                      (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])),
                                      color, 2)

                        label = f'{class_name} {confidence:.2f}'
                        cv2.putText(frame, label,
                                    (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)

        # Trigger alerts if needed
        if safety_violation and (current_time - self.last_alert_time) > self.alert_cooldown:
            winsound.Beep(1000, 500)
            warning_text = "SAFETY VIOLATION DETECTED!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, warning_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.last_alert_time = current_time

        return frame, safety_violation

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = self.detect(frame)
            frames.append(processed_frame)

        cap.release()

        if output_path:
            # Save processed video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

        return frames