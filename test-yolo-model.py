import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from openvino.runtime import Core
from ultralytics.utils import ops
import torch
import random


# Helper function to plot one box
def plot_one_box(box, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


# Paths
VIDEO_PATH = "video.mp4" # input file
OUTPUT_VIDEO_PATH = "output.mp4" # output file
DET_MODEL_NAME = "yolov8m" # model
DET_MODEL_PATH = f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"

# Initialize OpenVINO core and compile model
core = Core()
det_model = YOLO(f'{DET_MODEL_NAME}.pt')
if not Path(DET_MODEL_PATH).exists():
    det_model.export(format="openvino", dynamic=True, half=False)
det_ov_model = core.read_model(DET_MODEL_PATH)
device = "GPU.0"
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)


# Preprocess image
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scale_fill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


def preprocess_image(img0):
    img, _, _ = letterbox(img0, new_shape=(640, 640))
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img


def postprocess_and_draw(detections, input_hw, orig_img, label_map, min_conf_threshold=0.25, nms_iou_threshold=0.7):
    person_class_id = next((id for id, name in label_map.items() if name == 'person'), None)
    if person_class_id is None:
        return orig_img

    preds = ops.non_max_suppression(
        torch.from_numpy(detections[0]),
        min_conf_threshold,
        nms_iou_threshold,
        nc=80
    )
    for i, det in enumerate(preds):
        det[:, :4] = ops.scale_boxes(input_hw, det[:, :4], orig_img.shape).round()
        for *xyxy, conf, cls in det:
            print(cls)
            if int(cls) == person_class_id:
                label = f'{label_map[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, orig_img, label=label, color=[random.randint(0, 255) for _ in range(3)],
                             line_thickness=2)
    return orig_img


# Video processing
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file {VIDEO_PATH}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Inference
    input_tensor = preprocess_image(frame)
    input_tensor = np.expand_dims(input_tensor, 0)
    assert input_tensor.shape == (1, 3, 640, 640), f"Input tensor shape mismatch: {input_tensor.shape}"
    result = det_compiled_model([input_tensor])
    detections = result[det_compiled_model.output(0)]

    # Postprocess and draw
    input_hw = input_tensor.shape[2:]
    label_map = det_model.model.names
    output_frame = postprocess_and_draw([detections], input_hw, frame, label_map)

    # Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(output_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Write frame to output video
    out.write(output_frame)

    # Display the resulting frame
    cv2.imshow('Frame', output_frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
