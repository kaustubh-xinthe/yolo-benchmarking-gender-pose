import cv2
from ultralytics import YOLO
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle ABC (in degrees) using vectors AB and BC.

    Parameters:
    - a, b, c (numpy.ndarray): Coordinates of the points A, B, and C.

    Returns:
    - float: Angle ABC in degrees.
    """
    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# def classify_pose(keypoints):
#     """
#     Classifies the pose based on the angles between hip, knee, and foot keypoints.
#
#     Parameters:
#     - keypoints (numpy.ndarray): The keypoints data, expected shape (17, 3).
#
#     Returns:
#     - str: 'Standing' if the person is standing, 'Sitting' if the person is sitting, otherwise 'Unknown'.
#     """
#     # Right side (hip 12, knee 14, foot 16)
#     if keypoints[11][0] > 0 and keypoints[13][0] > 0 and keypoints[15][0] > 0:
#         right_angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
#         if right_angle > 150:
#             print('Standing',keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
#             return 'Standing'
#         elif 65 <= right_angle < 125:
#             print('Sitting',keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
#             print("Sitting Angle",right_angle)
#             return 'Sitting'
#
#     # Left side (hip 13, knee 15, foot 17)
#     # print(keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
#     if keypoints[12][0] > 0 and keypoints[14][0] > 0 and keypoints[16][0] > 0:
#         left_angle = calculate_angle(keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
#         if left_angle > 150:
#             print('Left Side','Standing',keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
#             return 'Standing'
#         elif 68 <= left_angle < 85:
#             print('Left Side', 'Sitting',keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
#             print("Sitting Angle", left_angle)
#             return 'Sitting'
#     return
def classify_pose(keypoints):
    """
    Classifies the pose based on the angles between various keypoints for "Leaning," "Standing," and "Sitting."

    Parameters:
    - keypoints (numpy.ndarray): The keypoints data, expected shape (17, 3).

    Returns:
    - str: 'Leaning', 'Standing', 'Sitting', or 'Unknown' based on the detected pose.
    """
    # Calculate right-side (hip 12, knee 14, foot 16) and left-side (hip 13, knee 15, foot 17) angles
    if keypoints[11][0] > 0 and keypoints[13][0] > 0 and keypoints[15][0] > 0:
        right_angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
    else:
        right_angle = None

    if keypoints[12][0] > 0 and keypoints[14][0] > 0 and keypoints[16][0] > 0:
        left_angle = calculate_angle(keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
    else:
        left_angle = None

    # Calculate leaning angle (shoulder 7, hip 13, knee 15)
    if keypoints[6][0] > 0 and keypoints[12][0] > 0 and keypoints[14][0] > 0:
        leaning_angle = calculate_angle(keypoints[6][:2], keypoints[12][:2], keypoints[14][:2])
    else:
        leaning_angle = None

    # Check for "Leaning"
    if leaning_angle and (90 <= leaning_angle < 135):
        if (right_angle and right_angle > 150) or (left_angle and left_angle > 150):
            print("Leaning", keypoints[6][:2], keypoints[12][:2], keypoints[14][:2])
            print("Leaning Angle:", leaning_angle)
            print(leaning_angle)
            return 'Leaning'

    # Check for "Standing"
    if (right_angle and right_angle > 150) and (left_angle and left_angle > 150):
        print("Standing", keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
        return 'Standing'

    # Check for "Sitting"
    if (right_angle and 65 <= right_angle < 125) or (left_angle and 68 <= left_angle < 85):
        print("Sitting", keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
        return 'Sitting'

    return


# Load the YOLO pose model
pose_model = YOLO("yolov8m-pose.pt")

# Open the video file
video_path = "/home/xactai/Videos/version_v0.3_footage_pose_06-11-2024.mp4"
cap = cv2.VideoCapture(video_path)

# Define video writer if you want to save the output
output_path = "/home/xactai/Videos/pose_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run pose detection with tracking on the frame
    results = pose_model.track(frame, conf=0.5, persist=True, tracker='bytetrack.yaml')
    annotated_frame = results[0].plot(boxes=False, kpt_line=True, kpt_radius=7, labels=False)

    if len(results[0].keypoints.data[0]) > 0:
        # Iterate over each detected person
        for person_index, (person_keypoints, person_box) in enumerate(zip(results[0].keypoints.data, results[0].boxes)):
            # Convert keypoints to numpy and classify the pose
            key_points = person_keypoints.cpu().numpy()  # Keypoints for the person
            pose = classify_pose(key_points)  # Classify pose based on keypoints

            # Extract bounding box coordinates (assuming person_box provides [x1, y1, x2, y2])
            x1, y1, x2, y2 = person_box.xyxy[0].cpu().numpy()

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Display the pose label on top of the bounding box
            label_position = (int(x1)+200, int(y1) - 10)  # Position above the bounding box
            if pose:
                cv2.putText(annotated_frame, f"{pose}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (150, 0, 100), 4)

    # Extract tracking IDs and their coordinates for each detection
    for box in results[0].boxes:
        if box.id is not None:  # Only display if tracking ID is available
            tracking_id = int(box.id.item())
            # Extract bounding box coordinates (x1, y1, x2, y2)
            xyxy = box.xyxy[0].tolist()  # Extract the first element of the tensor and convert to list
            x1, y1, x2, y2 = map(int, xyxy)  # Convert the values to integers

            # Display the tracking ID text on the frame
            cv2.putText(annotated_frame, f"ID: {tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (150, 0, 100), 4)

    # Write the frame to output video file
    out.write(annotated_frame)

    # Resize for display if necessary
    annotated_frame = cv2.resize(annotated_frame, (800, 450))

    # Display the annotated frame
    cv2.imshow("YOLO Pose Detection - Tracking IDs Only", annotated_frame)


    # Press 'q' to exit the display loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
