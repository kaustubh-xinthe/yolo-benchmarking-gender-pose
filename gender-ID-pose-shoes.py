import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolov8m-pose.pt")
gender_shoe_model = YOLO("gender_shoe_yolov8x-v2.pt")

# Open the video file
video_path = "/home/xactai/Videos/1730716660259.mp4"
cap = cv2.VideoCapture(video_path)

# Define video writer to save the output if needed
output_path = "/home/xactai/Videos/gender_ID_shoe_points.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,tracker='bytetrack.yaml')
        print(results[0].boxes.id)
        tracking_ids = results[0].boxes.id

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=False, kpt_line=True, kpt_radius=7, labels=True)

        for box in results[0].boxes:
            if box.id is not None:  # Only display if tracking ID is available
                tracking_id = int(box.id.item())
                # Extract bounding box coordinates (x1, y1, x2, y2)
                xyxy = box.xyxy[0].tolist()  # Extract the first element of the tensor and convert to list
                x1, y1, x2, y2 = map(int, xyxy)  # Convert the values to integers

                # Display the tracking ID text on the frame
                cv2.putText(annotated_frame, f"ID: {tracking_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 0), 5)

        shoe_results = gender_shoe_model.predict(frame, conf=0.10)


        people_boxes = []
        counter = 1

        for result in shoe_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if gender_shoe_model.names[class_id] in ["man", "woman"]:
                    # Check if there are enough tracking IDs

                    # Get the bounding box coordinates
                    box_coordinates = box.xyxy[0].cpu().numpy()
                    people_boxes.append(
                            (box_coordinates, gender_shoe_model.names[class_id]))

                    x1, y1, x2, y2 = map(int, box_coordinates)  # Bounding box coordinates

                    # Define color and label based on gender
                    if gender_shoe_model.names[class_id] == "man":
                        bbox_color = (255, 0, 0)  # Blue for men
                        gender_label = 'Male'
                    else:
                        bbox_color = (0, 0, 255)  # Red for women
                        gender_label = 'Female'

                        # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 3)

                        # Define text properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.75
                    thickness = 5

                        # Calculate the size of the text and background rectangle
                    text_size = cv2.getTextSize(gender_label, font, font_scale, thickness)[0]
                    background_rect_start = (x1, y1 - 20)
                    background_rect_end = (x1 + text_size[0], y1)

                        # Draw the background rectangle for the label
                    #cv2.rectangle(annotated_frame, background_rect_start, background_rect_end, bbox_color, cv2.FILLED)

                        # Put the text on the frame
                    cv2.putText(annotated_frame, f"{gender_label}", (x1+175, y1 - 5),
                                    font, font_scale, (0, 0, 0), thickness)


                # Draw shoe bounding boxes
                if gender_shoe_model.names[class_id] == "shoe":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for shoes

        out.write(annotated_frame)
        # Display the annotated frame
        annotated_frame = cv2.resize(annotated_frame, (800, 450))

        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
