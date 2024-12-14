import cv2
from ultralytics import YOLO
import os
import datetime
import time
from collections import defaultdict


def plot_images_with_boxes(img, results):
    global INDEX
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
    
    INDEX += 1
    cv2.imwrite(f'newdata/output_{INDEX}.jpg', img)


# track_history = set()


def main():
    person_class_id = 0
    INDEX = 0

    # rtsp_path = 'rtsp://admin:Test@1234@10.43.64.61/axis-media/media.amp'
    rtsp_path = 'output.avi'
    model = YOLO(model_path)

    cap = cv2.VideoCapture(rtsp_path)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed...")
            continue
        results = model.track(frame, persist=True)
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except Exception as e:
            print("SKIPPING, OBJECT LOST")
            continue
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls

        new_id = False
        image_width = frame.shape[1]
        image_height = frame.shape[0]

        annotations = [] 
        for box, track_id, class_id in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)


            if class_id == person_class_id: # and track_id not in track_history:
                # track_history.add(track_id)
                # frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                annotations.append(annotation)
                new_id = True

        # cv2.imshow('Recording', frame)
        if new_id:
            filename = f"0_{str(time.time()).split('.')[0]}"
            # filename = f"0_{str(datetime.datetime.now())}"
            cv2.imwrite(f'people/imgs/{filename}.jpg', frame)

            with open(f"people/labels/{filename}.txt", "w") as f:
                f.write("\n".join(annotations))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = os.path.join('models/yolov8s.pt')

    main()
