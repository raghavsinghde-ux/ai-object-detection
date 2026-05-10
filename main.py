from ultralytics import YOLO
import cv2

model=YOLO("yolov8m.pt")

cap= cv2.VideoCapture(0)
print(cap.isOpened())

cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, frame = cap.read()

    if not ret: 
        break
    results= model(frame, imgsz=480, conf=0.4,)
    annotated_frame = results[0].plot()

    cv2.imshow("All Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release() 
cv2.destroyAllWindows()        