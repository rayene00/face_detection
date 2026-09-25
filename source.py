import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8m-face.pt")

photo = cv2.imread('image2.jpg')
result = model(photo)
face_result=face_model(photo)

cat_person = result[0].plot()
cat_person_face = face_result[0].plot(img=cat_person)


cv2.imshow("my window", cat_person_face)
cv2.moveWindow("my window", 600, 5)
cv2.waitKey(500000)
cv2.destroyAllWindows()