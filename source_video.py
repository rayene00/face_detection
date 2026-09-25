import cv2
from ultralytics import YOLO #import for detection

model = YOLO("yolov8n.pt").to("cuda") #model for image detection
face_model = YOLO("yolov8m-face.pt").to("cuda")  #model for face detection

cap = cv2.VideoCapture(0)


while cv2.waitKey(1) !=ord("x"):
    _, frame= cap.read()
     
     #detection of the image2 (what's inside)
    result = model(frame, verbose=False)
    face_result=face_model(frame, verbose=False) #detection of the face(s) present on the photo
    
    cat_person = result[0].plot() #
    cat_person_face = face_result[0].plot(img=cat_person)
    
    cv2.imshow("my window",  cat_person_face) #display the window who contains the photo on the screen
    cv2.moveWindow("my window", 600, 5) #place where display appears
    
cv2.waitKey(500000) #time of window in ms
cap.release()
cv2.destroyAllWindows() #close the window when time finish
    