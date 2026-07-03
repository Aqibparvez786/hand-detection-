import cv2

cap= cv2.VideoCapture(0)

if not cap.isOpened():
    print("404 Error : Webacam is not working ")
    exit()

print("webcam is started.....  , type 'q' to exit ")

while True:
    ret,frame=cap.read()

    if not ret:
        print("Error : Failed to grab frame..!")
        break
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()