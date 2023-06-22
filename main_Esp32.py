import cv2
from urllib.request import urlopen
import numpy as np
import imutils

# Load the cascade XML files for face and helmet detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
helmet_cascade = cv2.CascadeClassifier('cascade.xml')
license_cascade=cv2.CascadeClassifier('haarcascade_russian_plate.xml')
# video_file = "video.mp4"
# cap = cv2.VideoCapture(video_file)
# Initialize the camera
url = r'http://192.168.69.187/capture'
while True:
    # Capture frame-by-frame
    
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    frame = cv2.imdecode(imgnp,-1)
    
    # Resize the frame to a smaller size for faster processing
    frame = imutils.resize(frame, width=600)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the face region and convert it to grayscale
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect helmets in the face region
        helmets = helmet_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If a helmet is detected, draw a rectangle around it
        if len(helmets) > 0:
            (hx, hy, hw, hh) = helmets[0]
            cv2.rectangle(frame, (x+hx, y+hy), (x+hx+hw, y+hy+hh), (0, 0, 255), 2)
            cv2.putText(frame, 'Helmet', (x+hx, y+hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'No Helmet', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the resulting frame
    
    license=license_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    c=0
    for (x,y,w,h) in license:
   
   # draw bounding rectangle around the license number plate
        if(len(license)>0):

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            gray_plates = gray[y:y+h, x:x+w]
            color_plates = frame[y:y+h, x:x+w]
            cv2.putText(frame, 'Number_Plate', (x+h, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite('./number_plate/image{}.png'.format(c),color_plates)
            c+=1



        
    # cv2.imshow('Number Plate Image', frame)
    cv2.imshow('Helmet & NO. Plate Detection', frame)
    # Exit the loop and release the camera when the user presses the "q" key

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
