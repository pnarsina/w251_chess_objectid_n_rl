import numpy as np
import cv2
from datetime import datetime

def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    try:
        dim = (1000, 750)
        #dim = (500, 500)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return frame, 1
    except: 
        return None, 0

cap = cv2.VideoCapture(0)

capture_num = 0
folder = "../images/"
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame,status = rescale_frame(frame)
    if(status == 1):
    # Display the resulting frame
        cv2.imshow('frame',  frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            date_string = datetime. now(). strftime("%Y_%m_%d_%I_%M_%S_%p")
            cv2.imwrite(folder + date_string  + 'frame' + str(capture_num) + '.jpeg', small_frame)
            print('Saved ' + str(capture_num))
            capture_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
