import numpy as np
import cv2
import paho.mqtt.client as mqtt
from datetime import datetime

LOCAL_MQTT_HOST="mosquitto-service"
#LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="w251/final_project/chess"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the NX onboard camera
cap = cv2.VideoCapture(0)

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
            local_mqttclient = mqtt.Client()
            local_mqttclient.on_connect = on_connect_local
            local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 300)
            local_mqttclient.publish(LOCAL_MQTT_TOPIC,frame,1)
            #cv2.imwrite(folder + date_string  + 'frame' + str(capture_num) + '.jpeg', small_frame)
            #print('Saved ' + str(capture_num))
            #capture_num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
 
cap.release()
cv2.destroyAllWindows()
#source.release()
