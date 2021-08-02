import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
#import cv2

LOCAL_MQTT_HOST="mosquitto-service"
#LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="w251/final_project/chess"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the NX onboard camera
#cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    try:
        dim = (1000, 750)
        #dim = (500, 500)
        #frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return frame, 1
    except: 
        return None, 0

    # Display the resulting frame
source = "data/test/images/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg"
#frame = cv2.imread(source)
frame = "Testing code"
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 300)
local_mqttclient.publish(LOCAL_MQTT_TOPIC,frame,1)
        
 
