import paho.mqtt.client as mqtt
import sys
import numpy as np
import cv2
import s3fs
from datetime import datetime

LOCAL_MQTT_HOST="ec2-35-82-17-89.us-west-2.compute.amazonaws.com"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="w251/server/final/chess/images"
FILE_LOCATION="final_project/"
BUCKET_NAME="w251-prabhu"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  #try:
    #print("message received: ",str(msg.payload.decode("utf-8")))
    # if we wanted to re-publish this message, something like this should work
    msg = msg.payload
    #print("got message", msg)
    print("got message in aws save")
    file_name = FILE_LOCATION +  datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+ ".png"  
    image = np.asarray(bytearray(msg), dtype="uint8")
    #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    fs = s3fs.S3FileSystem(anon=False, key="XXXXXXXXXXXX", secret="xxxxxxxxxxx")

    with fs.open(f"{BUCKET_NAME}/{file_name}",'wb') as f: 
        f.write(image)
        f.close()
    #with open (file_name, "wb") as f:
    #    f.write(image)
    #    f.close()
    print("saved to file ", file_name)
  #except:
  #print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message



# go into a loop
local_mqttclient.loop_forever()
