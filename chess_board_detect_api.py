import paho.mqtt.client as mqtt
import sys
from detect_chess import convert_To_fen_chess_board

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="w251/final_project/chess"

REMOTE_MQTT_HOST="ec2-35-82-17-89.us-west-2.compute.amazonaws.com"
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC="w251/server/final/chess/images"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC,1)
        
def on_connect_remote(client, userdata, flags, rc):
        print("connected to remote broker with rc: " + str(rc))
        client.subscribe(REMOTE_MQTT_TOPIC,1)

remote_mqttclient = mqtt.Client("forwarder")
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 300) 

def on_message(client,userdata, msg):
# try:
    #print("message received: ",str(msg.payload.decode("utf-8")))
    print("Message recieved:")
    msg = msg.payload
    convert_To_fen_chess_board(msg)
    remote_mqttclient = mqtt.Client("forwarder")
    remote_mqttclient.on_connect = on_connect_remote
    remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    
 # except:
 #   print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 300)
