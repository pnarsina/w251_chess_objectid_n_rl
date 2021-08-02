### Chess board and piece Identification and Legal move generation using Reinforcement Learning
This is for Berkeley's MIDS program - course w251 - Final Project.

Authors: Prabhu Narsina
Directory Structure:
    docker Folder:
    1 Docker files to create containers for Jetson NX device 
      (Dockerfile_cv_face, Dockerfile_mqtt_nx_broker, Dockerfile_forward,  Dockerfile_nx_logger)
    2 Docker files to create containers for AWS VM
      ( Dockerfile_mqtt_base_aws, Dockerfile_aws_save)

    kube Folder:
    1 Kuberenetes YAML file for NX device  (hw3_k3s_nx.yaml)
    2.Kubenertes YAML file for AWS (hw3_aws.yaml - Not tested fully)

    pyfiles folder:
    On Jetson NX
     1. readFromCamera.py - Read image from camera and identify face and send byte stream to MQTT broker
     2. msg_forward.py - subscribe to MQTT for topic published by readFromCamera.py and forward it to remote aws MQTT broker
     3. nx_logger.py - Log messages to log folder on NX device  
       
     On AWS VM
     1. msg_aws_save.py - Subscribe to MQTT broker on AWS machine and save the identified face to S3 bucket.
     
    working_dir folder
      Intermediate files and other files created part of homework for troubleshooting and learning.
     
    logs
      log out put from NX_Logger (didn't check into github)
Scripts used on AWS to setup for Docker networking on AWS machine
docker network create --driver bridge hw03

docker build -t pnarsina1/w251_aws_mqtt_broker -f docker/Dockerfile_mqtt_base_aws .
docker build -t pnarsina1/w251_aws_hw3_save -f docker/Dockerfile_aws_save .

docker run --name mosquitto --rm --network hw03 -p 1883:1883 pnarsina1/w251_aws_mqtt_broker
docker run --name aws_save --rm --network hw03 -v /home/ubuntu/s3fs:/s3fs pnarsina1/w251_aws_hw3_save

Scripts used to build, deploy and troubleshoot containers and Kubernetes
#Deployment on Jetson NX device

docker build -t pnarsina1/w251_nx_face_capture -f docker/Dockerfile_cv_face .
docker build -t pnarsina1/w251_nx_mqtt_broker -f docker/Dockerfile_mqtt_nx_broker .
docker build -t pnarsina1/w251_nx_msg_forward -f docker/Dockerfile_forward .
docker build -t pnarsina1/w251_nx_logger -f docker/Dockerfile_nx_logger .

docker push pnarsina1/w251_nx_face_capture
docker push pnarsina1/w251_nx_mqtt_broker
docker push pnarsina1/w251_nx_msg_forward
docker push pnarsina1/w251_nx_logger

#For all Kubernetes deployment on JETSON device
kubectl create -f kube/hw3_k3s_nx.yaml
kubectl delete -f kube/hw3_k3s_nx.yaml

#For AWS machine
docker build -t pnarsina1/w251_aws_mqtt_broker -f docker/Dockerfile_mqtt_base_aws .
docker build -t pnarsina1/w251_aws_hw3_save -f docker/Dockerfile_aws_save .

#For Kubernetes with K8Micro
microk8s kubectl create -f kube/hw3_aws.yaml

#For troubleshooting
kubectl logs podname containername
kubectl describe podname
use -v option with mosquitto broker

#For deleting deployment
kubectl delete deployment

#For deleting service
kubectl delete service

#For testing through mosquitto client cli
mosquitto_pub -h mosquitto-service -p 1883 -t w251/face/capture -m "test from command prompt"

#cleaning docker repository
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
docker system prune

s3fs setup on AWS machine
sudo apt install s3fs

Enter your credentials in a file ${HOME}/.passwd-s3fs and set owner-only permissions:
echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ${HOME}/.passwd-s3fs
chmod 600 ${HOME}/.passwd-s3fs
s3fs facesnxcapture ${HOME}/s3fs -o passwd_file=${HOME}/.passwd-s3fs

Linux commands used for troubleshooting
top ps -ef | grep python
ps -ef | grep mosquitto
sudo kill -9
netstat -tulpn | grep 1883
#for deleting old logs
sudo find /var/log -mtime +3 -type f -delete
