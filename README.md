## Chess board and piece Identification and Legal move generation using Reinforcement Learning
#### This is for Berkeley's MIDS program - course w251 - Final Project.

Authors: Prabhu Narsina

<b>summary</b>: Online chess has become prevalent with all young kids and most adults playing more during the last 18 months as we all sat at home with the COVID situation. However, there are a good number of players who prefer to play on a physical chess board. Online chess brings in a number of advantages like opening up to play anyone in the world and any time of the day. I believe we can provide the same online benefits to the players who prefer to play on a physical chess board.I here bring in the simple concept of digitizing a chess board using object identification and identifying / generating a legal move based on Reinforcement Learning. The digitization of the chess board involves three parts, 1) Chess board  2) Chess pieces identification 3) find the cell position for each chess piece. In the past, there have been efforts to identify chess pieces and translate them into a digital representation using the Canny edges, Hough transformation for finding lines and calculating intersection points. One of the techniques used to find chess piece type is to use cell piece location, extract image part for each cell and use Image classification technique (using CNNs).  I used a unique approach to solve the problem of finding a Chess board here, i.e use the same object identification technique for chess board by labelling for two different classes for inboard and outboard.

<b>Directory Structure</b>:    
  
    docker Folder:  
    1 Docker files to create containers for Jetson NX device 
      (Docker_chess_camera, Dockerfile_mqtt_nx_broker, Dockerfile_yolov5_chess)  
    2 Docker files to create containers for AWS VM  
      ( docker_chess_reinf)  
    note: docker_chess_reinf is used for tensorflow based reinforcemnt learning. Final implementation uses pytorch based and have been directly trained on the VM.
    
    kube Folder:  
    1 Kuberenetes YAML file for NX device  (chess_final.yaml)
    note: didn't use kubernetes in the cloud

    Python files (on the main folder)
    On Jetson NX
     1. readFromCamera.py - Read image from camera and send frame to MQTT broker when person presses key 's'  
     2. chess_board_detect_api.py - subscribe to MQTT for topic published by readFromCamera.py and integrates with Yolo Model and RL model. This also forward the images to AWS to save and used for further tuning of the model. This will be enhanced to save the results of object detection and Reinforced learning.
     3. detect_chess.py - takes in the frame and sends it to trained yolo model for Chess board and pieces identification and returns FEN notation of Chessboard
     4. rl_model_generate_legalMoves.py - takes in board with chess pieces and generates requested number of legal moves. It returns id of next action, which can be converted using chess environment defined in gym_chess_env.py
             
     On AWS VM 
     1. msg_aws_save.py - Subscribe to MQTT broker on AWS machine and save the chess board images S3 bucket.
     
     Files used for Training
     1.train.py from Yolov5- used for training Object detection model 
     2.detect_chess.py - used for testing Yolov5 model for chess board and chess pieces
     3.agent_chess_pytorch.py - used for training Reinforcement model for Chess Legal Move
     
    working_file folder
      Most of the python files in the main folder have notebook version to test and troubleshoot in this folder. This folder also have other things tried like vgg16,  
      training with RL with Keras.  This also have tools files that used to copy and split the folders for train & test.
     
   
<b>Docker/kube scripts</b>:    
  
    docker build -t w251_aws_mqtt_broker -f docker/Dockerfile_mqtt_base_aws .  
    docker build -t chess_rienf -f docker/docker_chess_reinf_aws .  
    docker build -t chess-yolo -f docker/Dockerfile_yolov5_chess .  
    docker build -t chess-live -f docker/Docker_chess_camera .  
    docker run -it --rm --net=host --ipc=host --runtime nvidia -e DISPLAY=$DISPLAY -v /home/prabhu/w251/chess_project:/chess chess_yolo  

    kubectl apply -f kube/chess_final.yaml  
    To log into the kubernetes based container  
    kubectl exec --stdin --tty <container> -- /bin/bash  
    
    --Troubleshooting Kubernetes deployment  
    If deployment has more than one container  
    kubectl exec -i -t my-pod --container main-app -- /bin/bash  

    Getting all the contaners in kube deployment  
    kubectl get pods --all-namespaces -o=jsonpath='{range .items[*]}{"\n"}{.metadata.name}{":\t"}{range .spec.containers[*]}{.name}{", "}{end}{end}' |sort  
    kubectl delete deployment chess-w251-final  
    kubectl delete service mosquitto-service  
    kubectl describe service mosquitto-service  
    kubectl port-forward service/mosquitto-service 1883:1883  
 
    kubectl get pods --show-labels  

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

<b>Linux troubleshooting scripts</b>:    
  
    top ps -ef | grep python  
    ps -ef | grep mosquitto   
    docker build -t chess-yolo -f docker/Dockerfile_yolov5_chess .
    sudo kill -9  
    netstat -tulpn | grep 1883  
    #for deleting old logs  
    sudo find /var/log -mtime +3 -type f -delete  
    
    --For mounting block device to aws ec2
    lsblk (find the one that is not attached)
    sudo mount <unattached disk from above step> <folder that you want to attach to>
   
  <b>Yolo training/testing script</b>:   
    
     
    python train.py --img 640 --batch 8 --epochs 50 --data ../chess_project/tiny.yaml --weights yolov5s.pt
 
    python detect.py --weights=runs/train/exp13/weights/best.pt --source=../chess_project/tripodimages

    note: Images are not provided in this git folder. 








