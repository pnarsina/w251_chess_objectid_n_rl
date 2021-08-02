#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import chess
import chess.svg
from svglib.svglib import svg2rlg
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression,     apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
#get_ipython().run_line_magic('matplotlib', 'inline')


from reportlab.graphics import renderPM
# import chesshres
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import re
import glob
import PIL

class Detect_Chess_Board_n_Pieces():
    
    def __init__(self):
        super(Detect_Chess_Board_n_Pieces, self).__init__()
#       parameter values for Yolo model
        self.imgsz = 640
        self.augment = False
        self.save_txt=True
        self.save_conf=True
        self.save_crop=True
        self.project='runs/detect'
        self.name="exp"
        self.half = True
        self.exist_ok=False
        self.view_img = False
        self.save_img = True
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes = None # For filtering a specific class
        self.agnostic_nms=False  # class-agnostic NMS
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.line_thickness=3  # bounding box thickness (pixels)
        self.update = False
        self.FEN_mapping = {0:'b', 1:'k', 2:'n', 3:'p', 4:'q', 5:'r', 7:'B', 8:'K', 9:'N', 10:'P', 11:'Q', 12:'R'}
        self.chesmodel_weights = "model/chess_tuned_v1.pt"
        self.board_width = 1000
        
        self.device='0'
        self.batch_size = 32
        self.device = select_device(self.device, batch_size=self.batch_size)
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # This is needed to reverse order of rank, needed for FEN Conversion
        self.rank_reverse_mapping = {1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1}

    
        self.load_weights()

    # ### Get the model, used for two models  
    # 1. Chess board  
    # 2. Chess pieces

    # In[6]:


    def load_chess_yolov5_model(self,weights):
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(self.imgsz, s=gs)  # check image size
        self.stride = int(self.model.stride.max())  # model stride
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.half()  # to FP16
        return self.model, self.stride, self.class_names


    # In[7]:


    # names


    # ### Define method to get Object coordinates based on model and image

    # In[8]:


    def get_cordinates(self,dataset, model):
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
#             t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
#             t2 = time_synchronized()


            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir  / 'labels' /   p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
    #             print("detections: ", det)
    #             print("detections-4: ", det[:,:4])

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #                 print("After change detections-4: ", det[:,:4])

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.class_names[c] if self.hide_conf else f'{self.class_names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.class_names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
    #             print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if self.view_img:
                    plt.imshow(im0)
        #             cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
    #         print(f"Results saved to {save_dir}{s}")

        if self.update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

        print(f'Done. ({time.time() - t0:.3f}s)')
        return(det)



  


    def get_board_n_chesspiece_coords (self,dataset):
    #     board_det = get_cordinates(dataset, board_model).squeeze().detach().cpu().numpy()
    #     inboard_det = get_cordinates(dataset, inboard_model).squeeze().detach().cpu().numpy()
        inboard_class = 14
        outboard_class = 13
        chesspieces_det = self.get_cordinates(dataset, self.chesspieces_model).squeeze().detach().cpu().numpy()

        for piece in chesspieces_det:
            if int(piece[5]) == outboard_class:
                board_det = piece
            elif int(piece[5]) == inboard_class:
                inboard_det = piece

    #   Changing from trapezoid to square using inboard and outboard positions
        pts1 = np.float32([[inboard_det[0],inboard_det[1]],[inboard_det[2],inboard_det[1]],[board_det[2],board_det[3]],[board_det[0],board_det[3]]])
        pts2 = np.float32([[0,0],[self.board_width,0],[self.board_width,self.board_width],[0,self.board_width]])

    #   transformation matrix for converting to square
        M = cv2.getPerspectiveTransform(pts1,pts2)


        cell_width =  self.board_width/8
        cell_height = self.board_width/8
        piece_adjust = 1

        chesspieces_pos = []

        for piece in chesspieces_det:
            piece_class = int(piece[5])
            if piece_class not in (13,14):

                piece_class_name = self.class_names[piece_class]
                piece_class_prob = piece[4]
                transformed_cell = np.dot(M, [(piece[0]+piece[2])/2,(piece[1]+3*piece[3])/4,1])
                piece_x = np.min([transformed_cell[0]/transformed_cell[2],self.board_width])
                piece_y = np.min([transformed_cell[1]/transformed_cell[2],self.board_width]) 


                chesspieces_pos.append([np.ceil(piece_x/cell_width), self.rank_reverse_mapping.get(np.ceil(piece_y/cell_height)), piece_class, piece_class_name, piece_class_prob ])

        return chesspieces_pos

    # Converting array based representation to FEN representation
    def convert_to_FEN(self,chess_positions, threshold=0.5):
        chess_pos_array = np.array(chess_positions)
    #     add chess_pos_array[:,4]) for sorting on probability
        sorted_chess_pos = chess_pos_array[np.lexsort((chess_pos_array[:,1], chess_pos_array[:,0] ))]
        FENString = ""
        rank=1
        file=1
        prev_rank = 0
        prev_file = 1
        ctr=1
        print('sorted Chess positions: \n', sorted_chess_pos)


        for row in sorted_chess_pos:
            file = int(float(row[0]))
            rank = int(float(row[1]))
            prob = float(row[4])
            ch_class = int(float(row[2]))
            if rank <= 0: rank = 1
            if file <=0: file = 1
            print ( 'file:', file, 'rank:', rank, 'prev_file:', prev_file, 'prev_rank:', prev_rank,  'class:', self.FEN_mapping.get(int(float(row[2]))))
            if prob > threshold and ch_class != 6 and ch_class < 13 and file <= 8 and rank <=8 :
                if ( (ctr > 1)  and (prev_rank == rank) and (prev_file == file)):
                    print("duplicate")
                else:
                    if (  (file ==prev_file) and (rank > (prev_rank + 1))):
                        FENString = FENString + str(rank-(prev_rank + 1))
                    if (file > prev_file):
    #                   to cover left over spots in the old row
                        if prev_rank < 8:
                            FENString = FENString + str(8-prev_rank)
                        prev_rank = 0
                        if prev_file != 8:  
                            FENString = FENString + "/"


        #               if complete row is missing    
                        for gr in range(prev_file +1, file):
                            FENString = FENString + "8"
                            if gr != 8: 
                                FENString = FENString + "/"

    #                   to cover empty spots in the new row
                        if ( rank > 1):
                            FENString = FENString + str(rank - 1)

                    FEN_value = self.FEN_mapping.get(ch_class)

                    if(FENString is None):
                        print("Non object - FENString")
                    if(FEN_value is None):
                        print("Non object - get FEN ")

                    FENString = FENString + FEN_value
                    prev_rank = rank
                    prev_file = file
            ctr += 1      
    #     For any gaps left
        if prev_rank < 8:
            FENString = FENString + str(8-prev_rank)
        if prev_file != 8:
            FENString = FENString + "/"

    #               if complete row is missing    
        for gr in range(prev_file +1, 9):
            FENString = FENString + "8"
            if gr != 8: 
                FENString = FENString + "/"


        return FENString


    # In[13]:


    def fen_to_image(self,fen):
        board = chess.Board(fen)
        current_board = chess.svg.board(board=board)

        output_file = open('current_board.svg', "w")
        output_file.write(current_board)
        output_file.close()

        svg = svg2rlg('current_board.svg')
        renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")
        return current_board

    def load_weights(self):

        # #### Load Chess pieces and chessboard model
        self.chesspieces_model, self.stride, self.class_names = self.load_chess_yolov5_model(self.chesmodel_weights)
        self.chesspieces_model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.chesspieces_model.parameters())))



    def identiy_board_pieces(self,frame):
        chess_pos = self.get_board_n_chesspiece_coords(frame)
        FEN_string = self.convert_to_FEN(chess_pos)
        board = self.fen_to_image(FEN_string)
        return (FEN_string, board)


if __name__ == "__main__":
    
    detect_class = Detect_Chess_Board_n_Pieces()
    #source = "data/test/images/0b47311f426ff926578c9d738d683e76_jpg.rf.40183eae584a653181bbd795ba3c353f.jpg"
    source = "data/test/images/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg"
    # source = "data/test/images/2021_07_11_04_58_37_PMframe177.jpeg"
    # source = "data/test/images/2021_07_11_04_50_57_PMframe157.jpeg" 

    dataset = LoadImages(source, img_size=detect_class.imgsz, stride=detect_class.stride)
    FEN_string, board = detect_class.identiy_board_pieces(dataset)



    # FEN_mapping.get(0)
 
#     board = detect_class.fen_to_image(FEN_string)
    board_image = cv2.imread('current_board.png')
    print(FEN_string)
    cv2.imshow(board_image)



