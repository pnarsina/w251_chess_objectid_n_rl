#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from gym_chess_env import ChessBoard_gym
from agent_chess_pytorch import DQN
import numpy as np
import math
import chess




# In[2]:
class Gen_Legal_move:
    def __init__(self, model_weights="checkpoint.pth-4rook_best-adamw.tar"):
        super(Gen_Legal_move, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DQN(8,8,112).to(device)
        model = load_from_saved_model(model,model_weights)
        
    def load_from_saved_model(model,  path = "checkpoint.pth.tar"):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        return(model)

    def generate_legal_moves(board, num_moves):
        state = torch.from_numpy(env.reset()).float()
        env = ChessBoard_gym()
        env.set_board(board)
        starting_pos_FEN = env.get_FEN()

        observation_space = 64
        state_model_input = torch.reshape(state, [1, observation_space])
        
        action_id = model(state_model_input).argmax(1)[0].detach()
        legal_move_ids = []
        for i in range(0,num_moves):
            next_state,reward, _, _  = env.step(action_id)
            next_state_model_input = torch.from_numpy(next_state).float()
            next_state_model_input = torch.reshape(next_state_model_input, [1, observation_space])
            action_id = actions_list.argmax(1)[0].detach()
            legal_move_ids.append(action_id)

        return(legal_move_ids)

