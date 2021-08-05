#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gym_chess_env import ChessBoard_gym
from gym_chess_env import ChessBoard_gym
import wandb
from onecyclelr import OneCycleLR
from torch.distributions import uniform #for custom Batch Norm
from sklearn.preprocessing import MinMaxScaler
# In[2]:


wandb.login()
wandb.init(project="w251-prabhu-final_chessproject")


# In[3]:


env = ChessBoard_gym()


# In[4]:


#get_ipython().run_line_magic('matplotlib', 'inline')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
scaler = MinMaxScaler()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[7]:

class CustomBatchNorm(nn.Module):

    def __init__(self, in_size, momentum=0.9, eps = 1e-5):
        super(CustomBatchNorm, self).__init__()
        
        self.momentum = momentum
        self.insize = in_size
        self.eps = eps
        
        U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.gamma = nn.Parameter(U.sample(torch.Size([self.insize])).view(self.insize))
        self.beta = nn.Parameter(torch.zeros(self.insize))
            
        self.register_buffer('running_mean', torch.zeros(self.insize))
        self.register_buffer('running_var', torch.ones(self.insize))
        
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, input):
        
        X = input

        if len(X.shape) not in (2, 4):
            raise ValueError("only support dense or 2dconv")
        
        # dense layer
        elif len(X.shape) == 2:
            if self.training:
                mean = torch.mean(X, axis=0)
                variance = torch.mean((X-mean)**2, axis=0)
                
                self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean
                self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (input.shape[0]/(input.shape[0]-1)*variance)
            
            else:
                mean = self.running_mean
                variance = self.running_var
                
            X_hat = (X-mean) * 1.0 /torch.sqrt(variance + self.eps)
            out = self.gamma * X_hat + self.beta
  
				# convolutional layer
        elif len(X.shape) == 4:
            if self.training:
                N, C, H, W = X.shape
                mean = torch.mean(X, axis = (0, 2, 3))
                variance = torch.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
                
                self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean
                self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (input.shape[0]/(input.shape[0]-1)*variance)
            else:
                mean = self.running_mean
                var = self.running_var
                
            X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.reshape((1, C, 1, 1)) + self.eps)
            out = self.gamma.reshape((1, C, 1, 1)) * X_hat + self.beta.reshape((1, C, 1, 1))
        
        return out

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.DataParallel(nn.Linear(64, 128,bias=False))
        #self.bn1 = CustomBatchNorm(128)
        self.fc2 = nn.DataParallel(nn.Linear(128, 512,bias=False))
        #self.bn2 = CustomBatchNorm(512)
        self.fc3 = nn.DataParallel(nn.Linear(512, 256,bias=False))
        #self.bn3 = CustomBatchNorm(256)
        #self.fc4 = nn.DataParallel(nn.Linear(256, 128))
        self.head = nn.DataParallel(nn.Linear(256, outputs))

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.fc4(x))
        return self.head(x.view(x.size(0), -1))


# In[8]:


env.reset()
env.print_board()


# In[ ]:





# In[9]:


# Chess Cells 8 x 8
height = 8 
width = 8
n_actions = env.action_space.n
# n_actions = 112
BATCH_SIZE = 65536 
GAMMA = 0.999
EPS_START = 1 
EPS_END = 0.05
EPS_DECAY = 500000
TARGET_UPDATE = 10


# In[10]:


policy_net = DQN(height, width, n_actions).to(device)
target_net = DQN(height, width, n_actions).to(device)
# policy_net = DQN(n_actions).to(device)
# target_net = DQN(n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
# optimizer = optim.AdamW(policy_net.parameters())
# optimizer = optim.SGD(lr=0.0001 )
memory = ReplayMemory(5000000)


# In[11]:


def select_action(state):
    global steps_done
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# In[12]:


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.5)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


# In[13]:


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)

    pred = output 
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# In[13]:
def save_checkpoint(state,  filename='checkpoint.pth.tar'):
    # save the model state!
    torch.save(state, filename)


# In[14]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return(0,0,0)
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
  
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    acc1, acc5 = accuracy(state_action_values,  expected_state_action_values.unsqueeze(1), topk=(1, 5)) 
   # print('accuracies ', acc1, " ", acc5)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return (loss,acc1,acc5)


# In[ ]:

if (__name__ == "__main__"):

    num_episodes = 50 
    steps_done = 0
    observation_space = 64
    episode_durations = []
    rewards_list = []
    for i_episode in range(num_episodes):
        reward_for_episode = 0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(memory),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(i_episode))

        # Initialize the environment and state
        state = env.reset()
        scaler.fit(state)
        state = scaler.transform(state)
        state = torch.from_numpy(state).float()

        total_loss = 0
        loss = 0
        t=0
        scheduler.step()
        end = time.time()
        legal_move_count=0
        tot_acc1 = 0
        tot_acc5 = 0
        while True:
            # Select and perform an action
    #         state_model_input = np.reshape(state, state.shape + (1,)) 
            state_model_input = torch.reshape(state, [1, observation_space])

            action = select_action(state_model_input)
            next_state, reward, done, info = env.step(action.item())
            reward_for_episode += reward

            reward = torch.tensor([reward], device=device)
            

            # Store the transition in memory
            scaler.fit(next_state)
            next_state = scaler.transform(next_state)
            next_state = torch.from_numpy(next_state).float()

    #         next_state_model_input = np.reshape(next_state, next_state.shape + (1,)) 
            next_state_model_input = torch.reshape(next_state, [1, observation_space])

            memory.push(state_model_input, action, next_state_model_input, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss, acc1, acc5 = optimize_model()


            if loss is not None: 
                total_loss = total_loss + loss
                tot_acc1 = tot_acc1 + acc1
                tot_acc5 = tot_acc5 + acc5

                # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if done:
                legal_move_count += 1

            if legal_move_count >=200:
                episode_durations.append(t + 1)
                t+=1

                if loss is not None: 
                    total_loss = total_loss + loss
                    losses.update(total_loss, BATCH_SIZE)
                    top1.update(tot_acc1,  BATCH_SIZE)
                    top5.update(tot_acc5, BATCH_SIZE)
                #plot_durations()
                break

        rewards_list.append(reward_for_episode)
        last_rewards_mean = np.mean(rewards_list[-50:])
        # Update the target network, copying all weights and biases in DQN
        print ("loss of the episode ", i_episode , " is :", total_loss, " acc1:", tot_acc1, "acc5:", tot_acc5, " Batch size:",BATCH_SIZE , " Memory size:", len(memory), "reward mean:", last_rewards_mean)
        wandb.log({"Loss/val": losses.avg, 'acc1/val': top1.avg, 'acc5/val': top5.avg})

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    save_checkpoint({
       'epoch': num_episodes,
       'arch': 'DQN',
       'state_dict': target_net.state_dict(),
       'optimizer' : optimizer.state_dict(),
        })

    print('Complete')
    # env.render()
    # env.close()
    # plt.ioff()
    # plt.show()




