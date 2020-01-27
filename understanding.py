import numpy as np
import cv2
import torch
import scipy as sp
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import trange
import tqdm
import math
import matplotlib
from matplotlib import pyplot as plt
import pyaudio
import struct
import moviepy.editor
import os
import wave
import sys
import conv3030
import librosa

FILE_NAME = sys.argv[1]
CONV_NAME = FILE_NAME[:-4]+"3030"+FILE_NAME[-4:]
WAV_NAME = FILE_NAME[:-4]+".wav"

plt.ion()
print(FILE_NAME, CONV_NAME, WAV_NAME)

def chunkTo2ChannelArray(raw, chunksize):
  data = struct.unpack(str(2*chunksize)+'h', raw)
  return np.array(data).reshape(-1,2)

def binit(data, n_bins):
  pad_len = n_bins-(data.size%n_bins)
  res = np.sum(np.pad(np.abs(data), (0,pad_len)).reshape(n_bins, -1), axis=1)
  return res

def getTargetFrame(cap, sr=24):
  for i in range(round(sr)):
    if i==round(sr)-1:
      ret, frame = cap.read()
      if not ret: break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
    else:
      ret, _ = cap.read()
      if not ret: break
  return gray
    
# CHECKING FOR VIDEO FILE
if not os.path.isfile(CONV_NAME):
  conv3030.convert(FILE_NAME, (30,30))

# CHECKING FOR AUDIOFILE
if not os.path.isfile(WAV_NAME):
  moviepy.editor.VideoFileClip(FILE_NAME).audio.write_audiofile(WAV_NAME)

# INIT CAPTURES

aud, SAMPLE_RATE = librosa.load(WAV_NAME, res_type='kaiser_fast')
FREQ_LIMIT = int(11025*(2000/11025))
N_FOUR_BINS = 100
N_MFCC = 40

vid = cv2.VideoCapture(CONV_NAME)
FPS = int(vid.get(cv2.CAP_PROP_FPS))

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.four_conv1 = nn.Conv1d(1, 4, 8, stride=2)
    self.four_conv2 = nn.Conv1d(4, 4, 6, stride=1)
    self.mfcc_conv1 = nn.Conv1d(1, 4, 5, stride=2)
    self.mfcc_conv2 = nn.Conv1d(4, 4, 3, stride=1)
    self.fc1 = nn.Linear(48, 32, bias=False)
    self.fc2 = nn.Linear(32, 900, bias=False)


  def forward(self, four, mfcc):
    x = four
    fosh = four.shape
    x = F.max_pool1d((self.four_conv1(x)), 2)
    x = F.max_pool1d((self.four_conv2(x)), 2)
    four = x.view(fosh[0], -1)

    x = mfcc
    mfsh = mfcc.shape
    x = F.max_pool1d((self.mfcc_conv1(x)), 2)
    x = F.max_pool1d((self.mfcc_conv2(x)), 2)
    mfcc = x.view(mfsh[0], -1)

    x = torch.cat((four, mfcc), 1)

    x = F.relu(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    #x = F.sigmoid(self.fc3(x))
    x = x.view(-1, 30, 30)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

# INIT NET
net = Net()
BATCH_SIZE = 10
N_EPOCHS = 10
N_BATCHES = int(np.floor(aud.size/(BATCH_SIZE*SAMPLE_RATE)))
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
criterion = nn.L1Loss()

# TRAINING
net.train()
for epoch_num in trange(N_EPOCHS):
  vid.open(CONV_NAME)

  for batch_num in trange(N_BATCHES):
    ## INPUT PROCESSING
    # combining first audio second
    aud_in = aud[batch_num*BATCH_SIZE*SAMPLE_RATE:(batch_num+1)*BATCH_SIZE*SAMPLE_RATE]
    aud_in = aud_in.reshape(-1, SAMPLE_RATE)
    four_data = np.fft.rfft(aud_in, norm="ortho", axis=1)[:FREQ_LIMIT]
    four_in = np.array([[binit(row, N_FOUR_BINS)] for row in four_data])
    mfcc_in = np.array([[np.mean(librosa.feature.mfcc(y=row, n_mfcc=N_MFCC), axis=1)] for row in aud_in])
    #tqdm.tqdm.write(str(four_in.shape)+" "+str(mfcc_in.shape))
    # getting the frame after one second
    target = list()
    for _ in range(BATCH_SIZE):
      target.append(getTargetFrame(vid, sr=FPS))
    target = torch.tensor(target, dtype=torch.float)
    
    #print(four_in[0])
    #print(mfcc_in[0])
    four_in = torch.tensor(four_in).float()
    mfcc_in = torch.tensor(mfcc_in).float()

    ## FORWARD
    optimizer.zero_grad()
    out = net(four_in, mfcc_in)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    # cv2.imshow("out", out[0].detach().numpy())
    # key = cv2.waitKey()
    # if key == 113:
    #   break
  scheduler.step()

# CREATING
net.eval()
fourcc = vid.get(cv2.CAP_PROP_FOURCC)
vid_writer = cv2.VideoWriter(FILE_NAME[:-4]+"_tripped"+FILE_NAME[-4:], int(fourcc), FPS, (30,30))

for sec_num in trange(int(aud.size/SAMPLE_RATE)):
  ## INPUT PROCESSING
  # combining first audio second
  aud_in = aud[sec_num*SAMPLE_RATE:(sec_num+1)*SAMPLE_RATE]
  four_data = np.fft.rfft(aud_in, norm="ortho")[:FREQ_LIMIT]
  four_in = np.array([[binit(four_data, N_FOUR_BINS)]])
  mfcc_in = np.array([[np.mean(librosa.feature.mfcc(y=aud_in, n_mfcc=N_MFCC), axis=1)]])

  four_in = torch.tensor(four_in).float()
  mfcc_in = torch.tensor(mfcc_in).float()

  out = (net(four_in, mfcc_in)[0].detach().numpy()*255).astype("uint8")
  col_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
  # cv2.imshow(FILE_NAME, out)
  # cv2.waitKey()
  vid_writer.write(col_out)

vid_writer.release()
vid.release()

