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

FILE_NAME = sys.argv[1]
CONV_NAME = FILE_NAME[:-4]+"3030"+FILE_NAME[-4:]
WAV_NAME = FILE_NAME[:-4]+".wav"

print(FILE_NAME, CONV_NAME, WAV_NAME)

def chunkTo2ChannelArray(raw, chunksize):
  data = struct.unpack(str(2*chunksize)+'h', raw)
  return np.array(data).reshape(-1,2)

def binit(data, n_bins):
  pad_len = n_bins-(data.size%n_bins)
  res = np.sum(np.pad(np.abs(data), (0,pad_len)).reshape(n_bins, -1), axis=1)
  return res

# CHECKING FOR VIDEO FILE
if not os.path.isfile(CONV_NAME):
  conv3030.convert(FILE_NAME, (30,30))

# CHECKING FOR AUDIOFILE
if not os.path.isfile(WAV_NAME):
  moviepy.editor.VideoFileClip(FILE_NAME).audio.write_audiofile(WAV_NAME)

# INIT CAPTURES
wf = wave.open(WAV_NAME, 'rb')
wf_max = wf.getnframes()
cap = cv2.VideoCapture(CONV_NAME)
FPS = int(cap.get(cv2.CAP_PROP_FPS))
AUD_PER_VID = int(44100/FPS)

# INIT CAPTURE PROCESSING
four_freq = np.fft.rfftfreq(int(AUD_PER_VID*FPS), 1/44100) # np.fft.rfft(data, norm="ortho")
N_FOUR_BINS = 300
N_DATA_BINS = 300
FOUR_LIMIT_FRAC = 7
FREQ_LIMIT = int(four_freq.size/FOUR_LIMIT_FRAC)
MAX_FREQ = np.max(four_freq[FREQ_LIMIT-1])
print("max freq:", MAX_FREQ)

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.wav_conv1 = nn.Conv1d(1, 4, 6, stride=2)
    self.wav_conv2 = nn.Conv1d(4, 8, 8, groups=2, stride=2)
    self.wav_conv3 = nn.Conv1d(8, 16, 5, groups=4, stride=2)
    self.fc1 = nn.Linear(16*3, 30)
    self.fc2 = nn.Linear(30, 30)
    self.fc3 = nn.Linear(30, 900)


  def forward(self, x):
    x = F.max_pool1d(F.sigmoid(self.wav_conv1(x)), 2)
    x = F.max_pool1d(F.sigmoid(self.wav_conv2(x)), 2)
    x = F.max_pool1d(F.sigmoid(self.wav_conv3(x)), 2)
    x = x.flatten()
    tqdm.tqdm.write(str(x))
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = F.sigmoid(self.fc3(x))
    x = x.view(30, 30)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

# INIT NET
net = Net()
BATCH_SIZE = 20
N_BATCHES = int(wf_max/(44100))
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, N_BATCHES/3, gamma=1)
criterion = nn.L1Loss()

# TRAINING
raw_size = 0
net.train()
for epoch_num in trange(0):
  wf.rewind()
  cap.open(CONV_NAME)
  for batch_num in trange(N_BATCHES):
    ## INPUT PROCESSING
    # combining first audio second
    raw = wf.readframes(44100)
    if not raw_size: raw_size = sys.getsizeof(raw)
    if  sys.getsizeof(raw) != raw_size: break
    data = np.average(chunkTo2ChannelArray(raw, 44100), axis=1)
    binned_data = binit(data, N_DATA_BINS)
    four_data = np.fft.rfft(data, norm="ortho")[:FREQ_LIMIT]
    binned_four = binit(four_data, N_FOUR_BINS)
    in_wav = torch.tensor([[binned_four]]).float()
    # getting the frame after one second
    for i in range(FPS):
      if i==FPS-1:
        ret, frame = cap.read()
        if not ret: break
        gray = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).float()/255
      else:
        ret, _ = cap.read()
        if not ret: break

    ## FORWARD
    optimizer.zero_grad()
    out = net(in_wav)
    loss = criterion(out, gray)
    loss.backward()
    optimizer.step()
    scheduler.step()

    cv2.imshow("out", out.detach().numpy())
    key = cv2.waitKey()
    if key == 113:
      break

# CREATING
wf.rewind()
net.eval()
framerate = FPS
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
vid_writer = cv2.VideoWriter(FILE_NAME[:-4]+"_tripped"+FILE_NAME[-4:], int(fourcc), framerate, (30,30))

for _ in trange(N_BATCHES, desc= "Generating VID"):
  raw = wf.readframes(44100)
  if not raw_size: raw_size = sys.getsizeof(raw)
  if  sys.getsizeof(raw) != raw_size: break
  data = np.average(chunkTo2ChannelArray(raw, 44100), axis=1)
  binned_data = binit(data, N_DATA_BINS)
  four_data = np.fft.rfft(data, norm="ortho")[:FREQ_LIMIT]
  binned_four = binit(four_data, N_FOUR_BINS)
  in_wav = torch.tensor([[binned_four]]).float()

  out = (net(in_wav).detach().numpy()*255).astype("uint8")
  col_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
  #cv2.imshow("trip", out)
  #cv2.waitKey()
  vid_writer.write(col_out)

vid_writer.release()
cap.release()
wf.close()

