import numpy as np
import torch
import scipy as sp
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import trange
import math
import matplotlib
from matplotlib import pyplot as plt
import pyaudio
import struct
import moviepy.editor
import os
import wave
import sys

CHUNK = 4096

def chunk2array(raw):
  data = struct.unpack(str(2*CHUNK)+'h', raw)
  return np.array(data).reshape(-1,2)

# CHECKING FOR AUDIOFILE
if not os.path.isfile("parting_ways.wav"):
  moviepy.editor.VideoFileClip("parting_ways.avi").audio.write_audiofile("parting_ways.wav")


# INIT
wf = wave.open("parting_ways.wav", 'rb')
p = pyaudio.PyAudio()
stream = p.open(wf.getframerate(), wf.getnchannels(),p.get_format_from_width(2), output=True)
plt.ion()
fig, ax = plt.subplots()

# START PLOTTING
raw = wf.readframes(CHUNK)
data = np.average(chunk2array(raw), axis=1)
x = np.arange(0, data.size)
line, = ax.plot(x, data)
#ax.set_xlim([xmin,xmax])
ax.set_ylim([-2**15,(2**15)-1])
fig.canvas.draw()
fig.canvas.flush_events()

while True:
  raw = wf.readframes(CHUNK)
  data = np.average(chunk2array(raw), axis=1)
  line.set_ydata(data)
  ax.draw_artist(ax.patch)
  ax.draw_artist(line)
  #fig.canvas.update()
  fig.canvas.flush_events()
  stream.write(raw)

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(400, 120)  # 6*6 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # If the size is a square you can only specify a single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x), dim=0)
    #x = self.fc3(x)
    return x


base = np.arange(0,100,0.1)[:100]
signal = np.sin(base*6.282)
fourier = np.fft.fft(signal).real[:int(signal.size/2)]
freq = np.fft.fftfreq(base.size,0.05)[:int(signal.size/2)]
print(freq)
plt.plot(freq, fourier, base, signal)
plt.show()