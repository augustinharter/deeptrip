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
import librosa

CHUNK = 4410

def chunk2array(raw):
  data = struct.unpack(str(2*CHUNK)+'h', raw)
  return np.array(data).reshape(-1,2)

def binit(data, n_bins):
  pad_len = n_bins-(data.size%n_bins)
  res = np.sum(np.pad(np.abs(data), (0,pad_len)).reshape(n_bins, -1), axis=1)
  return res

def logbinner(data, n_bins, min_bin=5):
  bins = list()
  start = 0
  stop = min_bin
  gamma = math.pow(data.size/min_bin, 1/n_bins)
  #print("Size:", data.size, " Gamma:", gamma)

  for _ in range(n_bins):
    bins.append(np.sum(np.abs(data[int(start):math.ceil(stop)])))
    start = stop
    stop = stop * gamma

  return np.array(bins)

# CHECKING FOR AUDIOFILE
if not os.path.isfile("parting_ways.wav"):
  moviepy.editor.VideoFileClip("parting_ways.avi").audio.write_audiofile("parting_ways.wav")


# INIT
p = pyaudio.PyAudio()
print(p.get_default_output_device_info())
in_stream = p.open(44100, 2, p.get_format_from_width(2), input=True, input_device_index=p.get_default_input_device_info()['index'])
plt.ion()
fig, axes = plt.subplots(3)
wav_ax = axes[0]
four_ax = axes[1]
hist_ax = axes[2]

# START PLOTTING
n_data_bins = 1000
data = np.average(chunk2array(in_stream.read(CHUNK)), axis=1)
binned_data = data #binit(data, n_data_bins)
wav_x = np.arange(0, binned_data.size)
wav_line, = wav_ax.plot(wav_x, binned_data)
#wav_ax.set_xlim([xmin,xmwav_ax])
wav_max = np.max(data)
wav_ax.set_ylim([-wav_max, wav_max])
wav_ax.axis('off')

max_freq = 3000
limit = int(data.size*(max_freq/44100))
four_data = np.fft.rfft(data, norm="ortho")[:limit]
four_x = np.fft.rfftfreq(data.size, d=1/44100)[:limit]
print(list(four_x))
four_line, = four_ax.plot(four_x, four_data)
four_max = np.max(four_data)
four_ax.set_ylim([-four_max,four_max])
#four_ax.axis('off')

n_bins = 20
hist_data = binit(four_data, n_bins)
#hist_data = np.mean(librosa.feature.mfcc(y=data, sr=44100, n_mfcc=n_bins), axis=1)
hist_x = np.arange(n_bins)
hist_line, = hist_ax.plot(hist_x, hist_data)
hist_max = np.max(hist_data)
hist_ax.set_ylim([-hist_max,hist_max])
hist_ax.axis('off')



fig.canvas.draw()
fig.canvas.flush_events()

while True:
  data = np.average(chunk2array(in_stream.read(CHUNK)), axis=1)
  four_data = np.fft.rfft(data, norm="ortho")[:limit]
  hist_data = logbinner(four_data, n_bins)
  #hist_data = np.mean(librosa.feature.mfcc(y=data, sr=44100, n_mfcc=n_bins), axis=1)

  #data = binit(data, n_data_bins)
  wav_max = np.max(data)
  wav_ax.set_ylim([-wav_max, wav_max])
  wav_line.set_ydata(data)
  wav_ax.draw_artist(wav_ax.patch)
  wav_ax.draw_artist(wav_line)

  four_max = np.max(four_data)
  four_ax.set_ylim([-four_max,four_max])
  four_line.set_ydata(four_data)
  four_ax.draw_artist(four_ax.patch)
  four_ax.draw_artist(four_line)

  hist_max = np.max(hist_data)
  hist_ax.set_ylim([-hist_max,hist_max])
  hist_line.set_ydata(hist_data)
  hist_ax.draw_artist(hist_ax.patch)
  hist_ax.draw_artist(hist_line)

  #fig.canvas.draw()
  fig.canvas.flush_events()