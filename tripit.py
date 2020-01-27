import librosa
import numpy as np
import sys
import os
import pyaudio
import cv2
import struct

CHUNK = 4096
DIM = (512,512)
CENT = 256

def chunkTo2ChannelArray(raw, chunksize):
  data = struct.unpack(str(2*chunksize)+'h', raw)
  return np.array(data).reshape(-1,2)

def binit(data, n_bins):
  pad_len = n_bins-(data.size%n_bins)
  res = np.sum(np.pad(np.abs(data), (0,pad_len)).reshape(n_bins, -1), axis=1)
  return res

p = pyaudio.PyAudio()
print(p.get_default_output_device_info())
in_stream = p.open(44100, 2, p.get_format_from_width(2), input=True, input_device_index=p.get_default_input_device_info()['index'])

# LOOP
mfcc_max = np.ones(20)
last = np.zeros((1,))
while True:
  raw = in_stream.read(CHUNK)
  aud = chunkTo2ChannelArray(raw, CHUNK)
  right = aud[:,0]
  left = aud[:,1]
  aud = np.mean(aud, axis=1)
  if aud.size <4096*10:
    aud = np.concatenate((last, aud))
  else:
    aud = np.concatenate((last[CHUNK:], aud))

  mfcc = librosa.feature.mfcc(aud)
  mfcc_max = np.array([max(np.max(row), mfcc_max[i]) for (i, row) in enumerate(mfcc)])
  mfcc = np.abs(mfcc.T/mfcc_max)
  mfcc2D = mfcc.T
  mfcc1D = np.mean(mfcc, axis=1)

  iters = mfcc1D.size
  xbase = 100
  ybase = 100
  rbase = 6.28/iters
  #mfcc = mfcc.astype("uint8")
  #print(mfcc)
  img = np.zeros(DIM)
  for i in range(iters):
    y = int( 256 + ybase * np.cos(rbase*i) )
    x = int( 256 + xbase * np.sin(rbase*i) )
    img[y,x] = int(mfcc1D[i]*255)

  print(img)
  cv2.imshow("img", img)
  cv2.waitKey(delay=50)

  
