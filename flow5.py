import noise
import numpy as np
import tkinter as tk
import cv2
import math
import pyaudio
import struct
from matplotlib import pyplot as plt

rnd = np.random.rand

def chunk2array(raw, num):
  data = struct.unpack(str(2*num)+'h', raw)
  return np.array(data).reshape(-1,2)

def binner(data, n_bins):
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

# AUDIO
CHUNK = 4410
p = pyaudio.PyAudio()
print(p.get_default_output_device_info())
in_stream = p.open(44100, 2, p.get_format_from_width(2), input=True, input_device_index=p.get_default_input_device_info()['index'])

max_freq = 3000
min_freq = 50
upper_limit = int(CHUNK*(max_freq/44100))
lower_limit = int(CHUNK*(min_freq/44100))
n_data_bins = 20
maxi = 10000

# CV2
cv2.namedWindow('noisflow', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("noisflow",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
root = tk.Tk()
WIDTH = 1920#root.winfo_screenwidth()
HEIGHT = 1080#root.winfo_screenheight()
HALF_DIAG = math.pow(WIDTH**2 + HEIGHT**2, 0.5)
xb = WIDTH/2
yb = HEIGHT/2
img = np.zeros((HEIGHT, WIDTH))


# FIELDS
nfn = 50
fhscl = HEIGHT/nfn
fwscl = WIDTH/nfn
f = np.zeros((nfn,nfn,2))
z = 10
fs = 2/nfn
nfp = 4

spectrum = []

border_power = 1
border = True


class Dot:
  def __init__(self, y, x, vy, vx, r=1.0, vr=0, c=1):
    self.y = y
    self.x = x
    self.vy = vy
    self.vx = vx
    self.r = r
    self.vr = r
    self.c = c

  def gety(self):
    return self.y%HEIGHT

  def getx(self):
    return self.x%WIDTH

# PARTICLES
pn = 300
particles = [Dot(rnd()*HEIGHT, rnd()*WIDTH, 0, 0) for _ in range(pn)]
vmax = 15

amp_max = 5
amplitude = 200000/amp_max
space = HEIGHT/8
x_spread = 15
y_spread = 3*HEIGHT/(4*n_data_bins)

grow_rate = 1
degrowth = 0.7

def get_audio(plot=False):
  global maxi
  maxi *=0.995
  if maxi<1: maxi = 1

  n_avlb = in_stream.get_read_available()
  data = np.average(chunk2array(in_stream.read(n_avlb), n_avlb), axis=1)
  four_data = np.fft.rfft(data, norm="ortho")[lower_limit:upper_limit]
  four_data = logbinner(four_data, n_data_bins, min_bin=5)

  if plot:
    plt.plot(four_data)
    plt.show()

  candidate = np.max(four_data)
  if candidate>maxi:
    maxi = candidate
  amp_data = amp_max*four_data/maxi
  return amp_data

def calc_field(draw=False):
  global z
  a_len = 40
  for i in range(nfn):
    for j in range(nfn):
      f[i,j,0] = noise.snoise3(j*fs,i*fs,z, octaves=1)
      f[i,j,1] = noise.snoise3(j*fs,i*fs,z+100, octaves=1)
      if draw and i%10==0 and j%10 == 0:
        start = (0.5*fhscl+i*fhscl, 0.5*fwscl+j*fwscl)
        dest = (start[0]+f[i,j,0]*a_len, start[1]+f[i,j,1]*a_len)
        edge, _, _ = check_edge(*dest)
        if not edge: 
          line(start, dest)
  z+=0.001

def check_edge(y, x):
  edge = False
  if y < 0: 
    y = y%HEIGHT
    edge = True
  elif y >= HEIGHT: 
    y = y%HEIGHT
    edge = True
  if x < 0: 
    x = x%WIDTH
    edge = True
  elif x >= WIDTH: 
    x = x%WIDTH
    edge = True
  return edge,y,x

def apply_force_simple(dot):
  dot.vy += fp*f[int(dot.y/fhscl),int(dot.x/fwscl),0]
  if dot.vy>vmax: dot.vy = vmax
  if dot.vy<-vmax: dot.vy = -vmax
  dot.vx += fp*f[int(dot.y/fhscl),int(dot.x/fwscl),1]
  if dot.vx>vmax: dot.vx = vmax
  if dot.vx<-vmax: dot.vx = -vmax #-vmax

def apply_force(dot):
  global xb, yb

  if border:
    xd = (xb-dot.getx())/xb
    yd = (yb-dot.gety())/yb
    base = 2
    bf = (base/2)
    xf = bf*math.pow(base, abs(xd))-bf
    yf = bf*math.pow(base, abs(yd))-bf
    xif = 1-xf
    yif = 1-yf
  else:
    xif = 1
    yif = 1

  dot.vy += yif*nfp*f[int(dot.gety()/fhscl),int(dot.getx()/fwscl),0]
  dot.vy += border_power*yf*np.sign(yd)
  if dot.vy>vmax: dot.vy = vmax
  if dot.vy<-vmax: dot.vy = -vmax
  dot.vx += xif*nfp*f[int(dot.gety()/fhscl),int(dot.getx()/fwscl),1]
  dot.vx += border_power*xf*np.sign(xd)
  if dot.vx>vmax: dot.vx = vmax
  if dot.vx<-vmax: dot.vx = -vmax
  
def move_particle(dot):
  edge = False
  start = (int(dot.y), int(dot.x))
  dist = math.pow((dot.gety()-yb)**2 + (dot.getx()-xb)**2, 0.5)/HALF_DIAG
  #dist = min(dist, 0.9999)
  msf = spectrum[int(n_data_bins*dist)]
  #mmf = np.average(spectrum[:int(n_data_bins*dist)+1])
  dot.r *= degrowth
  dot.r += grow_rate * (msf/amp_max)
  dot.r = max(min(amp_max, dot.r), 0)
  dot.y += dot.r * dot.vy
  dot.x += dot.r * dot.vx
  edge, dot.y, dot.x = check_edge(dot.y, dot.x)
  if edge: return False
  dest = (int(dot.y), int(dot.x))
  if not edge: line(start, dest, c=dot.r)
  return True

def new_point(start, dest):
  global img
  img[start] = 0
  img[dest] = 1

def point(y, x, c=1):
  img[y,x] = c

def circle(pos, r, c=1):
  yy, xx = pos
  circ = round(math.pi*r*2)
  step = 1/r
  for i in range(int(circ)):
    y = round(yy + math.sin(i*step) * r)
    x = round(xx + math.cos(i*step) * r)
    edge, *_ = check_edge(y, x)
    if not edge: point(int(y), int(x), c)

def line(start, dest, c=1):
  global img

  yd = dest[0]-start[0]
  xd = dest[1]-start[1]

  n = int(math.ceil(max(abs(xd), abs(yd))))
  if n == 0:
    return

  xdd = xd/n
  ydd = yd/n

  for step in range(n):
    img[int(start[0]+step*ydd),int(start[1]+step*xdd)] = c

def alpha():
  global img


  img = img * 0.9

def paint_spectrum():
  global spectrum
  step = HALF_DIAG/n_data_bins
  for i, a in enumerate(spectrum):
    circle((yb, xb), (i+1)*step, c=a)

def draw():
  global spectrum
  spectrum = get_audio()
  print(np.argmax(spectrum))
  #paint_spectrum()
  calc_field()
  alpha()
  for p in particles:
    apply_force(p)
    move_particle(p)
    
# LOOP
while True:
  draw()
  cv2.imshow("noisflow", img)
  key = cv2.waitKey(delay=1)
  if key == 27:
    cv2.destroyAllWindows()
    break