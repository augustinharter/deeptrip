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

def logbinner(data, n_bins, min_bin=1):
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
n_data_bins = 50
      

# CV2
cv2.namedWindow('noisflow', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("noisflow",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
root = tk.Tk()
WIDTH = 1920#root.winfo_screenwidth()
HEIGHT = 1080#root.winfo_screenheight()
img = np.zeros((HEIGHT, WIDTH))


# FIELD 
fn = 50
fhscl = HEIGHT/fn
fwscl = WIDTH/fn
f = np.zeros((fn,fn,2))
z = 10
fs = 2/fn
fp = 2

class Dot:
  def __init__(self, y, x, vy, vx):
    self.y = y
    self.x = x
    self.vy = vy
    self.vx = vx

# PARTICLES
pn = 10
particles = [Dot(rnd()*HEIGHT, rnd()*WIDTH, 0, 0) for _ in range(pn)]
vmax = 25

spawn_norm = 3
maxi = 200000
space = WIDTH/8
baseline = int(HEIGHT/2)
y_spread = HEIGHT/(spawn_norm*4)
x_spread = 3*WIDTH/(4*n_data_bins)

def spawn(plot=True):
  global particles
  global maxi
  maxi *=0.95
  n_avlb = in_stream.get_read_available()

  data = np.average(chunk2array(in_stream.read(n_avlb), n_avlb), axis=1)
  four_data = np.fft.rfft(data, norm="ortho")[lower_limit:upper_limit]
  four_data = logbinner(four_data, n_data_bins, min_bin=2)

  if plot:
    plt.plot(four_data)
    plt.show()

  candidate = np.max(four_data)
  if candidate>maxi:
    maxi = candidate
  amp_data = spawn_norm*four_data/maxi
  
  length = len(particles)>1000
  for x, amp in enumerate(amp_data):
    for y in range(int(amp)):
      if length:
        particles.remove(particles[0])
        particles.remove(particles[0])
      xpos = space+x*x_spread
      ypos = 10+y*y_spread
      particles.append(Dot(baseline+ypos, xpos, 0, 0))
      particles.append(Dot(baseline-ypos, xpos, 0, 0))


def calc_field(draw=False):
  global z
  a_len = 40
  for i in range(fn):
    for j in range(fn):
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

def apply_force(dot):
  dot.vy += fp*f[int(dot.y/fhscl),int(dot.x/fwscl),0]
  if dot.vy>vmax: dot.vy = vmax
  if dot.vy<-vmax: dot.vy = -vmax
  dot.vx += fp*f[int(dot.y/fhscl),int(dot.x/fwscl),1]
  if dot.vx>vmax: dot.vx = vmax
  if dot.vx<-vmax: dot.vx = -vmax
  
def move_particle(dot):
  edge = False
  #img[int(dot.x), int(dot.y)] = 0
  start = (int(dot.y), int(dot.x))

  dot.y += dot.vy
  dot.x += dot.vx
  edge, dot.y, dot.x = check_edge(dot.y, dot.x)
  if edge: return False
  #dot.y = dot.y%HEIGHT
  #dot.y = dot.y%WIDTH
  #img[int(dot.x), int(dot.y)] = 1
  dest = (int(dot.y), int(dot.x))
  #new_point(start, dest)
  line(start, dest)
  return True

def new_point(start, dest):
  global img
  img[start] = 0
  img[dest] = 1

def line(start, dest):
  global img

  yd = dest[0]-start[0]
  xd = dest[1]-start[1]

  n = int(math.ceil(max(abs(xd), abs(yd))))
  if n == 0:
    return

  xdd = xd/n
  ydd = yd/n

  for step in range(n):
    img[int(start[0]+step*ydd),int(start[1]+step*xdd)] = 1

def alpha():
  global img


  img = img * 0.95

def draw():  
  calc_field()
  spawn(plot=False)
  alpha()
  for p in particles:
    apply_force(p)
    if not move_particle(p):
      particles.remove(p)
    
# LOOP
while True:
  draw()
  cv2.imshow("noisflow", img)
  key = cv2.waitKey(delay=1)
  if key == 27:
    cv2.destroyAllWindows()
    break