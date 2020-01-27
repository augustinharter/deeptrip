import noise
import numpy as np
import tkinter as tk
import cv2
import math
import pyaudio
import struct
from matplotlib import pyplot as plt
import colorsys

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
print(p.get_device_info_by_index)
in_stream = p.open(44100, 2, p.get_format_from_width(2), input=True, input_device_index=p.get_default_input_device_info()['index'])

max_freq = 3000
min_freq = 50
upper_limit = int(CHUNK*(max_freq/44100))
lower_limit = int(CHUNK*(min_freq/44100))
n_data_bins = 20

def get_audio(plot=False):
  global maxi
  maxi *=0.995
  if maxi<1: maxi = 1

  n_avlb = in_stream.get_read_available()
  data = np.average(chunk2array(in_stream.read(n_avlb), n_avlb), axis=1)
  four_data = np.fft.rfft(data, norm="ortho")[lower_limit:upper_limit]
  four_data = logbinner(four_data, n_data_bins, min_bin=3)

  if plot:
    plt.plot(four_data)
    plt.show()

  candidate = np.max(four_data)
  if candidate>maxi:
    maxi = candidate
  amp_data = spawn_norm*four_data/maxi
  return amp_data    

# CV2
cv2.namedWindow('noisflow', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("noisflow",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
root = tk.Tk()
WIDTH = 1920#root.winfo_screenwidth()
HEIGHT = 1080#root.winfo_screenheight()
img = np.zeros((HEIGHT, WIDTH))


# FIELD 
SPRING = False
fn = 50
fhscl = HEIGHT/fn
fwscl = WIDTH/fn
f = np.zeros((fn,fn,2))
z = 10
fs = 2/fn
field_power = 2
border_power = 1

class Dot:
  def __init__(self, y, x, vy, vx, r=1.0, vr=0):
    self.y = y
    self.x = x
    self.vy = vy
    self.vx = vx
    self.r = r
    self.vr = r

  def gety(self):
    return self.y%HEIGHT

  def getx(self):
    return self.x%WIDTH

# PARTICLES
n_max_part = 500
pn = 10
particles = []
vmax = 50
spring_strength = 0.005
spring_stiffness = 0.8 # math.pow(0.02, 1/n_data_bins)
spring_norm = 2000/n_data_bins
grow_rate = 30
degrowth = 0.7
friction = 0.95

spawn_norm = 100
maxi = 200000
space = WIDTH/8
baseline = int(HEIGHT/2)
y_spread = HEIGHT/(spawn_norm*10)
x_spread = 3*WIDTH/(4*n_data_bins)

def spawn_spring():
  global SPRING
  SPRING = True
  for x in range(n_data_bins):
    xpos = space+x*x_spread
    ypos = (baseline)*math.sin(x*math.pi/n_data_bins)
    particles.append(Dot(baseline/2+ypos, xpos, 0, 0))

def calc_spring():
  apply_spring()
  amp_data = get_audio()
  for i in range(n_data_bins):
    p = particles[i]
    apply_force(p)
    p.r *= degrowth
    p.r += grow_rate * (amp_data[i]/spawn_norm)
    p.r = max(min(spawn_norm, p.r), 1)
    move_particle(p)

def apply_friction():
  for p in particles:
    p.vx *= friction
    p.vy *= friction

def apply_spring():
  global particles
  global spring_strength
  global spring_stiffness
  for i in range(n_data_bins):
    p = particles[i]
    for j in range(i, n_data_bins):
      if i==j: continue
      d = abs(i-j)
      q = particles[j]
      dx = q.x -p.x
      dy = q.y -p.y
      sx = np.sign(dx)
      sy = np.sign(dy)
      dd = math.pow((dx**2) + (dy**2), 0.5) - d*spring_norm
      dd *= math.pow(spring_stiffness, (d-1))
      nx = np.abs(dx)/(np.abs(dx)+np.abs(dy))
      ny = 1-nx
      dx = sx*dd*nx
      dy = sy*dd*ny
      p.vx += dx*spring_strength
      q.vx -= dx*spring_strength
      p.vy += dy*spring_strength
      q.vy -= dy*spring_strength
    

def spawn_eye():
  global particles

  amp_data = get_audio()
  
  length = len(particles)>n_max_part
  for x, amp in enumerate(amp_data):
    for y in range(int(amp)):
      if length:
        particles.remove(particles[0])
        particles.remove(particles[0])
      xpos = space+x*x_spread
      ypos = 10+y*y_spread + (baseline/2)*math.sin(x*math.pi/n_data_bins)
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

xb = WIDTH/2
yb = HEIGHT/2

def apply_force(dot):
  global xb, yb
  xd = (xb-dot.getx())/xb
  yd = (yb-dot.gety())/yb
  base = 2
  bf = (base/2)
  xf = bf*math.pow(base, abs(xd))-bf
  yf = bf*math.pow(base, abs(yd))-bf
  xif = 1-xf
  yif = 1-yf

  dot.vy += yif*field_power*f[int(dot.gety()/fhscl),int(dot.getx()/fwscl),0]
  dot.vy += border_power*yf*np.sign(yd)
  if dot.vy>vmax: dot.vy = vmax
  if dot.vy<-vmax: dot.vy = -vmax
  dot.vx += xif*field_power*f[int(dot.gety()/fhscl),int(dot.getx()/fwscl),1]
  dot.vx += border_power*xf*np.sign(xd)
  if dot.vx>vmax: dot.vx = vmax
  if dot.vx<-vmax: dot.vx = -vmax
  
def move_particle(dot):
  start = (int(dot.gety()), int(dot.gety()))
  dot.y += dot.vy
  dot.x += dot.vx
  dest = (int(dot.gety()), int(dot.getx()))
  circle(dest, dot.r)

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

def point(y, x):
  img[y,x] = 1

def circle(pos, r):
  yy, xx = pos
  circ = round(math.pi*r*2)
  step = 1/r
  for i in range(int(circ)):
    y = round(yy + math.sin(i*step) * r)
    x = round(xx + math.cos(i*step) * r)
    edge, *_ = check_edge(y, x)
    if not edge: point(int(y), int(x))



def alpha():
  global img


  img = img * 0.90

def draw():  
  calc_field()
  alpha()
  calc_spring()
  apply_friction()


# LOOP
spawn_spring()
while True:
  draw()
  #cv2.imshow("noisflow", cv2.resize(img,(53,30)))
  cv2.imshow("noisflow", img)
  key = cv2.waitKey(delay=1)
  if key == 27:
    cv2.destroyAllWindows()
    break