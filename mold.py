import noise
import numpy as np
import tkinter as tk

# TKINTER
root = tk.Tk()
root.attributes('-fullscreen', True)
root.bind('<Escape>',lambda e: root.destroy())
WIDTH = root.winfo_screenwidth()
HEIGHT = root.winfo_screenheight()
w = tk.Canvas(root, width=WIDTH, height=HEIGHT)
w.configure(background='black')
w.pack()
img = tk.PhotoImage(width=WIDTH, height=HEIGHT)
w.create_image((WIDTH/2, HEIGHT/2), image=img, state="normal")


# FIELD 
fn = 100
fscl = WIDTH/fn
f = np.zeros((fn,fn,2))
z = 10
fs = 0.01

# PARTICLES
pn = 100
pp = np.abs(np.random.rand(pn,2))*min(WIDTH, HEIGHT)
pv = np.ones((pn,2))
vmax = 20

def p_calc():
  global z
  for i in range(fn):
    for j in range(fn):
      f[i,j,0] = noise.snoise3(j*fs,i*fs,z)
      f[i,j,1] = noise.snoise3(j*fs,i*fs,z+100)
  z+=0.001

def p_apply(i):
  pv[i,0] += f[int(pp[i,0]/fscl),int(pp[i,1]/fscl),0]
  if pv[i,0]>vmax: pv[i,0] = vmax
  if pv[i,0]<-vmax: pv[i,0] = -vmax
  pv[i,1] += f[int(pp[i,0]/fscl),int(pp[i,1]/fscl),1]
  if pv[i,1]>vmax: pv[i,1] = vmax
  if pv[i,1]<-vmax: pv[i,1] = -vmax
  
def p_mv(i):
  img.put('#000000', (int(pp[i,1]), int(pp[i,0])))
  pp[i,0] += pv[i,0]
  pp[i,1] += pv[i,1]

  if pp[i,0] < 0: pp[i,0] = HEIGHT-0.0001
  if pp[i,0] >= HEIGHT: pp[i,0] = 0
  if pp[i,1] < 0: pp[i,1] = WIDTH-0.0001
  if pp[i,1] >= WIDTH: pp[i,1] = 0

  img.put('#ffffff', (int(pp[i,1]), int(pp[i,0])))

def draw():  
  while True:
    p_calc()
    for i in range(pn):
      p_apply(i)
      p_mv(i)
    w.update()
    
  #root.after(10, draw)

root.after(10,draw())
root.mainloop()