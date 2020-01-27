import tkinter as tk
import PIL
import numpy as np
from matplotlib import pyplot as plt
import noise
import cv2

nois = noise.snoise3
grid = np.linspace(0,2,30)
z = 0

def world_builder():
  z = 0
  while True:
    z+= 0.01
    base = 100 #*np.random.rand()
    world = np.array([[0.5*nois(x,y,z, octaves=3)+0.5
      for x in grid] for y in grid])
    new_world = cv2.resize(world, (500 , 500))
    cv2.imshow("nois!", new_world)
    cv2.waitKey()

def walker():
  x, y = (50.0,50.0)
  d = 0
  world = np.zeros((100,100))
  while True:
    world[int(y), int(x)] = 0.0
    x += nois(d,0,0)
    y += nois(0,d,0)
    x = x%100
    y = y%100
    world[int(y), int(x)] = 1.0
    cv2.imshow("walk!", world)
    cv2.waitKey()
    d += 0.01
  
def homer():
  x, y = (50.0,50.0)
  xmean, ymean = (0,0)
  d = 0
 
  while True:
    x += nois(d,0,0)
    y += nois(0,d,0)
    d += 0.01
    xmean += x
    ymean += y
    if (int(d*100)%100000) ==0:
      print(x, y, xmean, ymean)

world_builder()

root = tk.Tk()
root.attributes('-fullscreen', True)
root.bind('<Escape>',lambda e: root.destroy())
WIDTH = root.winfo_screenwidth()
HEIGHT = root.winfo_screenheight()
w = tk.Canvas(root, width=WIDTH, height=HEIGHT)
w.configure(background='black')
print(w.keys())
w.pack()
img = tk.PhotoImage(width=WIDTH, height=HEIGHT)
w.create_image((WIDTH/2, HEIGHT/2), image=img, state="normal")

for x in range(4 * WIDTH):
    y = int(HEIGHT/2 + HEIGHT/4 * np.sin(x/80.0))
    img.put("#ffffff", (x//4,y))

w.mainloop() 
