import cv2
import numpy as np
import torch
import sys

def convert(filename, size):

    cap = cv2.VideoCapture(filename)
    size_str = str(size)
    size_str = size_str[1:3]+size_str[5:7]
    print("converting to size:",size_str)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    print("frame rate:", framerate)
    out = cv2.VideoWriter(filename[:-4]+size_str+filename[-4:], int(fourcc), framerate, size)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(frame, size)
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv)==2:
        convert(sys.argv[1], (30,30))
    else: 
        convert(sys.argv[1], (int(sys.argv[2]), int(sys.argv[3])))