import urllib
import time
import os
import sys
import serial
import signal
import threading
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.image as mpimg
from pygame.locals import *

debug = sys.argv[1]

if not debug:
   PORT = '/dev/rfcomm1'
   SPEED = 115200

   ser = serial.Serial(PORT)

anTargets = 0
nTargets = 0
status = False
move = None
targets = []
inputs = []
sync = True
n = 0

import pygame
pygame.init()
w = 288
h = 352
size=(w,h)
screen = pygame.display.set_mode(size)
c = pygame.time.Clock() # create a clock object for timing
pygame.display.set_caption('Driver')
ubuntu = pygame.font.match_font('Ubuntu')
font = pygame.font.Font(ubuntu, 13)



class getKeyPress(threading.Thread):
    def run(self):      
        import pygame
        pygame.init()
        global targets
        global status
        global nTargets
        global anTargets
        while not status:
               pygame.event.pump()
               keys = pygame.key.get_pressed()
               targets, status = processOutputs(keys, targets)
               #targets = targets[n:n+1]
               #indices = n+1, -1
               #targets = [i for j, i in enumerate(targets) if j not in indices]
               nTargets = len(targets)
               if status:
                  break
        targets = np.array(targets)
        targets = flattenMatrix(targets)
        sio.savemat('targets.mat', {'targets':targets})      


def rgb2gray(rgb):
    r, g, b = np.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b

def processImages(inputX, inputs):
    inputX = flattenMatrix(inputX)
    if len(inputs) == 0:
       inputs = inputX
    elif inputs.shape[1] >= 1:
       inputs = np.hstack((inputs, inputX))
    return inputs

def flattenMatrix(mat):
    mat = mat.flatten(1)
    mat = mat.reshape((len(mat), 1))
    return mat

def send_command(val):
    connection = serial.Serial( PORT,
                                SPEED,
                                timeout=0,
                                stopbits=serial.STOPBITS_TWO
                                )
    connection.write(val)
    connection.close()

def processOutputs(keys, targets):
    label = []
    global move
    global n
    global status
    global sync
    global nTargets
    status = False
    keypress = [K_p, K_UP, K_LEFT, K_DOWN, K_RIGHT]
    labels = [1, 2, 3, 4, 5]
    commands = ['p', 'w', 'r', 'j', 's']
    text = ['S', 'Up', 'Left', 'Down', 'Right']
    if keys[K_q]:
       status = True
       return targets, status            
    else:
       for i, j, k, g in zip(keypress, labels, commands, text):
           if keys[i]:
              move = g
              label.append(j)  
              if not debug:  
                 send_command(k)
    if len(label) != 0:
       targets.append(label[:])
       targets[n+1:] = []
    else:
       send_command('p')
    return targets, status


gkp = getKeyPress()
gkp.start()

if __name__ == '__main__':
   inputs = []
   try:
     while not status:
           urllib.urlretrieve("http://192.168.0.10:8080/shot.jpg", "input.jpg")
           try:
             inputX = mpimg.imread('input.jpg')
           except IOError:
             status = True
           inputX = rgb2gray(inputX)/255
           out = inputX.copy()
           out = scipy.misc.imresize(out.T, (352, 288), interp='bicubic', mode=None)
           scipy.misc.imsave('input.png', out)
           if nTargets != anTargets:
              inputs = processImages(inputX, inputs)
              print "Images:", inputs.shape[1], "Labels:", nTargets
              anTargets = nTargets
              n += 1
           img=pygame.image.load('input.png')
           screen.blit(img,(0,0))
           pygame.display.flip() 
           c.tick(3)
           if move != None:
              text = font.render(move, False, (255, 255, 0))
              textRect = text.get_rect()
              textRect.centerx = 20 #screen.get_rect().centerx
              textRect.centery = 20 #screen.get_rect().centery
              screen.blit(text, textRect)
              pygame.display.update()
           if status:
              sio.savemat('inputs.mat', {'inputs':inputs})
   except KeyboardInterrupt:
     sio.savemat('inputs.mat', {'inputs':inputs})

sio.savemat('inputs.mat', {'inputs':inputs})
