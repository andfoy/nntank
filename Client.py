import urllib
import time
import os
import sys
import serial
import signal
import pygame
from multiprocessing import Process
import numpy as np
import scipy
import scipy.io as sio
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pygame.locals import *


PORT = '/dev/rfcomm0'
SPEED = 115200

ser = serial.Serial(PORT)

pygame.init()
w = 288
h = 352
size=(w,h)
screen = pygame.display.set_mode(size) 
c = pygame.time.Clock() # create a clock object for timing
pygame.display.set_caption('Driver')
ubuntu = pygame.font.match_font('Ubuntu')
font = pygame.font.Font(ubuntu, 13)

'''#def getTrainingSet():
    #targets = []
    #inputs = []
    #win=curses.initscr()
    while True:
          inputX = getImage()
          inputX = processImages(inputX, inputs)
          #result = win.getkey()
          #targets = processKeyPress(result, targets)
          #inputs = processImages(inputX, inputs)
          #if targets[-1] == 0:
             #break
    #curses.endwin()
    sio.savemat('inputs.mat', {'inputs':inputs})
    targets = np.array(targets)
    sio.savemat('targets.mat', {'targets':targets}) '''


#def Driver():
    #status = False
    #p = Process(target=getKeyPress, args=())
    #p2 = Process(target=getImage, args=())
    #p.start()
    #p2.start()
    #targets = p.get()
    #inputs = p2.get()
    #p.join()
    #p2.join()
    #targets = np.array(targets)
    #targets = flattenMatrix(targets)
    #sio.savemat('inputs.mat', {'inputs':inputs})
    #sio.savemat('targets.mat', {'targets':targets})



def getKeyPress():
    while not status:
          pygame.event.pump()
          keys = pygame.key.get_pressed()
          #print keys
          targets, status = processOutputs(targets, keys)
    return targets

def getImage():
    x = 0
    targets = []
    inputs = []
    while True:
          urllib.urlretrieve("http://192.168.0.10:8080/shot.jpg", "input.jpg")
          inputX = mpimg.imread('input.jpg')
          inputX = rgb2gray(inputX)/255
          out = inputX.copy()
          out = scipy.misc.imresize(out, (352, 288), interp='bicubic', mode=None)
          scipy.misc.imsave('input.png', out)
          inputs = processImages(inputX, inputs)
          #plt.imshow(inputX, cmap = plt.get_cmap('gray'))
          #plt.ion()
          #plt.show()
          ###plt.draw()
          img=pygame.image.load('input.png')
          screen.blit(img,(0,0))
          pygame.display.flip() # update the display
          c.tick(3) # only three images per second
          pygame.event.pump()
          keys = pygame.key.get_pressed()
          #print keys
          targets, status = processOutputs(targets, keys)
          if status:
             break
          else:
             x += 1
    targets = np.array(targets)
    targets = flattenMatrix(targets)
    sio.savemat('inputs.mat', {'inputs':inputs})
    sio.savemat('targets.mat', {'targets':targets})
    return inputs, targets

def rgb2gray(rgb):
    r, g, b = np.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b

def processKeyPress(result, targets):
    keyPress = ['w', 'a', 'd', 's', 'q']
    instruct = [1, 2, 3, 4, 0]    
    for i in range(0, len(keyPress)):
        key = keyPress[i]
        target = instruct[i]
        if str(result) == key:
           targets.append(target)
    return targets

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

def processOutputs(targets, keys):
    send_command('p')
    global status
    status = False
    #key = raw_input('Label: ')
    keypress = ['K_p', 'K_UP', 'K_LEFT', 'K_DOWN', 'K_RIGHT'] 
    labels = [1, 2, 3, 4, 5]
    commands = ['p', 'w', 'r', 'j', 's']
    text = ['S', 'Up', 'Left', 'Down', 'Right']
    if not keys[K_q]:
       for i, j, k, g in zip(keypress, labels, commands, text):
           move = compile('cond = keys['+i+']', '<string>', 'exec')
           exec move
           if cond:
              displayMode(g)
              targets.append(j)
              send_command(k)
              
    else:
       status = True
    #time.sleep(0.5)       
    return targets, status
 
def displayMode(mode):
    text = font.render(mode, False, (255, 0, 255), (0, 0, 0))
    textRect = text.get_rect()
    #print textRect
    textRect.centerx = 20 #screen.get_rect().centerx
    textRect.centery = 20 #screen.get_rect().centery
    screen.blit(text, textRect)
    pygame.display.update()   
