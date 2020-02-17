import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import csv 
import sys
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import img_to_array


img = cv2.imread('pokemonimages/Groudon.jpg',cv2.COLOR_BGR2RGB)
print (img.shape)
im = Image.open("pokemonimages/Groudon.jpg")
im1 = im.resize((200,200))
#im1= img_to_array(im1, dtype='uint8')
print(im1)



def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

y=remove_transparency(im1)

y=y.convert("RGB")
print("rgb")
y.show()
y= img_to_array(y, dtype='uint8')
print(y.shape)



#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)




mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)
height, width = img.shape[:2]

rect = (0,0,width-10,height-10)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
imgnew= img*mask2[:,:,np.newaxis]
background=img-imgnew
background[np.where((background>[0,0,0]).all(axis=2))]=[255,255,255]

final=background+imgnew
#print mask2

#plt.imshow(final)
#plt.show()