
#from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import matplotlib 
from matplotlib import pyplot
from matplotlib.image import imread
import numpy as np
from keras.preprocessing.image import load_img
from numpy import load
from keras.preprocessing.image import img_to_array
import cv2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from PIL import Image

df = pd.read_csv('pokemonim.csv', header=0)
pokemon_names = df['Name']
#print (pokemon_names[1])
images=list()

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

for i in range(pokemon_names.size):
	filename='pokemonimages/'+pokemon_names[i]+'.jpg'
	x=Image.open(filename)
	#x=np.array(x)
	x = x.resize((128,128))

	x=remove_transparency(x)
	x=x.convert("RGB")

	#x.reshape(128,128,3)
	#pyplot.imshow(x)
	#pyplot.show()


	x= img_to_array(x, dtype='uint8')
	#x=x.reshape(x.shape[0],128,128,1
	#print(x.shape)
	images.append(x)
	#print(pokemon_names[i]+'done')

#pyplot.imshow(images[0])
#pyplot.show()
#print(images[0])


images=np.array(images)
print(images.shape)
images=images.astype('float32')
images=images/255

# pyplot.imshow(images[0])
# pyplot.show()



types = pd.read_csv('SparseTypes.csv', index_col=0)
ftypes = np.array(types)
#ftypes=ftypes[:100]
print (len(ftypes))
#print(ftypes[0])
X_train, X_test, Y_train, Y_test = train_test_split(images,ftypes, test_size=0.2, random_state=12345)

model = keras.models.Sequential()

# model.add(Conv2D(32, (3, 3), padding="same",input_shape=(128,128,3)))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))



#model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

# 		# (CONV => RELU) * 2 => POOL
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(Conv2D(32, (3, 3), padding="same"))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# 		# (CONV => RELU) * 2 => POOL
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# #model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
# #model.add(BatchNormalization())
#model.add(Dropout(0.5))

#model.add(Flatten())
model.add(Dense(18))
model.add(Activation("softmax"))
#model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(28, 28, 3)))
# model.add(Dropout(0.3))
# model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))     
# model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(Dense(18, activation='softmax'))   # Final Layer using Softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, batch_size=16)
print(score)


model.save("model.h5")









