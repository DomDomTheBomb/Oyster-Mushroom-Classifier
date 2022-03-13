# Purpose: Basically downsamples the images and saves them into 256 x 256 images
# ends up saving us time during the training process
# @author: Dominic Sobocinski

from skimage import io
from skimage import transform as trans
import os

from multiprocessing import Pool as P

oysters = "images/oyster/"
background = "images/background/"

def oysters_downsize(directory):
    image = io.imread(directory)
    image = trans.resize(image, (256, 256))
    io.imsave("preprocessed/oyster/" + os.path.split(directory)[1], image)

def background_downsize(directory):
    image = io.imread(directory)
    image = trans.resize(image, (256, 256))
    io.imsave("preprocessed/background/" + os.path.split(directory)[1], image)

dir = []

for file in os.listdir(oysters):
    dir.append(oysters + file)

# create a pool so we can multithread process
pool = P()
pool.map(oysters_downsize, dir)
pool.close()

dir = []

for file in os.listdir(background):
    dir.append(background + file)

pool = P()
pool.map(background_downsize, dir)
pool.close()
