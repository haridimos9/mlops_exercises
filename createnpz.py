from PIL import Image
import os
import numpy as np
import glob
import cv2

path_to_files = "./data/example_images/"    
array_of_images = []

# for _, file in enumerate(os.listdir(path_to_files)):
#     if "97.jpg" in file:
#         single_im = Image.open(path_to_files+file)
#         single_array = np.array(single_im)
#         array_of_images.append(single_array)            
# np.savez("image9.npz",array_of_images) # save all in one file


# imgs = np.load('data/example_images.npz')
# lst = imgs.files

# for i in lst:
#     print(i)
#     print(imgs[i].shape)

#imgs = glob.glob('data/example_images/*.jpg')
images = [cv2.imread(file) for file in glob.glob('data/example_images/*.jpg')]
print(images[0].shape)