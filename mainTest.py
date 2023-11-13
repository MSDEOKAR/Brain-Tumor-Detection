import cv2
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from PIL import Image

model=load_model('BrainTumor10EpochsCategorical.keras')
image =cv2.imread('D:\\Brain Tumor\\pred\\pred23.jpg')

img=Image.fromarray(image)
img=img.resize((64,64))

img=np.array(img)
input_img=np.expand_dims(img,axis=0)
result=model.predict(input_img)
predicted_classes=np.argmax(result,axis=-1)
print(predicted_classes)