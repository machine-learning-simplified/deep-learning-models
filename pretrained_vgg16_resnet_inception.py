import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

mobile = keras.applications.mobilenet.MobileNet()

from IPython.display import Image
Image(filename='1.PNG', width=300,height=200)


def prepare_image(file):
    img_path = 'img/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('1.PNG')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results


#inception_resnet_v2
model = keras.applications.inception_resnet_v2.InceptionResNetV2()

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


from keras.applications import imagenet_utils



#InceptionV3
model = tf.keras.applications.inception_v3.InceptionV3()

img_path = r'img\1.png'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = keras.applications.inception_v3.preprocess_input(x)

preds = model.predict(x)
print('Predicted:', imagenet_utils.decode_predictions(preds))


#resnet50
model = keras.applications.resnet50.ResNet50()

#model = ResNet50(include_top=True, weights='imagenet')

img_path = r'img\1.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.resnet50.preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', imagenet_utils.decode_predictions(preds))




import tensorflow as tf
model = tf.keras.applications.vgg16.VGG16()
#model = VGG16(include_top=True, weights='imagenet')
img_path = r'img\1.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.vgg16.preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', imagenet_utils.decode_predictions(preds))







