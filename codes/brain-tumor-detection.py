#importing libraries 
import numpy as np  
import cv2 
import os 
import shutil 
import itertools 
from sklearn.utils import shuffle 
import imutils 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode, iplot 
from plotly import tools 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras import layers 
from tensorflow.keras.models import Model, Sequential, load_model 
from tensorflow.keras.optimizers import Adam, RMSprop 
from tensorflow.keras.callbacks import EarlyStopping 
#Mounting google drive 
from google.colab import drive 
drive.mount('/content/drive') 
TRAIN_DIR = BASE_DIR + 'TRAIN/' 
VAL_DIR = BASE_DIR + 'VAL/' 
TEST_DIR = BASE_DIR + 'TEST/' 
import os 
root_path = "/content/" 
BASE_DIR=os.path.join(os.path.dirname(root_path), '/content/drive/MyDrive/brain tumor/') 
#Data import and processing 
def load_data(dir_path, img_size=(100,100)): 
X = [] 
y = [] 
for yesorno in (sorted(os.listdir(dir_path))): 
for imgname in os.listdir(dir_path + yesorno): 
img = cv2.imread(dir_path + yesorno + '/' + imgname) 
X.append(img) 
if yesorno=='NO': 
y.append(0) 
else: 
y.append(1) 
X = np.array(X) 
y = np.array(y) 
print(f'{len(X)} images loaded from {dir_path} directory.') 
return X, y 
def plot_confusion_matrix(cm, classes, 
normalize=False, 
title='Confusion matrix', 
cmap=plt.cm.Blues): 
thresh = cm.max() / 2. 
cm = np.round(cm,2) 
plt.figure(figsize = (6,6)) 
plt.imshow(cm, interpolation='nearest', cmap=cmap) 
plt.title(title) 
plt.colorbar() 
tick_marks = np.arange(len(classes)) 
plt.xticks(tick_marks, classes, rotation=90) 
plt.yticks(tick_marks, classes) 
if normalize: 
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black") 
 
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 
    plt.show() 
 
# use predefined function to load the image data into workspace 
IMG_SIZE = (224,224) 
X_train, y_train = load_data(TRAIN_DIR, IMG_SIZE) 
X_test, y_test = load_data(TEST_DIR, IMG_SIZE) 
X_val, y_val = load_data(VAL_DIR, IMG_SIZE) 
 
y_val 
 
#Plotting the histogram of ratio distributions 
RATIO_LIST = [] 
for set in (X_train, X_test, X_val): 
    for img in set: 
        RATIO_LIST.append(img.shape[1]/img.shape[0]) 
         
plt.hist(RATIO_LIST) 
 
plt.title('Distribution of Image Ratios') 
plt.xlabel('Ratio Value') 
plt.ylabel('Count') 
plt.show() 
 
#First step of normalization to crop the brain out of the images 
def crop_imgs(set_name, add_pixels_value=0): 
    """ 
    Finds the extreme points on the image and crops the rectangular out of them 
    """ 
    set_new = [] 
    for img in set_name: 
       gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
 
        gray = cv2.GaussianBlur(gray, (5, 5), 0) 
 
        # threshold the image, then perform a series of erosions + 
        # dilations to remove any small regions of noise 
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1] 
        thresh = cv2.erode(thresh, None, iterations=2) 
        thresh = cv2.dilate(thresh, None, iterations=2) 
 
        # find contours in thresholded image, then grab the largest one 
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = imutils.grab_contours(cnts) 
        c = max(cnts, key=cv2.contourArea) 
 
        # find the extreme points 
        extLeft = tuple(c[c[:, :, 0].argmin()][0]) 
        extRight = tuple(c[c[:, :, 0].argmax()][0]) 
        extTop = tuple(c[c[:, :, 1].argmin()][0]) 
        extBot = tuple(c[c[:, :, 1].argmax()][0]) 
 
        ADD_PIXELS = add_pixels_value 
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]
ADD_PIXELS:extRight[0]+ADD_PIXELS].copy() 
        set_new.append(new_img) 
 
    return np.array(set_new) 
 
#Function to scan mri reports 
img = cv2.imread('brain_tumor_dataset/yes/Y108.jpg') 
img = cv2.resize( 
            img, 
            dsize=IMG_SIZE, 
            interpolation=cv2.INTER_CUBIC 
        ) 
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
gray = cv2.GaussianBlur(gray, (5, 5), 0) 
# threshold the image, then perform a series of erosions + 
 
# dilations to remove any small regions of noise 
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1] 
thresh = cv2.erode(thresh, None, iterations=2) 
thresh = cv2.dilate(thresh, None, iterations=2) 
# find contours in thresholded image, then grab the largest one 
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts) 
c = max(cnts, key=cv2.contourArea) 
# find the extreme points 
extLeft = tuple(c[c[:, :, 0].argmin()][0]) 
extRight = tuple(c[c[:, :, 0].argmax()][0]) 
extTop = tuple(c[c[:, :, 1].argmin()][0]) 
extBot = tuple(c[c[:, :, 1].argmax()][0]) 
# add contour on the image 
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4) 
# add extreme points 
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1) 
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1) 
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1) 
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1) 
i = 100 
plt.subplot(121) 
# crop 
ADD_PIXELS = 0 
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]
ADD_PIXELS:extRight[0]+ADD_PIXELS].copy() 
# apply this for each set 
X_train_crop = crop_imgs(set_name=X_train) 
X_val_crop = crop_imgs(set_name=X_val) 
X_test_crop = crop_imgs(set_name=X_test) 
 
plt.imshow(X_train[i]) 
 
plt.subplot(122) 
plt.imshow(X_train_crop[i]) 
 
def save_new_images(x_set, y_set, folder_name): 
    i = 0 
    for (img, imclass) in zip(x_set, y_set): 
        if imclass == 0: 
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img) 
 
        else: 
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img) 
        i += 1 
 
# saving new images to the folder 
 
save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/') 
save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/') 
save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/') 
 
a = BASE_DIR + 'TRAIN_CROP/YES' 
b = BASE_DIR + 'TRAIN_CROP/NO' 
c = BASE_DIR + 'VAL_CROP/YES' 
d = BASE_DIR + 'TRAIN_CROP/NO' 
e = BASE_DIR + 'TEST_CROP/YES' 
f = BASE_DIR + 'TRAIN_CROP/NO' 
 
def preprocess_imgs(set_name, img_size): 
    """ 
    Resize and apply VGG-15 preprocessing 
    """ 
    set_new = [] 
    for img in set_name: 
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC) 
        set_new.append(preprocess_input(img)) 
         return np.array(set_new) 
 
X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE) 
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE) 
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE) 
 
X_train_prep,y_train = shuffle(X_train_prep,y_train,random_state=0) 
X_test_prep,y_test = shuffle(X_test_prep,y_test,random_state=0) 
X_val_prep,y_val = shuffle(X_val_prep,y_val,random_state=0) 
 
y_train 
 
# set the paramters we want to change randomly 
demo_datagen = ImageDataGenerator( 
    rotation_range=15, 
    width_shift_range=0.05, 
    height_shift_range=0.05, 
    rescale=1./255, 
    shear_range=0.05, 
    brightness_range=[0.1, 1.5], 
    horizontal_flip=True, 
    vertical_flip=True 
) 
 
x = X_train_crop[0]   
x = x.reshape((1,) + x.shape)  
 
i = 0 
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', 
save_format='jpg'): 
    i += 1 
    if i > 20: 
        break 
 
plt.imshow(X_train_crop[0]) 
plt.xticks([]) 
plt.yticks([]) 
plt.title('Original Image') 
plt.show() 
plt.figure(figsize=(15,6)) 
i = 1 
for img in os.listdir('preview/'): 
img = cv2.imread('preview/' + img) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
plt.subplot(3,7,i) 
plt.imshow(img) 
plt.xticks([]) 
plt.yticks([]) 
i += 1 
if i > 3*7: 
break 
plt.suptitle('Augemented Images') 
plt.show() 
TRAIN_DIR = BASE_DIR + 'TRAIN_CROP/' 
VAL_DIR = BASE_DIR + 'VAL_CROP/' 
train_datagen = ImageDataGenerator( 
rotation_range=15, 
width_shift_range=0.1, 
height_shift_range=0.1, 
shear_range=0.1, 
brightness_range=[0.5, 1.5], 
horizontal_flip=True, 
vertical_flip=True, 
preprocessing_function=preprocess_input 
) 
TRAIN_DIR, 
color_mode='rgb', 
test_datagen = ImageDataGenerator( 
preprocessing_function=preprocess_input 
) 
train_generator = train_datagen.flow_from_directory( 
 
    target_size=IMG_SIZE, 
    batch_size=1, 
    class_mode='binary' 
) 
 
validation_generator = test_datagen.flow_from_directory( 
    VAL_DIR, 
    color_mode='rgb', 
    target_size=IMG_SIZE, 
    batch_size=1, 
    class_mode='binary', 
) 
 
train_generator.samples 
 
# load base model 
vgg16_weight_path = BASE_DIR + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' 
base_model = VGG16( 
    weights=vgg16_weight_path, 
    include_top=False,  
    input_shape=IMG_SIZE + (3,) 
) 
 
model = Sequential() 
model.add(base_model) 
model.add(layers.Flatten()) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(1, activation='sigmoid')) 
 
model.layers[0].trainable = False 
model.compile( 
    loss='binary_crossentropy', 
    optimizer=RMSprop(lr=1e-4), 
  metrics=['accuracy'] 
) 
model.summary() 
 
 
#Training of our model 
EPOCHS = 30 
es = EarlyStopping( 
    monitor='val_accuracy',  
    mode='max', 
    patience=6 
) 
 
history = model.fit_generator( 
    train_generator, 
    steps_per_epoch = 193, 
    epochs=EPOCHS, 
    validation_data=validation_generator, 
    validation_steps=50, 
    callbacks=[es], 
) 
model.save('braintumor.h5') 