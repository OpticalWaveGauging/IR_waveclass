## CAM.py 
## A script to carry out class activation mapping on some images of each class
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe.nau.edu

import numpy as np
import ast
import scipy   
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing import image    
from tensorflow.python.keras.models import Model   
import sys, glob, os
import pandas as pd

def pretrained_path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
	
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
	
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
	
    # convert RGB -> BGR, subtract mean ImageNet pixel, and return 4D tensor
    return preprocess_input(x)

def get_model():
    # define ResNet50 model
    model = ResNet50(weights='imagenet')
	
    # get AMP layer weights
    amp_weights = model.layers[-1].get_weights()[0]
	
    # extract wanted output
    model = Model(inputs=model.input, 
        outputs=(model.layers[-4].output, model.layers[-1].output)) 
    return model, amp_weights
    
def get_CAM(img_path, model, amp_weights):
    # get filtered images from convolutional output + model prediction vector
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
	
    # change dimensions of last convolutional outpu tto 7 x 7 x 2048
    last_conv_output = np.squeeze(last_conv_output) 
	
    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec)
	
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
	
    # get AMP layer weights
    amp_layer_weights = amp_weights[:, pred] # dim: (2048,) 
	
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224,224) # dim: 224 x 224
    return final_output, pred

#==============================================================	
if __name__ == '__main__':
	model, amp_weights = get_model()

	print ("[INFO] computing activation maps ...")
	
    use = 5	
    images = glob.glob(os.getcwd()+os.sep+'test\spill\*.jpg')[:use]
    C1 = []; I1 = []
    for img_path in images:
       CAM, pred = get_CAM(img_path, model, amp_weights)
       C1.append(CAM)
       I1.append(imread(img_path))	
	
    images = glob.glob(os.getcwd()+os.sep+'test\plunge\*.jpg')[:use]
    C2 = []; I2 = []
    for img_path in images:
       CAM, pred = get_CAM(img_path, model, amp_weights)
       C2.append(CAM)
       I2.append(imread(img_path))

    images = glob.glob(os.getcwd()+os.sep+'test\nonbreaking\*.jpg')[:use]
    C3 = []; I3 = []
    for img_path in images:
       CAM, pred = get_CAM(img_path, model, amp_weights)
       C3.append(CAM)
       I3.append(imread(img_path))

    levels = [0,.001,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    origin='upper'

    ax=plt.subplot(251)
    ax.set_aspect('equal')
    plt.imshow(I1[0], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('A) Spilling 1', fontsize=6, loc='left')

    ax=plt.subplot(252)
    ax.set_aspect('equal')
    plt.imshow(I1[1], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('B) Spilling 2', fontsize=6, loc='left')

    ax=plt.subplot(253)
    ax.set_aspect('equal')
    plt.imshow(I1[2], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('C) Spilling 3', fontsize=6, loc='left')

    ax=plt.subplot(254)
    ax.set_aspect('equal')
    plt.imshow(I1[3], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('D) Spilling 4', fontsize=6, loc='left')

    ax=plt.subplot(255)
    ax.set_aspect('equal')
    plt.imshow(I1[4], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('E) Spilling 5', fontsize=6, loc='left')

    ax=plt.subplot(256)
    ax.set_aspect('equal')
    plt.contourf(C1[0]/np.max(C1[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C1[0]/np.max(C1[0]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(257)
    ax.set_aspect('equal')
    plt.contourf(C1[1]/np.max(C1[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C1[1]/np.max(C1[1]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(258)
    ax.set_aspect('equal')
    plt.contourf(C1[2]/np.max(C1[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C1[2]/np.max(C1[2]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(259)
    ax.set_aspect('equal')
    plt.contourf(C1[3]/np.max(C1[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C1[3]/np.max(C1[3]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(2,5,10)
    ax.set_aspect('equal')
    plt.contourf(C1[4]/np.max(C1[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C1[4]/np.max(C1[4]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    #plt.tight_layout()
    plt.savefig('av_actmap_spill.png', bbox_inches='tight', dpi=300)
    plt.close()   	
	print ("[INFO] spilling waves activation maps plotted ...")


    plt.close('all')

    ax=plt.subplot(251)
    ax.set_aspect('equal')
    plt.imshow(I2[0], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('F) Plunging 1', fontsize=6, loc='left')

    ax=plt.subplot(252)
    ax.set_aspect('equal')
    plt.imshow(I2[1], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('G) Plunging 2', fontsize=6, loc='left')

    ax=plt.subplot(253)
    ax.set_aspect('equal')
    plt.imshow(I2[2], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('H) Plunging 3', fontsize=6, loc='left')

    ax=plt.subplot(254)
    ax.set_aspect('equal')
    plt.imshow(I2[3], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('I) Plunging 4', fontsize=6, loc='left')

    ax=plt.subplot(255)
    ax.set_aspect('equal')
    plt.imshow(I2[4], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('J) Plunging 5', fontsize=6, loc='left')

    ax=plt.subplot(256)
    ax.set_aspect('equal')
    plt.contourf(C2[0]/np.max(C2[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C2[0]/np.max(C2[0]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(257)
    ax.set_aspect('equal')
    plt.contourf(C2[1]/np.max(C2[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C2[1]/np.max(C2[1]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(258)
    ax.set_aspect('equal')
    plt.contourf(C2[2]/np.max(C2[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C2[2]/np.max(C2[2]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(259)
    ax.set_aspect('equal')
    plt.contourf(C2[3]/np.max(C2[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C2[3]/np.max(C2[3]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(2,5,10)
    ax.set_aspect('equal')
    plt.contourf(C2[4]/np.max(C2[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C2[4]/np.max(C2[4]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    #plt.tight_layout()
    plt.savefig('av_actmap_plunge.png', bbox_inches='tight', dpi=300)
    plt.close()   
	print ("[INFO] plunging waves activation maps plotted ...")


    plt.close('all')

    ax=plt.subplot(251)
    ax.set_aspect('equal')
    plt.imshow(I3[0], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('K) Unbroken 1', fontsize=6, loc='left')

    ax=plt.subplot(252)
    ax.set_aspect('equal')
    plt.imshow(I3[1], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('L) Unbroken 2', fontsize=6, loc='left')

    ax=plt.subplot(253)
    ax.set_aspect('equal')
    plt.imshow(I3[2], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('M) Unbroken 3', fontsize=6, loc='left')

    ax=plt.subplot(254)
    ax.set_aspect('equal')
    plt.imshow(I3[3], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('N) Unbroken 4', fontsize=6, loc='left')

    ax=plt.subplot(255)
    ax.set_aspect('equal')
    plt.imshow(I3[4], cmap='gray', origin=origin)
    plt.axis('off')
    plt.title('O) Unbroken 5', fontsize=6, loc='left')

    ax=plt.subplot(256)
    ax.set_aspect('equal')
    plt.contourf(C3[0]/np.max(C3[0]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C3[0]/np.max(C3[0]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(257)
    ax.set_aspect('equal')
    plt.contourf(C3[1]/np.max(C3[1]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C3[1]/np.max(C3[1]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(258)
    ax.set_aspect('equal')
    plt.contourf(C3[2]/np.max(C3[2]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C3[2]/np.max(C3[2]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(259)
    ax.set_aspect('equal')
    plt.contourf(C3[3]/np.max(C3[3]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C3[3]/np.max(C3[3]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    ax=plt.subplot(2,5,10)
    ax.set_aspect('equal')
    plt.contourf(C3[4]/np.max(C3[4]), levels, cmap='bwr', vmin=0, vmax=1, origin=origin)
    CS=plt.contour(C3[4]/np.max(C3[4]),levels, colors='k', linewidths=0.5, origin=origin)
    plt.clabel(CS, inline=1, fontsize=4)
    plt.axis('off')

    #plt.tight_layout()
    plt.savefig('av_actmap_unbroken.png', bbox_inches='tight', dpi=300)
    plt.close()   

	print ("[INFO] unbroken waves activation maps plotted ...")





