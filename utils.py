import scipy.io
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import shutil

# get a single 3D model from .mat file
def get_model(file_path): 
    
    mat = scipy.io.loadmat( file_path )
    model_array = mat['instance']
    model_array = np.pad(model_array,1,'constant',constant_values=0)
    return model_array

# load all models for a single rotation
def load_all(folder_name, contains = None):
    
    file_names = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    
    if (contains != None):
        file_names = [s for s in file_names if contains in s]
    
    models = []
    
    for m in range(len(file_names)):
        file_path = (folder_name + '/' + file_names[m])
        models.append(get_model(file_path))
    return np.array(models)

# visualize 3D voxel models, input is a list of a batch of 3D arrays to visualize all conditions together  
def visualize_all(models , save = False, name = "output", fig_count = 4, fig_size = 5):
    
    fig = plt.figure()
    
    m = 0
    for model in models:
        
        if(model.dtype == bool):
            voxel = model
        else:
            voxel = np.squeeze(model) > 0.5
    
        ax = []
        colors = []
        
        for i in range(fig_count):
            ax.append( fig.add_subplot(len(models), fig_count, (m*fig_count) + i+1, projection='3d') )

        for i in range(fig_count):
            ax[i].voxels(voxel[i], facecolors='red', edgecolor='k', shade=False)
            ax[i].grid(False)
            ax[i].axis('off')
        
        m += 1
    
    plt.tight_layout()
    
    fig.set_figheight(fig_size)
    fig.set_figwidth(fig_size*fig_count)
    #plt.show()
    if(save):
        fig.savefig(name +'.png')
        plt.close(fig)
        fig.clear()
    else :
        plt.show()

# plot loss graph        
def plot_graph(lists, name):
    for l in lists:
        plt.plot(l)
    
    plt.savefig(name +'.png')
    plt.close()

# create the log folder   
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)