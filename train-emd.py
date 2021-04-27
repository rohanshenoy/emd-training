"""
For training EMD with different hyperparameters
@author: Rohan
"""
import Conv2D
import pandas as pd
import os
import numpy as np

#Data
    
df=[]
mean_data=[]
std_data=[]
nfilt_data=[]
ksize_data=[]
neuron_data=[]
numlayer_data=[]
convlayer_data=[]
epoch_data=[]
z_score=[]

#List of lists of Hyperparamters

hyp_list=[[128,3,256,1,4],
          [64,5,32,1,4],
          [32,5,128,1,3],
          [128,5,32,1,4],
          [128,5,256,1,2],
          [256,1,64,2,4],
          [32,5,256,1,3],
          [128,3,64,1,4],
          [128,5,32,1,3],
          [128,5,64,1,3],
         ]

it=1

for hyp in hyp_list:
    num_filt=hyp[0]
    kernel_size=hyp[1]
    num_dens_neurons=hyp[2]
    num_dens_layers=hyp[3]
    num_conv_2d=hyp[4]
    num_epochs=29
    
    for i in [0,1,2]:
        num_epochs=num_epochs+1
        obj=Conv2D.CNNEMD(True)
        mean, sd = obj.ittrain(num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs)
        mean_data.append(mean)
        std_data.append(sd)
        nfilt_data.append(num_filt)
        ksize_data.append(kernel_size)
        neuron_data.append(num_dens_neurons)
        numlayer_data.append(num_dens_layers)
        convlayer_data.append(num_conv_2d)
        z=abs(mean)/sd
        z_score.append(z)
        
for_pdata=[mean_data,std_data,nfilt_data,ksize_data,neuron_data,numlayer_data,convlayer_data,z_score]
    
current_directory=os.getcwd()
data_directory=os.path.join(current_directory,r'FG.xlsx')
    
df=pd.DataFrame(for_pdata)
df.to_excel(data_directory)
