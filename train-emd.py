
"""
For training EMD with different hyperparameters
@author: Rohan
"""
import Conv2D
import pandas as pd
import os
import numpy as np

class TRAINEMD:
    
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
    
    #Initialize lists of hyperparamters
    
    num_filt=[32,64,128,256]
    kernel_size=[1,3,5]
    num_dens_neurons=[32,64,128,256]
    num_dens_layers=[1,2]
    num_conv_2d=[1,2,3,4]
    num_epochs=64
    
    #Training loop
    
    for n_f in num_filt:
        
        for k_s in kernel_size:
            
            for n_d_n in num_dens_neurons:
                
                for n_d_l in num_dens_layers:
                    
                    for n_c_l in num_conv_2d:
                        
                        obj=Conv2D.CNNEMD(True)
                        mean, sd = obj.ittrain(n_f,k_s, n_d_n ,n_d_l,n_c_l,num_epochs)
                        mean_data.append(mean)
                        std_data.append(sd)
                        nfilt_data.append(n_f)
                        ksize_data.append(k_s)
                        neuron_data.append(n_d_n)
                        numlayer_data.append(n_d_l)
                        convlayer_data.append(n_c_l)
                        z=abs(mean)/sd
                        z_score.append(z)

    for_pdata=[mean_data,std_data,nfilt_data,ksize_data,neuron_data,numlayer_data,convlayer_data,z_score]
    
    current_directory=os.getcwd()
    data_directory=os.path.join(current_directory,r'EMD_data.xlsx')
    
    df=pd.DataFrame(for_pdata)
    df.to_excel(data_directory)
    
    
