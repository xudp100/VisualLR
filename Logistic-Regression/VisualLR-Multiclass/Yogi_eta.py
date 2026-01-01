import numpy as np
import pandas as pd
import os
from utils import import_data
from func import batch_grad_multi


dataset = "letter"
X, y = import_data(dataset)
n_data, n_feature = np.shape(X)  


##############################
n_iteration = int(20001) 
n_trial = 1   
S = 8   
seed = 1
path = 'results/Yogi'
##############################

##############################
r = 0.2  
alpha = 1e-3    
beta = 0.9    
theta = 0.999    
eps = 1e-7
##############################

# 
unique_labels = np.unique(y)
n_classes = len(unique_labels)
print(f"{dataset}: {n_data} samples, {n_feature} features, {n_classes} classes")

# one-hot
y_onehot = np.eye(n_classes)[(y - 1).astype(int)]



############################## ### ##############################
def set_seed(seed): 
    os.environ['PYTHONHASHSEED'] = str(seed)     
    np.random.seed(seed)  
############################## ### ##############################


############################## ### ##############################
def save ():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ROOT_PATH = os.path.abspath('./')
    history_folder = os.path.join(ROOT_PATH, path)  
    paras = ('%s_%s_b=%s_lr=%s_seed=%s' %
            (alg, dataset, S, alpha, seed))
    history_file = os.path.join(history_folder, paras)

    global statistics
    statistics = {
    "eta_min": [],
    "eta_max": [],
    "max_min": [],
    }   
    
    statistics ["eta_min"].append(eta_min)
    statistics ["eta_max"].append(eta_max)
    statistics ["max_min"].append(max_min)
    
    
    for key, value in statistics.items():
        pd.DataFrame(value).to_csv(history_file + "_" + str(key) + '.csv') 
############################## ### ##############################     


############################## ### ##############################
def delete_files_in_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)  
    
    if os.path.isdir(folder_path):    
        for file_name in os.listdir(folder_path):   
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        print(f"Folder '{folder_name}' does not exist.")

# 
delete_files_in_folder(path)   
############################## ### ##############################

  
tr = 0      
    
############################## Yogi ##############################
for trial in range(n_trial):
    m = 0
    v = 0 
    set_seed(seed)
    W = np.random.randn(n_feature, n_classes)
    tr = tr + 1
    set_seed(seed + 2*tr)
    eta_min = []
    eta_max = []
    max_min = []

    for it in range(n_iteration):
        
        index_set = np.random.randint(n_data, size=S)
        g = batch_grad_multi(index_set, n_data, n_feature, n_classes, 
                            X, y_onehot, W, r)
        
        ################################
        m = beta*m + (1 - beta)*g
        v = v - (1 - theta)*np.sign(v - g*g)*g*g
        eta = alpha/np.sqrt(v + eps)
        W = W - eta*m
        ################################
          
          
        eta_hat = 1/np.sqrt(v + eps)
        print(f"Iter {it}: eta_hat (max) = {np.max(eta_hat):.6f}")
        print(f"Iter {it}: eta_hat (min) = {np.min(eta_hat):.6f}")
        
        eta_min.append(np.min(eta_hat))
        eta_max.append(np.max(eta_hat))
        max_min.append(np.max(eta_hat)*np.max(eta_hat)/np.min(eta_hat))


    alg = "Yogi"+"_"+str(trial) 
    save()