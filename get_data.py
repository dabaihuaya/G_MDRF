
import numpy as np
import scipy.io as sio
# from skimage import data, transform
import os
from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
def get_data():
    data_path = os.path.join(os.getcwd(), 'datasets')
    img = sio.loadmat(os.path.join(data_path, 'salinas.mat'))['HSI_original']
    img_gt = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['Data_gt']
    img_w, img_h, img_b = img.shape
    img_gt_line = np.reshape(img_gt, (img_w * img_h, -1))
    indices = np.where(img_gt_line > 0)
    img_gt_non_zero = img_gt_line[indices]
    new_array = np.vstack((indices[0], img_gt_non_zero))
    reshaped_img = img.reshape(-1, img_b)
    scaler = StandardScaler()
    reshaped_img1 = scaler.fit_transform(reshaped_img)
    return img_w, img_h, img_b, reshaped_img1[indices[0], :], new_array, reshaped_img1, new_array[0, :], new_array[1, :]








    
    

