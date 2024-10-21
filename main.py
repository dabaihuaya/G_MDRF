# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:28:49 2024

@author: pc
"""
import time
from get_data import get_data
from random_samples import random_samples_and_remaining
from sivgp import gp
from samrf import samrf
from tqdm import tqdm
t_start = time.time()
#类别数
num_class = 16
#加载数据
img_w, img_h, img_b, img, img_gt, all_img ,img_gt_ind, img_gt_lab = get_data()
#每类随机抽取num_samples个样本
'''
x_train, y_train, x_ind, x_test, y_test, y_ind, random_indices_log = random_samples_and_remaining(img, img_gt_lab, img_gt_ind, num_samples=80)

#高斯过程推断潜在函数
result_matrix1,oa,precision_per_class1 =gp(x_train,y_train,all_img,img_gt_lab,img_gt_ind,x_ind,num_class)
gdmrf_result,look_plt = samrf(result_matrix1,x_ind, y_train,num_class,img_w,img_h,img_gt_ind, img_gt_lab,img_gt)
look_plt()
'''
gpq,qq,qq_kappa = [],[],[]
for w in tqdm(range(100)):
    
    x_train, y_train, x_ind, x_test, y_test, y_ind, random_indices_log = random_samples_and_remaining(img, img_gt_lab, img_gt_ind, num_samples=10)
    
    #高斯过程推断潜在函数
    result_matrix1,oa,precision_per_class1 =gp(x_train,y_train,all_img,img_gt_lab,img_gt_ind,x_ind,num_class)
    gdmrf_result,look_plt,kappa_score2 = samrf(result_matrix1,x_ind, y_train,num_class,img_w,img_h,img_gt_ind, img_gt_lab,img_gt)
    qq.append(max(gdmrf_result))
    qq_kappa.append(kappa_score2)
    gpq.append(oa*100)
import numpy as np
print('gp分类精度',np.mean(gpq),'+-',np.std(gpq),'gdmrf分类精度',np.mean(qq),'+-',np.std(qq))