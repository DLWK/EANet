import glob
import numpy as np
import torch
import os
import cv2
import csv
import tqdm
from models import EANet
from evaluation import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":

    fold=1
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    
    net =EANet(n_channels=1, n_classes=1)
    net.to(device=device)
    # 加载训练模型参数
    net.load_state_dict(torch.load('./JZX/new_best_'+str(fold)+'.pth', map_location=device))    
    # 测试模式
    net.eval()
    # 读取所有图片路径
   
    tests_path = glob.glob('/home/###/data/JZX/test/image/*.bmp')
    
    mask_path = "/home/###/data/JZX/test/GT/"
  
    save_path = "/home/####/data/JZX/test/new/"
    save_path1 = "/home/####/data/JZX/test/Promaps/new/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_path1):
        os.mkdir(save_path1)    
    # 遍历素有图片

    acc = 0.	# Accuracy
    SE = 0.		# Sensitivity (Recall)
    SP = 0.		# Specificity
    PC = 0. 	# Precision
    F1 = 0.		# F1 Score
    JS = 0.		# Jaccard Similarity
    DC = 0.		# Dice Coefficient
  
    count = 0
    f = open('./JZX_box/mynet.csv', 'w')
    f.write('name,JS,DC'+'\n')
    for test_path in tqdm.tqdm(tests_path):
        name = test_path.split('/')[-1][:-4]
        mask = mask_path + name+".bmp"
        mask1 = cv2.imread(mask, 0)
        mask = torch.from_numpy(mask1).cuda()
        mask=mask/255

       

        # save_res_path = save_path+name + '_res.jpg'
        save_mask_path = save_path+name + '.png'
        save_prob_path = save_path1+name + '.png'

        # 读取图片
        img = cv2.imread(test_path,0)
        # 转为灰度图
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为96*96的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
      
        pred, p1,p2,p3,p4,e= net(img_tensor)
        sig = torch.nn.Sigmoid()
        pred = sig(pred)
        #保存结果
        # 提取结果
        pred1 = np.array(pred.data.cpu()[0])[0]
     
        pred1[pred1 >= 0.5] = 255
        pred1[pred1 < 0.5] = 0
        img = pred1
        
        acc += get_accuracy(pred,mask)
        SE += get_sensitivity(pred,mask)
        SP += get_specificity(pred,mask)
        PC += get_precision(pred,mask)
        F1 += get_F1(pred,mask)
        JS += get_JS(pred,mask)
        DC += get_DC(pred,mask)
        
      
        count+=1
    acc = acc/count
    SE = SE/count
    SP = SP/count
    PC = PC/count
    F1 = F1/count
    JS = JS/count
    DC = DC/count
  
   

    print('ACC:%.4f' % acc)
    print('SE:%.4f' % SE)
    print('SP:%.4f' % SP)
    print('PC:%.4f' % PC)
    print('F1:%.4f' % F1)
    print('JS:%.4f' % JS)
    print('DC:%.4f' % DC)
  





# python<predict.py>sce2.txts
#by kun wang