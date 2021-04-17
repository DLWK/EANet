#  -*- coding: utf-8 -*- 
import cv2
import os

def Edge_Extract():

    img_root = '/home/src_unet/data_5fold_edge/test_96_4/label/'
    edge_root = img_root
    count = 0
    for name in os.listdir(img_root):
        image_name = name[0:-9]
        image_path = img_root+name
        img = cv2.imread(image_path,0)
        # print(image_name)
        edge = cv2.Canny(img,30,100)
        cv2.imwrite(edge_root+image_name+"_edge.png", edge)
        count+=1
    print(count)
if __name__ == '__main__':
    Edge_Extract()
