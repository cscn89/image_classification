# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import glob
import numpy as np
import os
import config

'''
this script includes the functions
calculate IOU by the func calc_IOU
display the elasped time in the form hour:minute:second by the func humanTime
display image keypoints detected by surf by the func displayKeyPoints

'''

def calc_IOU(box1, box2):
    '''
    :param box1: the coordinate of box in the form (x1,y1,x2,y2)
    :param box2: the coordinate of box in the form (x1,y1,x2,y2)
    :return:
    '''
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    intersection = float(abs((x2-x1)*(y2-y1)))
    union = abs((box1[2]-box1[0])*(box1[3]-box1[1])) + abs((box2[2]-box2[0])*(box2[3]-box2[1]))
    IOU = intersection/union
    return IOU

def humanTime(sec):
    '''
    print the elasped time in the form of day:hour:minute:second
    :param sec:
    :return:
    '''
    mins, secs = divmod(sec,60)
    hours, mins = divmod(mins,60)
    days, hours = divmod(hours,24)

    return '%02d:%02d:%02d:02f'%(days,hours,mins,secs)


def displayKeyPoints(dirname,outpath):
    '''
    display keypoints detected by the descriptors
    :param dirname: the direcotry contains images to be detected
    :param outpath: the direcotry to save the featured images
    :return:
    '''
    img_paths = glob.glob(os.path.join(dirname,'*.*'))
    # img = (cv2.imread(img_path) for img_path in img_paths)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        filename = os.path.split(img_path)[1]
        surf_feature = cv2.xfeatures2d.SURF_create(hessianThreshold=config.SURF_HESSIANTHRESHOLD)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        dst = np.zeros(gray.shape)
        kp, des = surf_feature.detectAndCompute(gray,None)
        cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        cv2.imwrite(os.path.join(outpath,filename),img)