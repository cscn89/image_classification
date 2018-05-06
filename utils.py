# -*- coding:utf-8 -*-
__author__ = 'shichao'

import cv2
import glob
import numpy as np
import os
import config


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