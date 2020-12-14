#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import message_filters
import pandas as pd
import numpy as np
from itertools import groupby

class env_detect:
    def __init__(self):
        # delete when merging
        rospy.init_node('env_det')

        self.bridge = cv_bridge.CvBridge()
        self.image_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
        self.dep_img_sub = message_filters.Subscriber('camera/depth/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.dep_img_sub], 10, 0.2)
        self.ts.registerCallback(self.callback)


        self.rgb_resized, self.dep_resized, self.hsv, self.mask, self.canny = None, None, None, None, None
        self.cont = None
        self.rgb_img = None

        self.keys = ['fire_hydrant', 'green_box', 'mail_box', 'number_5']
        values = [dict(mark=40, suspect=40, lower_bound = np.array([0, 220, 5]), upper_bound = np.array([15, 255, 100])),
                dict(mark=80, suspect=80, lower_bound = np.array([60, 235, 30]), upper_bound = np.array([60, 255, 100])),
                dict(mark=120, suspect=120, lower_bound = np.array([110, 120, 20]), upper_bound = np.array([135, 190, 70])), 
                dict(mark=160, suspect=160,lower_bound = np.array([0, 0, 88]), upper_bound = np.array([0, 0, 105]))]
        self.beer_grey = 200
        self.mark_dict = dict(zip(self.keys, values))

        self.base = None
        self.output = None
        

    def callback(self, rgb_img_msg, dep_img_msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(rgb_img_msg,desired_encoding='bgr8')
        (h, w) = self.rgb_img.shape[:2]
        self.base = np.zeros((h/4, w/4), np.uint8)
        self.output = np.zeros((h/4, w/4), np.uint8)

        self.rgb_resized = cv2.resize(self.rgb_img, (w/4, h/4))
        self.blurred = cv2.GaussianBlur(self.rgb_resized, (3, 3), 0)
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)

        dep_img = self.bridge.imgmsg_to_cv2(dep_img_msg, desired_encoding='32FC1')
        self.dep_resized = cv2.resize(dep_img, (w/4, h/4))
        
        # self.acquire_info()

        mask_dict = self.mask_generate()
        filled_dict = self.fill(mask_dict)
        
        mask_canny_dict = self.canny_dic(mask_dict)
        filled_canny_dict = self.canny_dic(filled_dict) 

        # value of this dict will be a list
        # three situations about length of this list
        # 0: nothing detected, 1: only 1 (potential) object, 1+: more than 1
        sliced_masks_canny_dict = self.mask_slice(mask_canny_dict)
        sliced_filled_canny_dict = self.mask_slice(filled_canny_dict)
        
        candidate_dict = self.mask_slice(filled_dict)
        
        # None if nothing, otherwist like [T/F, T/F, T/F]
        fi_check = self.confirm(sliced_masks_canny_dict['fire_hydrant'], sliced_filled_canny_dict['fire_hydrant'])
        if fi_check is not None:
            self.generate_output(fi_check, candidate_dict['fire_hydrant'], self.mark_dict['fire_hydrant']['mark'])
        # green box
        
        # need sort beer can from mail_box
        beer_marker = self.beer_sort(sliced_masks_canny_dict['mail_box'])

        if len(beer_marker) != 0:
            beer_candidate = []
            for ind, ele in enumerate(candidate_dict['mail_box']):
                if ind == len(beer_marker):
                    break
                if beer_marker[ind]:
                    beer_candidate.append(ele)
            beer_pre_checked = [True] * len(beer_candidate)
            self.generate_output(beer_pre_checked, beer_candidate, self.beer_grey)

        di_check = self.confirm(sliced_masks_canny_dict['number_5'], sliced_filled_canny_dict['number_5'])
        if di_check is not None:
            self.generate_output(di_check, candidate_dict['number_5'], self.mark_dict['number_5']['mark'])

        # no need to operate mail_box list if no beer
        if len(beer_marker) == 0 or beer_marker == [False] * len(beer_marker):
            ma_check = self.confirm(sliced_masks_canny_dict['mail_box'], sliced_filled_canny_dict['mail_box'])
            if ma_check is not None:
                self.generate_output(ma_check, candidate_dict['mail_box'], self.mark_dict['mail_box']['mark'])
        else:
            smcd_mail_box, sfcd_mail_box, candidate_mail_box = [], [], []
            for ind, ele in enumerate(beer_marker):
                if ind == len(sliced_masks_canny_dict['mail_box']) or ind == len(sliced_filled_canny_dict['mail_box']) or ind == len(candidate_dict['mail_box']):
                    break

                if not ele:
                    smcd_mail_box.append(sliced_masks_canny_dict['mail_box'][ind]) 
                    sfcd_mail_box.append(sliced_filled_canny_dict['mail_box'][ind]) 
                    candidate_mail_box.append(candidate_dict['mail_box'][ind])
            ma_check = self.confirm(smcd_mail_box, sfcd_mail_box)
            if ma_check is not None:
                self.generate_output(ma_check, candidate_mail_box, self.mark_dict['mail_box']['mark'])

        # green box has flat surface
        if len(sliced_masks_canny_dict['green_box']) == 0:
            gb_check = None
        else:
            gb_check = [True] * len(sliced_masks_canny_dict['green_box'])
        if gb_check is not None:
            self.generate_output(gb_check, candidate_dict['green_box'], self.mark_dict['green_box']['mark'])
        

        
        cv2.imshow('output', self.output)
        cv2.imshow('rgb', self.rgb_resized)
        cv2.waitKey(3)
    
    def beer_sort(self, masks):
        if len(masks) == 0:
            return []

        (h, w) = self.rgb_resized.shape[:2]    
        # 270, 480

        beer_marker = [False] * len(masks)

        for ind, ele in enumerate(masks):

            holder = []
            # detect vertical gap to determine it's beer or mail_box
            for inde in range(h):
                b = np.where(ele[inde,:]==255)[0]
                holder.append(b.size)
            seg = self.list_segment(holder)
            
            # len(seg) > 1 means there is a vertical gap, it's beer
            if len(seg) > 1:
                beer_marker[ind] = True

        return beer_marker

        
    def generate_output(self, check_res, candidate, mark):
        # print(len(check_res), len(candidate))
        for ind, ele in enumerate(candidate):
            if ind == len(check_res):
                break
            if check_res[ind]:
                # replace with pre-defined grey value
                ele[ele>0] = mark

                self.output = cv2.bitwise_or(self.output, ele)

    def confirm(self, masks, filled):
        # detected nothing
        if len(masks) == 0:
            return None
        '''
        if len(masks) != len(filled):
            print('edge gap exists, fix it by increase gap_threshold')
        '''
        res = [False] * len(masks)

        for i in range(0, len(masks)):
            # print(self.contours(masks[i]), self.contours(filled[i]))
            if i == len(filled):
                break
            if self.contours(masks[i]) != self.contours(filled[i]):
                res[i] = True
            # sometime to far to detect texture in the object surface
            # use a threshold to deal with far situation
            else: 
                res[i] = False

        return res
        
    @staticmethod
    def list_segment(li):
        segment = []
        nonzeroind = np.nonzero(li)[0]
        gap_threshold = 2
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(nonzeroind), fun):
            l1 = [j for i, j in g] 
            if len(l1) > gap_threshold:
                scop = (min(l1), max(l1))
                segment.append(scop)

        return segment

    def mask_slice(self, dic):
        (h, w) = self.rgb_resized.shape[:2]    
        # 270, 480

        sliced = [None] * len(self.keys)

        for ind, key in enumerate(self.keys):
            res = []

            # detected nothing on this mask, no need to slice
            if dic[key].sum() == 0:
                sliced[ind] = res
                continue
            
            holder = []
            # 0: balck, 255: white
            for inde in range(w):
                a = np.where(dic[key][:,inde]==255)[0]
                holder.append(a.size)
            seg = self.list_segment(holder)
            
            # only found 1 object, no need to slice
            if len(seg) == 1:
                res.append(dic[key])
            else:    
                for index, ele in enumerate(seg):
                    res.append(self.single_obj_frame_generate(ele, dic[key]))

            sliced[ind] = res
        
        sliced_masks_dict = dict(zip(self.keys, sliced))

        return sliced_masks_dict
        
    def single_obj_frame_generate(self, ind_tuple, ori_img):
        frame = self.base.copy()
        frame[:,ind_tuple[0]:ind_tuple[1]] = ori_img[:,ind_tuple[0]:ind_tuple[1]]
        return frame

    def canny_dic(self, dic):
        canny = [None] * len(self.keys)

        for ind, key in enumerate(self.keys):
            canny[ind] = cv2.Canny(dic[key], 16, 128, apertureSize=3)

        canny_dic = dict(zip(self.keys, canny))

        return canny_dic

    def contours(self, binary_img):
        a = binary_img.copy()
        _, contours, _ = cv2.findContours(a,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)    
        
        return len(contours)

    def fill(self, dic):
        floodfilled = [None] * len(self.keys)
        (h, w) = self.rgb_resized.shape[:2]

        for ind, key in enumerate(self.keys):
            a = dic[key].copy()
            fill_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(a, fill_mask, (0,h-1), 255)
            a = cv2.bitwise_not(a)
            a = dic[key] | a
            floodfilled[ind] = a

        filled = dict(zip(self.keys, floodfilled))

        return filled

    def acquire_info(self):
        h = self.hsv[:,:,0]
        h_df = pd.DataFrame(h)
        h_df.to_csv('h.csv')
        s = self.hsv[:,:,1]
        s_df = pd.DataFrame(s)
        s_df.to_csv('s.csv')
        v = self.hsv[:,:,2]
        v_df = pd.DataFrame(v)
        v_df.to_csv('v.csv')

    def mask_generate(self):
        mask = [None] * len(self.keys)
        
        for ind, key in enumerate(self.keys):

            mask[ind] = cv2.inRange(self.hsv, self.mark_dict[key]['lower_bound'], self.mark_dict[key]['upper_bound'])

        mask_dict = dict(zip(self.keys, mask))

        return mask_dict
        

env_det = env_detect()
rospy.spin()