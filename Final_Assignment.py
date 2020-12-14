#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:56:10 2020

"""
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
#import pandas as pd

import math
import numpy as np
import message_filters
from sensor_msgs.msg import Image
import cv2, cv_bridge
from itertools import groupby

LINEAR_SPEED=0.2
ANGULAR_SPEED=0.5

class ObjectSearch():
    def __init__(self):
        rospy.init_node('ObjectSearch')
        #VARIABLES
        #Scanning variables
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        self.r=rospy.Rate(10)
        
        self.nodes_list=[Point(-0.49,-0.31,0)]#,Point(1.09,3.13,0),Point(3.2,1.0,0),Point(4.54,1.0,0)
        self.keys = ['fire_hydrant', 'green_box', 'mail_box', 'number_5']
        best_node_values =[dict(mark=40, best_node=None, orientation=None, visibility=1000),
                           dict(mark=80, best_node=None, orientation=None, visibility=1000),
                           dict(mark=120, best_node=None, orientation=None, visibility=1000),
                           dict(mark=160, best_node=None, orientation=None, visibility=1000)]
        
        self.node_dict = dict(zip(self.keys,best_node_values))
        
        self.quarternion=None
        
        
        #Computer vision variables
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
        self.dep_img_sub = message_filters.Subscriber('camera/depth/image_raw', Image)
        self.odom=message_filters.Subscriber("odom",Odometry)
        self.laser_sub = message_filters.Subscriber('/scan', LaserScan)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.dep_img_sub,self.odom,self.laser_sub], 10, 0.2)
        self.ts.registerCallback(self.callback)


        self.rgb_resized, self.dep_resized, self.hsv, self.mask, self.canny = None, None, None, None, None
        self.cont = None
        self.single_track = None
        self.rgb_img = None
        self.hei, self.wid = None, None

        self.keys = ['fire_hydrant', 'green_box', 'mail_box', 'number_5']
        values = [dict(mark=40, suspect=40, lower_bound = np.array([0, 220, 5]), upper_bound = np.array([15, 255, 100]),edges=[1]),
                dict(mark=80, suspect=80, lower_bound = np.array([60, 235, 30]), upper_bound = np.array([60, 255, 100]),edges=[1.65]),
                dict(mark=120, suspect=120, lower_bound = np.array([110, 120, 20]), upper_bound = np.array([135, 190, 70]),edges=[0.5]), 
                dict(mark=160, suspect=160,lower_bound = np.array([0, 0, 88]), upper_bound = np.array([0, 0, 105]),edges=[0.8])]
        self.beer_grey = 200
        self.mark_dict = dict(zip(self.keys, values))

        self.base = None
        self.output = None
        
        self.h=1080
        self.w=1920
        
        # Beaconing variables
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.controller = Twist()
        self.fwd_speed = 0.2
        self.reach_dis = 0.5
        self.safe_dis_front = 0.4
        self.safe_dis_side = 0.3
        self.front_check_angle = 35
        self.side_check_angle = 45
        self.reach_count = 0
        self.rate = rospy.Rate(10) 
        self.turn_right_marker = False
        self.turn_left_marker = False
        self.sing_res = False
        self.laser_msg=[]
    def callback(self, rgb_img_msg, dep_img_msg,msg, laser_sub):
        quarternion = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,\
		            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.quarternion=quarternion
        self.laser_msg=laser_sub
        self.rgb_img = self.bridge.imgmsg_to_cv2(rgb_img_msg,desired_encoding='bgr8')
        (h, w) = self.rgb_img.shape[:2]
        self.hei = self.rgb_img.shape[:2][0]/4
        self.wid = self.rgb_img.shape[:2][1]/4

        self.base = np.zeros((h/4, w/4), np.uint8)
        self.output = np.zeros((h/4, w/4), np.uint8)

        self.rgb_resized = cv2.resize(self.rgb_img, (w/4, h/4))
        self.blurred = cv2.GaussianBlur(self.rgb_resized, (3, 3), 0)
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)
        self.hsv2 = cv2.cvtColor(self.rgb_resized, cv2.COLOR_BGR2HSV)
        
        dep_img = self.bridge.imgmsg_to_cv2(dep_img_msg, desired_encoding='32FC1')
        self.dep_resized = cv2.resize(dep_img, (w/4, h/4))
        
        #self.acquire_info()

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

        self.single_track = self.output.copy()
        
        self.visualization()
        
    ##### the beaconing part #####
    ##### CODE BY KAICHEN ZHANG #####

    # main controller of beaconing process
    def beacon_control(self, indicator, laser_sub):
        dedi_pixel = self.find_related_grey(indicator)

        # mark other non-interested pixel as black
        self.single_track[self.single_track!=dedi_pixel] = 0

        # uncomment next sentence if want see images
        #self.visualization()
        
        # row-range in img of objective
        pos = self.find_obj_pos(dedi_pixel)

        if pos is not None and self.reached_check(dedi_pixel, pos) and self.better_reach_res():
            print('Objective Reached')
            self.controller = Twist()
            self.pub.publish(self.controller)
            return True

        # front_left, front_right, side_left, side_right 
        las_dic = self.laser_regroup(laser_sub.ranges)
        
        cant_beacon_front = las_dic['front_left']['too_close'] or las_dic['front_right']['too_close']
        cant_beacon_side = las_dic['side_left']['too_close'] or las_dic['side_right']['too_close']
 
        if cant_beacon_front:
            if las_dic['front_left']['too_close'] and not las_dic['front_right']['too_close']:
                self.turn_right_marker, self.turn_left_marker = True, False
                # print('turning right')
                self.turn('right')
            if not las_dic['front_left']['too_close'] and las_dic['front_right']['too_close']:
                self.turn_right_marker, self.turn_left_marker = False, True
                # print('turning left')
                self.turn('left')
            if las_dic['front_left']['too_close'] and las_dic['front_right']['too_close']:
                if not self.fully_blocked(las_dic['front_left']['data']) and self.fully_blocked(las_dic['front_right']['data']):
                    self.turn('left')
                elif self.fully_blocked(las_dic['front_left']['data']) and not self.fully_blocked(las_dic['front_right']['data']):
                    self.turn('right')
                elif not self.fully_blocked(las_dic['front_left']['data']) and not self.fully_blocked(las_dic['front_right']['data']):
                    if np.mean(las_dic['front_left']['data']) >= np.mean(las_dic['front_right']['data']):
                        self.turn('left')
                    else:
                        self.turn('right')
                else:
                    return 'failed'
            return False

        if cant_beacon_side:      
            if self.turn_left_marker and las_dic['side_right']['too_close']:
                # want turn right but right side blocked, go forward
                self.forward()
                return False
            if self.turn_right_marker and las_dic['side_left']['too_close']:
                # want turn left but left side blocked, go forward
                self.forward()
                return False
            if las_dic['side_right']['too_close'] and las_dic['side_left']['too_close']:
                # print('narrow escape')
                self.forward()
                return False
        
        if pos is None:
            if self.turn_right_marker:
                # print('left back')
                self.turn('left')
            elif self.turn_left_marker:
                # print('right back')
                self.turn('right')
            else:
                return 'failed'
        else:
            err = self.cal_err(pos)
            self.free_beacon(err)

        return False
        

    def fully_blocked(self, nump):
        if len(nump[nump>=self.safe_dis_front]) == 0:
            return True
        else:
            return False

    # ensure a better pos of reached status
    def better_reach_res(self):
        sta, end = int(round(self.wid/3)), int(round(2*self.wid/3))
        a = np.where(self.single_track[:,sta:end]!=0)[0]
        if a.size == 0:
            return False
        else:
            return True

    def visualization(self):
        cv2.imshow('rgb', self.rgb_resized)
        cv2.imshow('output', self.output)
        cv2.imshow('single_track', self.single_track)
        cv2.waitKey(3)

    def forward(self):
        # print('forwarding')
        self.controller.linear.x = self.fwd_speed
        self.controller.angular.z = 0
        self.pub.publish(self.controller)
    
    def turn(self, dir):
        if dir == 'right':
            self.controller.angular.z = -0.4
        if dir == 'left':
            self.controller.angular.z = 0.4
        self.controller.linear.x = 0
        self.pub.publish(self.controller)

    def free_beacon(self, err):
        # print('free beaconing')
        self.controller.angular.z = -float(err)/700
        self.controller.linear.x = self.fwd_speed
        self.pub.publish(self.controller)        

    def cal_err(self, pos):
        l_p = pos[0]
        r_p = (self.wid - 1) - pos[1]

        return l_p - r_p

    # average depth of interested pixels
    def reached_check(self, dedi_pixel, pos):
        # generate a T/F array which T represent interest pixel
        a = np.where(self.single_track[:,0:pos[1]]==dedi_pixel, True, False)

        activated_count = (np.where(a==1)[0]).size
        
        # will be 3 type element: 0, NAN, depth
        activated_depth_array = a * self.dep_resized[:,0:pos[1]]
        # replace NAN with 0
        activated_depth_array = np.nan_to_num(activated_depth_array)

        activated_depth_sum = np.sum(activated_depth_array)

        avg_activated_dep = activated_depth_sum / activated_count

        if avg_activated_dep < self.reach_dis:
            return True
        else:
            return False

    # find interested pixel's existence range (row)
    # if non-found, return None
    # otherwise will be like (255, 355) -> left first objective in img
    def find_obj_pos(self, dedi_pixel):
        holder = []
        for ind in range(self.wid):
            a = np.where(self.single_track[:,ind]==dedi_pixel)[0]
            holder.append(a.size) 
        seg = self.list_segment(holder)

        if len(seg) == 0:
            return None
        else:
            return seg[0]

    # extract interested laser groups
    def laser_regroup(self, laser):
        a = ['front_left', 'front_right', 'side_left', 'side_right']
        laser = np.array(laser)
        b = [dict(data=laser[0:self.front_check_angle], 
                too_close=True if len(laser[0:self.front_check_angle][laser[0:self.front_check_angle]<self.safe_dis_front]) > 0 else False),
            dict(data=laser[360-self.front_check_angle:360],  
                too_close=True if len(laser[360-self.front_check_angle:360][laser[360-self.front_check_angle:360]<self.safe_dis_front]) > 0 else False),
            dict(data=laser[self.side_check_angle:self.side_check_angle*2], 
                too_close=True if len(laser[self.side_check_angle:self.side_check_angle*2][laser[self.side_check_angle:self.side_check_angle*2]<self.safe_dis_side]) > 0 else False),
            dict(data=laser[360-2*(self.side_check_angle):360-self.side_check_angle], 
                too_close=True if len(laser[360-2*(self.side_check_angle):360-self.side_check_angle][laser[360-2*(self.side_check_angle):360-self.side_check_angle]<self.safe_dis_side]) > 0 else False)
            ]

        return dict(zip(a, b))
        
    # function to find related grey value of objective
    def find_related_grey(self, indicator):
        dedi_pixel = None

        if indicator == 'beer_can':
            dedi_pixel = self.beer_grey
        else:
            dedi_pixel = self.mark_dict[indicator]['mark']

        return dedi_pixel
    
     ##### end beaconing part #####
     
 ##### The computer vision part #####
    ##### CODE BY KAICHEN ZHANG #####
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


            if dic[key][h-1,0] == 0:
                seed = (0, h-1)
            elif dic[key][h-1,w-1] == 0:
                seed = (w-1, h-1)
            elif dic[key][0, w-1] == 0:
                seed = (w-1, 0)
            else:
                seed = (0, 0)

            a = dic[key].copy()
            fill_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(a, fill_mask, seed, 255)
            a = cv2.bitwise_not(a)
            a = dic[key] | a
            floodfilled[ind] = a

        filled = dict(zip(self.keys, floodfilled))

        return filled

    """def acquire_info(self):
        h = self.hsv[:,:,0]
        h_df = pd.DataFrame(h)
        h_df.to_csv('h.csv')
        s = self.hsv[:,:,1]
        s_df = pd.DataFrame(s)
        s_df.to_csv('s.csv')
        v = self.hsv[:,:,2]
        v_df = pd.DataFrame(v)
        v_df.to_csv('v.csv')"""

    def mask_generate(self):
        mask = [None] * len(self.keys)
        
        for ind, key in enumerate(self.keys):

            mask[ind] = cv2.inRange(self.hsv, self.mark_dict[key]['lower_bound'], self.mark_dict[key]['upper_bound'])

        mask_dict = dict(zip(self.keys, mask))

        return mask_dict
####End computer vision part###


####Navigation part #######        
    def move(self,linear,angular):
        t=Twist()
        t.linear.x=linear
        t.angular.z=angular
        self.pub.publish(t)
        self.r.sleep()
    #Sending goal coordinates to MoveBase Client#
    def move_to_goal(self, point,orientation):
        client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        client.wait_for_server()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position =  point
        goal.target_pose.pose.orientation.x = orientation[0]
        goal.target_pose.pose.orientation.y = orientation[1]
        goal.target_pose.pose.orientation.z = orientation[2]
        goal.target_pose.pose.orientation.w = orientation[3]
        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            rospy.loginfo('node reached')
            return client.get_result()
     
    
    def navigate(self):
        base_quart=[0,0,0.0,1.0]
        (h, w) = (1080,1920)
        for i in self.nodes_list:
            #Move to node i
            self.move_to_goal(i,base_quart)
            t2=rospy.Time.now()
            # Start scanning 360°
            while rospy.Time.now()-t2< rospy.Duration(math.pi*2/ANGULAR_SPEED):
                self.move(0,ANGULAR_SPEED)
                for k in self.keys:
                    vis=1000
                    #Calculating visibility Score (the score depend on depth and object edges also with the amount of pixels)
                    mask=cv2.inRange(self.hsv2,self.mark_dict[k]['lower_bound'],self.mark_dict[k]['upper_bound'])
                    pix=np.count_nonzero(self.output==self.node_dict[k]['mark'])
                    edges_img=cv2.Canny(cv2.bitwise_and(self.hsv2,self.hsv2,mask=mask),w/4,h/4)
                    depth_img=cv2.bitwise_and(self.dep_resized,self.dep_resized,mask=mask)
                    out=np.where(depth_img==0,float("nan"),depth_img)
                    d=np.nanmean(out)
                    e=np.sum(edges_img>0)
                    #print(k,float((h*w-pix))/(h*w),abs((1439.3*pow(d,-0.971))/e))#float((h*w-pix))/(h*w),abs((1439.3*pow(d,-0.971))/e)
                    vistab=[]
                    
                    if(e>0 and d>0 and pix>0):
                        for j in range(len(self.mark_dict[k]['edges'])):
                            #Formula used to calculate the visibility score ( The lower the better)
                            vistab.append(1/pix+((abs((1439.3*pow(d,-0.971))/e-self.mark_dict[k]['edges'][j]))))
                        vis=np.min(vistab)
                        rospy.loginfo(str(k)+" Visibility Score: "+str(vis))
                    if(vis<self.node_dict[k]['visibility']):
                        self.node_dict[k]['visibility']=vis
                        self.node_dict[k]['best_node']=i
                        self.node_dict[k]['orientation']=self.quarternion
                        rospy.loginfo("Best "+str(k))
        for i in self.keys:
            print(i,self.node_dict[i]['visibility'])
        self.move(0,0)          
    def object_found_selection(self):
        st=""
        keys=[]
        #Declare object found and ask user which one to reach
        for k in self.keys:
            
            if(self.node_dict[k]['visibility']!=0):
                keys.append(k)
                print(keys)
                st+="\n Object "+str(k)+" detected and referenced as: "+str(len(keys)-1)
        rospy.loginfo(st)
        rospy.loginfo("Select one: ")
        choice=input()
        
        return keys[choice]
    
    def go_to_node(self):
        choice=self.object_found_selection()
        print(self.node_dict[choice]['best_node'],self.node_dict[choice]['orientation'])
        self.move_to_goal(self.node_dict[choice]['best_node'],self.node_dict[choice]['orientation'])
        print(self.node_dict[choice]['best_node'],self.node_dict[choice]['orientation'])
        return choice
#### End of navigation paart####
        
        
o=ObjectSearch()
o.move(0,0)
rospy.sleep(3) 
o.navigate()
o.move(0,0)
while(True):
    choice=o.go_to_node()
    b=o.beacon_control( choice, o.laser_msg)
    while(b==False):
        b=o.beacon_control( choice, o.laser_msg)
    if(b==True):
        rospy.loginfo(str(choice)+" reached.")
    if(b=='failed'):
        rospy.loginfo(str(choice)+" unreachable.")
    o.sing_res=False
    #o.beaconing('number_5')
rospy.spin()        
     