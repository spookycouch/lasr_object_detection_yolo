#!/usr/bin/python

import numpy as np

import rospy
import actionlib
from sensor_msgs.msg import Image
from lasr_object_detection_yolo.srv import Pcl2ToImage

def extract_image(req):
    bridge = CvBridge()
    frame = np.fromstring(req.depth_points.data, dtype=np.uint8)
    frame = frame.reshape(req.depth_points.height, req.depth_points.width,32)
    frame = frame[:,:,16:19]
    imgmsg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    return imgmsg

def pcl2_to_image_server():
    rospy.init_node('pcl2_to_image')
    serv = rospy.Service('pcl2_to_image', Pcl2ToImage, extract_image)
    rospy.loginfo('Pcl2 to Image service initialised')
    rospy.spin()

if __name__ == '__main__':
    pcl2_to_image_server()
