#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PIL_Image
import time

from lasr_object_detection_yolo.msg import Detection
from lasr_object_detection_yolo.srv import YoloDetection, YoloDetectionResponse

import torch
import torchvision.transforms as transforms
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

import rospkg
import sys
sys.path.append(rospkg.RosPack().get_path('lasr_object_detection_yolo') + '/src/PyTorch-YOLOv3')

from models_python2 import Darknet
from utils.utils import non_max_suppression

MODEL_ROOT = rospkg.RosPack().get_path('lasr_object_detection_yolo') + '/models/'
from os.path import isdir

def yolo_transform():
    return transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])

class object_detection_server():
    def __init__(self):
        self.bridge = CvBridge()
        self.transform = yolo_transform()
        self.model_name = None
        self.load_model('coco')
    
    def load_model(self, model_name):
        model_path = MODEL_ROOT + model_name + '/'

        if isdir(model_path):
            self.model_name = model_name

            start_time = time.time()

            self.labels = open(model_path + 'classes.txt').read().strip().split('\n')
            self.yolov3 = Darknet(model_path + 'yolov3.cfg')
            self.yolov3.load_darknet_weights(model_path + 'yolov3.weights')
            self.yolov3.eval()
            self.yolov3.to(DEVICE)
        
            rospy.loginfo('Time to load {} model: {:.2f} seconds'.format(model_name, time.time() - start_time))
        else:
            self.model_name = None
            self.yolov3 = None

    def detect_objects(self, req):
        # load model if it is not already up
        if not self.model_name == req.dataset:
            self.load_model(req.dataset)

        assert self.yolov3 is not None
        
        # preprocess
        frame = self.bridge.imgmsg_to_cv2(req.image_raw, 'rgb8') # opencv images are bgr - we want rgb
        image = PIL_Image.fromarray(frame)
        image = torch.stack([self.transform(image)]).to(DEVICE)

        # net forward and nms
        outputs = self.yolov3(image)
        outputs = non_max_suppression(outputs, nms_thres=req.nms)

        # build response
        # see: YoloDetection.srv, Detection.msg
        response = YoloDetectionResponse()
        if not outputs[0] is None:
            for detection in outputs[0].tolist():
                bbox = detection[:4]
                bbox[0] *= req.image_raw.width/416.0
                bbox[1] *= req.image_raw.height/416.0
                bbox[2] *= req.image_raw.width/416.0
                bbox[3] *= req.image_raw.height/416.0
                x1,y1,x2,y2 = [int(i) for i in bbox]

                obj_conf, class_score = detection[4:6]
                class_pred = int(detection[6])

                if class_score > req.confidence:
                    xywh = [x1, y1, x2 - x1, y2 - y1]
                    response.detected_objects.append(Detection(name=self.labels[class_pred], confidence=class_score, xywh=xywh))
        
        return response
                

if __name__ == '__main__':        
    rospy.init_node('yolo_detection')
    server = object_detection_server()
    serv = rospy.Service('yolo_detection', YoloDetection, server.detect_objects)
    rospy.loginfo('YOLOv3 object detection service initialised')
    rospy.spin()