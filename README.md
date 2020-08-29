## YOLOv3 ROS object detection service

**object_detection_server.py** spins a rosservice that performs object detection on a single Image using YOLOv3 and PYTORCH.

Datasets can be specified at service call, such that multiple sets of classes can be detected using one server.

## Installation
    git clone --recursive git@gitlab.com:sensible-robots/lasr_object_detection_yolo.git
    git fetch
    git checkout pytorch
    catkin build lasr_object_detection_yolo

## Pretrained models
Download at: https://leeds365-my.sharepoint.com/:f:/g/personal/sc18j3j_leeds_ac_uk/ElRLuIOCNpxDgXhASJbXf2EBKQ2y5aFoe-nMe49_bkAniQ?e=Ha8TWo

Place them in the **models** directory at the root of this package.

## Loading new datasets

To add a dataset for object detection, create a directory with the intended name of the dataset in the **models** directory.

For each dataset folder, the following files are required:
* classes.txt
* yolov3.weights
* yolov3.cfg

To generate the **classes.txt**, **yolov3.cfg** files and train **yolov3.weights** for a new custom dataset, refer to https://gitlab.com/joejeffcock/lasr_darknet_config

## Usage

Ensure the **object_detection_server.py** server node is running

    rosrun lasr_object_detection_yolo object_detection_server.py


In the client, create a handle to the service

    detect_objects = rospy.ServiceProxy('/yolo_detection', YoloDetection)
    
Then call the service with the following arguments:

    result = detect_objects(image_msg, dataset_name, confidence, nms)

| Type | Name | Description |
| ------ | ------ | ------ |
| sensor_msgs.msg/Image | image_msg | message to perform object detection on |
| string | dataset_name | name of dataset to use |
| float | confidence | confidence threshold for detection |
| float | nms | threshold for non-maximum suppression on bounding boxes |

## Output

| Type | Name | Description |
| ------ | ------ | ------ |
| sensor_msgs/Image | image_bb | Image with bounding boxes for detected objects |
| Detection[] | detected_objects | Array of Detection messages |

Each Detection is composed of name, confidence and bounding box dimensions:
* string name
* float32 confidence
* int32[] xywh