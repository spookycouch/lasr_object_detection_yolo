# YOLOv3 ROS object detection service

**object_detection_server.py** spins a rosservice that performs object detection on a single Image using YOLOv3 and OpenCV's deep neural network module. <br/>
Datasets can be specified at service call, such that multiple sets of classes can be detected using one server.

# Software requirements

OpenCV > 4.0 is required for YOLOv3 DNN support. I have built and installed version 4.0.0 in the package for use in the LASR singularity container, however it is **recommended** to rebuild
on other environments as installation is machine-specific.

If OpenCV > 4.0 is not imported by default, ensure that the path in line 5 points to a folder containing the installed **cv2.so** shared object file generated from the 4.0 build. <br/>
If OpenCV > 4.0 is the default import, line 5 can be commented out.

# Installation
Clone the package into the catkin workspace and run catkin_make or catkin build (where appropriate) to generate the .msg and .srv headers
> This repo include opencv4 pre-built files and custom models, see [lasr_object_detection_yolo-lite](https://gitlab.com/sensible-robots/lasr_object_detection_yolo-lite) to get a pure repo and it can help you reduce around 1.3GB space usage.

# Loading new datasets

To add a dataset for object detection, create a directory with the intended name of the dataset in the **models** folder.

For each dataset folder, the following files are required:
* classes.txt
* yolov3.weights
* yolov3.cfg

To generate the **classes.txt**, **yolov3.cfg** files and train **yolov3.weights** for a new custom dataset, refer to https://gitlab.com/joejeffcock/lasr_darknet_config

# Usage

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

# Output

| Type | Name | Description |
| ------ | ------ | ------ |
| sensor_msgs/Image | image_bb | Image with bounding boxes for detected objects |
| Detection[] | detected_objects | Array of Detection messages |

Each Detection is composed of name, confidence and bounding box dimensions:
* string name
* float32 confidence
* float32[] xywh



