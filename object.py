# Author Abhishek Biswas
# Date of Creation: 1/06/2022
# Project name: Detecting the enviorment stuff and extra stuffs

# we have imported the nessary libaray needed for the object detection which is open cv this libaray is naturally used for the 
# computer vision perpose and it was created by intel and we gonna use for this kitchen unitensis detction perpose
# the library is used for making label as well as for preprocess of image need for the model 

#cv2 --> opencv python
#numpy --> numeric python lib

import cv2
import numpy as np


# here we have just used a subset of classes
classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table","unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "burger",  "Tea",
  "Coffee", "mouse", "Mobile", "pen", "fan"]


# this command is used to give colour as the object label which we gonna use to detect
colors = np.random.uniform(0, 255, size = (len(classes), 3))

# this commmed gonna trigger the webcam to be ON postion
cam = cv2.VideoCapture(1)


# loading the pb weight file of the model and the config file so, that the tensorflow graph can properly work in opencv for object detection purpose 
# here the pb represents the protocol buffer and protocol buffer text format for pbtxt 

pb = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

# reading our dense nural network file using the function cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)

cvNet = cv2.dnn.readNetFromTensorflow(pb, pbt)

while(True):

    # reading the image from the web cam and strong the rows and cols index in the variable called rows and cols 
    ret_val, img = cam.read()
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size = (300, 300), swapRB = True, crop = False))

    # Now running the object detection pre-trained model
    cvOut = cvNet.forward()

    # Go through each object detected and label it
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])

        if score > 0.3:
            idx = int(detection[1])

            left = detection[3]*cols
            top = detection[4]*rows
            right = detection[5]*cols
            bottom = detection[6]*rows
            


            # This command is used for creating the rectangle around the object created and this is done using the libaray in cv2.rectangle
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness = 2)

            # draw the prediction on the frame
            # This command is done for making a label and also put the score in decimal format around the objected deteted which shown in the form of rectangle border 
            label = "{}: {:.2f}%".format(classes[idx],score * 100)
            y = top-15 if top -15 > 15 else top+15

            # This commad is used to put text around the created rectangle 
            cv2.putText(img, label, (int(left), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    # This command is used to show the final detcted image which is detected by our model 
    cv2.imshow('My Webcam', img)

    #This commnd is used to break the infinate loop and pressing of ESC of keyboard will do the job for you when ever we want to exit this application why 27??
    # as 27 represented as ESC in keyboard AS in ASCII format 
    if cv2.waitKey(1) == 27:
        break

# This command will tell the opencv libaray and the computer to shut down the webcam as the work has been done
cam.release()

# this command will destroy all the unnessary windows and shurt down the program
cv2.destroyAllWindows()
