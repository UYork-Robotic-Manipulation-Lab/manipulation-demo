# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:57:43 2023
YOLO trial with webcam
"""
#imports
import cv2
import numpy as np

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

# Load the classes file
classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()
print(classes)

# Access the default camera device (index 0) to capture frames
cap = cv2.VideoCapture(0)#we can change the video capture device by changing the number

# Read frames from the camera device
while cap.isOpened():
    _, img = cap.read()
    
    # Get the height, width, and channels of the captured image
    height, width, _ = img.shape
    
    # Prepare the image to be inputted to the neural network
    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0), swapRB=True,crop=False)
    net.setInput(blob)
    
    # Get the output layers of the neural network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    # Create empty lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Loop through each output layer and its detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Only keep detections with a confidence greater than 0.5
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x ,y ,w ,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                
    #print(len(boxes))  

     # Perform non-maximum suppression on the detected boxes     
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    
    #print(indexes.flatten())  
    
    # Draw a bounding box and label for each detected object
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    
        # Show the image with detected objects
    cv2.imshow('Webcam',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the camera device and close all windows
cap.release()
cv2.destroyAllWindows()

