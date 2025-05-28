# project
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.

```
DEVELOPED BY : VARSHINI D
reg no : 212223230234

```
## I)Perform ROI from an image:

```

import cv2
import numpy as np

image = cv2.imread('C:/Users/admin/Downloads/Digital images/WhatsApp Image 2025-03-29 at 11.19.38_b1ca8789.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('Original Image', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
roi_mask = np.zeros_like(image_rgb)
roi_mask[100:300, 200:400, :] = 255  
segmented_roi = cv2.bitwise_and(image_rgb, roi_mask)
cv2.imshow('Segmented ROI', segmented_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


```
## II)Perform handwritting detection in an image:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'C:/Users/admin/Pictures/Screenshots/Screenshot 2025-05-20 231731.png'
detect_handwriting(image_path)
```
## III)Perform object detection with label in an image:
```
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))
img=cv2.imread('cow.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

```

## OUTPUT:

### I)Perform ROI from an image:

![Screenshot 2025-05-27 204026](https://github.com/user-attachments/assets/8970b35b-b58d-4f44-82f0-a43edbabf014)

![Screenshot 2025-05-27 204036](https://github.com/user-attachments/assets/0004c15e-b6fa-4511-bcb8-7c5b47239998)

### II)Perform handwritting detection in an image:

![image](https://github.com/user-attachments/assets/3121affe-a1ee-4ed4-8481-5dbfb3a85613)

### III)Perform object detection with label in an image:

![image](https://github.com/user-attachments/assets/7836b7d4-0f47-4cae-812f-884c245e2998)

## Result:
Thus, a python program using OpenCV for following image manipulations is done successfully
