import numpy as np
import matplotlib.pyplot as plt
import cv2
#path = r'C:\Users\aditi\Documents\Py_Proj\file-20200407-18916-1p3qplf.jpg'
#img =cv2.imread(path,1)
#resized_image = cv2.resize(img, (700,700)) 
#cv2.imshow('image', resized_image)
#print(resized_image.shape)
#print(img[0])

#So, when we read any image using OpenCV it returns object of numpy array by
#default and using img.shape we are checking the height and width of image and
#also it returns 3 which is the color channel of image.
#Now we can see the values of array are the color values actually.

#print(plt.imshow(resized_image))

haar_data = cv2.CascadeClassifier('data.xml')
#print(haar_data.detectMultiScale(resized_image))

#This code returns x, y, width and height of the face detected in the image.
#And we can draw a rectangle on the face using this code:

'''while True:
    faces = haar_data.detectMultiScale(resized_image)
    for x,y,w,h in faces:
        cv2.rectangle(resized_image, (x,y),(x+w, y+h), (255,0,255), 4)
    cv2.imshow('result',resized_image)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()'''

#Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
#---------------------------------------------------------------------

capture = cv2.VideoCapture(0)
data2 = [] #to store face data
while True:
    flag, img1 = capture.read() # read video frame by frame and return true/false and one frame at a time
    if flag: # will check if flag is true(camera available o not)
        faces = haar_data.detectMultiScale(img1)# detect face from frame
    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y),(x+w, y+h), (255,0,255), 4)
        face = img1[y:y+h, x:x+w, :] # slicing only face from the frame
        face = cv2.resize(face, (50,50)) # resizing all frames to 50x50
        print(len(data2))
        if len(data2) < 400:
            data2.append(face) # storing face data        
    cv2.imshow('result',img1)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()

np.save('without1_mask.npy',data2)



