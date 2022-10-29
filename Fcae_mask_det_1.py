import numpy as np
import cv2

with1_mask = np.load('with1_mask.npy')
without1_mask =np.load('without1_mask.npy')

print(with1_mask.shape)
print(without1_mask.shape)

with1_mask = with1_mask.reshape(400,50*50*3)
without1_mask = without1_mask.reshape(400,50*50*3)

'''Now we can see that data is loaded with the shape 200, 50, 50, 3.

Here 200 is the number of images we have collected
50, 50 is the size of each image
3 is the color channel (red, green, blue)
We can reshape the data to make it 2D :'''

print(with1_mask.shape)
print(without1_mask.shape)

#And we will concatenate the data into a single array :

X = np.r_[with1_mask, without1_mask]
print(X.shape)

'''Using NPR will help you to store data row wise. So our features are ready.
Now we need target variable. So let’s create one array of zeros and assign first
200 indexes as zero and next 200 indexes as one. Because first 200 images belong
to faces with mask and next 200 images belong to faces without mask.'''

labels = np.zeros(X.shape[0])
labels[400:] = 1.0
names = {0:'Mask', 1: 'No Mask'}

#apply machine learning on our data after dividing it into train and test.

#svm - Support Vector Machine
#SVC - Support Vector Classification

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.25)

#The algorithm we are using is SVM here.
#And after training this data on SVM we are getting accuracy of 98%.


'''from sklearn.decomposition import PCA
pca = PCA(n_components=3) 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.20)
'''
svm = SVC()
svm.fit(x_train, y_train)

#x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print("accuracy",accuracy_score(y_test, y_pred))

haar_data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)
data1 = [] #to store face data
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img1 = capture.read() # read video frame by frame and return true/false and one frame at a time
    if flag: # will check if flag is true(camera available o not)
        faces = haar_data.detectMultiScale(img1)# detect face from frame
    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y),(x+w, y+h), (255,0,255), 4)
        face = img1[y:y+h, x:x+w, :] # slicing only face from the frame
        face = cv2.resize(face, (50,50)) # resizing all frames to 50x50
        face = face.reshape(1,-1)
        #face = pca.transform(face)
        pred =svm.predict(face)
        n=names[int(pred)]
        cv2.putText(img1, n, (x,y), font, 1, (244,250,250), 2)
        print(n)
    cv2.imshow('result',img1)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()
