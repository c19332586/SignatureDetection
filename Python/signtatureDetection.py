'''
Signature Detection Program 
Aran, Vlad and Catriona

This is a program that has three features:
1. Extract signature from document
2. Extract signature using camera
3. Remove signature from document

The program uses OpenCV and Scikit-image libraries to extract the signature from the document. 
It also uses various techniques to enhance the signature and remove the background noise.

1. Extract signature from document
This feature extracts the signature from the document. 
morphology.remove_small_objects() function is used to remove the small objects from the image, ensuring imperfections are removed.
A threshold is applied to the image. The user has an option to crop out the largest signature from the document.
When the user selects yes to crop out the signature, the largest signature is cropped out and displayed. 
This is done by using findContours and cropping a rectangle around the largest contour.
If the user selects no, the signatures are displayed in the original document with the text removed.

2. Extract signature using camera
This feature extracts the signature a document using the camera. 
The user is asked to place the document in front of the camera and when the user hits the space bar, a snapshot is taken.
The adaptiveThreshold is applied to the snapshot and the background noise is removed using morphology.remove_small_objects() function from the skimage library.
findContours is used to find the countours in the snapshot and the largest contour is cropped out and displayed.

3. Remove signature from document
This feature removes the signature from the document. A threshold is applied to the image and 
the background noise is removed using morphology.remove_small_objects() function from the skimage library.
bitwise_xor() and bitwise_not() functions are used to remove the signature from the document.
'''



from tkinter.messagebox import YES
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np

# constant paramters that are used throughout the program
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100
constant_parameter_4 = 18

# readDocument() function reads in the document and returns the image
def readDocument():
    img = cv2.imread('pics/doc2.png', 0)
    img = cv2.resize(img, dsize = (600, 800))
    displayImage(img)
    return img

# displayImage() function displays the image
def displayImage(img): 
    cv2.imshow("Image", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# extractBlobs() function extracts the blobs from the image
# skimage.measure.label() function is used to label the blobs
def extractBlobs(img):
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    return blobs_labels

# averageRegionSize() function calculates the average region size of the blobs in the image
def averageRegionSize(blobs_labels):
    blobs_area = 0
    counter = 0
    average = 0.0

    for blobs in regionprops(blobs_labels):
        if (blobs.area > 10):
            blobs_area = blobs_area + blobs.area
            counter = counter + 1
    
    average = (blobs_area/counter)
    return average

# extractSignatureDocument() function extracts the signature from the document
def extractSignatureDocument():

    img = readDocument()
    # apply threshold 
    thres_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
   
    # extract small objects from the image
    blobs_labels = extractBlobs(thres_image)
    average = averageRegionSize(blobs_labels)
    
    # calculate outliar size
    small_outliar = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
    big_outliar = small_outliar*constant_parameter_4
    
    # remove small objects from the image
    morph_image = morphology.remove_small_objects(blobs_labels, small_outliar)

    # set background to white
    component_sizes = np.bincount(morph_image.ravel())
    too_small = component_sizes > (big_outliar)
    too_small_mask = too_small[morph_image]
    morph_image[too_small_mask] = 0
    
    plt.imsave('pics/morph_image.png', morph_image)

    img = cv2.imread('pics/morph_image.png', 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # resize image to fit the screen
    img = cv2.resize(img, dsize = (600, 800))

    selection2=str(input("Would you like to crop out the signature? Yes/No ")).upper()

    if selection2 == "YES":
        cropping_image = img
        inverse_mask = cv2.bitwise_not(img)

        # find contours
        contours,_ = cv2.findContours(inverse_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        # find the largest contour
        cont = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cont)
    
        # crop out the largest contour
        cv2.rectangle(inverse_mask,(x,y),(x+w,y+h),(0,255,0),2)                           
        cropped_signature = cropping_image[y:y+h,x:x+w]
        
        displayImage(cropped_signature)
        
        main()


    elif selection2 == "NO":
        displayImage(img)
        main()
    


def extractSignatureCamera():
    # open users camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "pics/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            img_counter += 1
            break

    cam.release()
    img = cv2.imread('pics/opencv_frame_0.png', 0)

    # apply threshold
    thres_image = cv2.adaptiveThreshold(img, maxValue = 255, 
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType = cv2.THRESH_BINARY,
    blockSize = 5,C = 2)

    # extract small objects from the image
    blobs_labels = extractBlobs(thres_image)
    average = averageRegionSize(blobs_labels)

    # calculate outliars
    small_outliar = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
    big_outliar = small_outliar*constant_parameter_4

    # remove small objects from the image
    morph_image = morphology.remove_small_objects(blobs_labels, small_outliar)

    # set background to white
    component_sizes = np.bincount(morph_image.ravel())
    too_small = component_sizes > (big_outliar)
    too_small_mask = too_small[morph_image]
    morph_image[too_small_mask] = 0

    plt.imsave('pics/morph_image2.png', morph_image)
    img = cv2.imread('pics/morph_image2.png', 0)

    thres_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cropping_image = thres_image.copy()

    inverse_mask = cv2.bitwise_not(thres_image)
    # cropping_image = img.copy()
  
    # find contours
    contours,_ = cv2.findContours(inverse_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    inverse_mask = cv2.drawContours(thres_image, contours, contourIdx=-1, 
                                        color=(0,0,255), thickness=5)

    # find the largest contour
    cont = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cont)

    cv2.rectangle(thres_image,(x,y),(x+w,y+h),(0,255,0),2)

    # crop out the largest contour
    cropped_signature = cropping_image[y:y+h,x:x+w] 
    displayImage(cropped_signature)

    main()


def removeSignatureDocument():
    
    img = readDocument()
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
    
    # extract small objects from the image
    blobs_labels = extractBlobs(img)
    average = averageRegionSize(blobs_labels)

    a4_constant = ((average/84.0)*250.0)+100

    # remove small objects from the image
    morph_image = morphology.remove_small_objects(blobs_labels, a4_constant)
    plt.imsave('pics/morph_image3.png', morph_image)

    img2 = cv2.imread('pics/morph_image3.png', 0)
    img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite("pics/output.png", img2)

    # add two images
    xor_image = cv2.bitwise_xor(img,img2)
    not_image = cv2.bitwise_not(xor_image)

    signature = cv2.resize(not_image, dsize = (600, 800))
    displayImage(signature)

    main()


def main():

    # display menu
    print("Selection Menu:")
    print("     1. Extract signature from document")
    print("     2. Extract signature using camera")
    print("     3. Remove signature from document")
    print("     4. Exit")

    # get user input
    selection=int(input("Enter choice:"))
    if selection == 1:
        extractSignatureDocument()
    elif selection == 2:
        extractSignatureCamera()
    elif selection == 3:
        removeSignatureDocument()
    elif selection == 4:
        exit
    else:
        print("Invalid input. Please try again.")
        main()

if __name__ == "__main__":
    main()