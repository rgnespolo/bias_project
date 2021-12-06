import cv2, os
import numpy as np

listOfFiles = os.listdir('./CADIS/Masks_iris/')
store = []
for entry in listOfFiles:
    print(entry)
    img = cv2.imread('./CADIS/Masks_iris/'+ entry)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    img[np.where((img==[4,4,4]).all(axis=2))] = [255,255,255]
    img[np.where((img!=[255,255,255]).all(axis=2))] = [0,0,0]
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    cv2.imwrite(entry, img)


