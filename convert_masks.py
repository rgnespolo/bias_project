import cv2, os
import numpy as np

listOfFiles = os.listdir('./CADIS/Masks_dark_iris/')
store = []
for entry in listOfFiles:
    print(entry)
    img = cv2.imread('./CADIS/Masks_dark_iris/'+ entry)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    img[np.where((img==[4,4,4]).all(axis=2))] = [255,255,255]
    img[np.where((img!=[255,255,255]).all(axis=2))] = [0,0,0]
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    cv2.imwrite(entry, img)


# Index	Class
# 0	Pupil
# 1	Surgical Tape
# 2	Hand
# 3	Eye Retractors
# 4	Iris
# 5	Skin
# 6	Cornea
# 7	Hydrodissection Cannula
# 8	Viscoelastic Cannula
# 9	Capsulorhexis Cystotome
# 10	Rycroft Cannula
# 11	Bonn Forceps
# 12	Primary Knife
# 13	Phacoemulsifier Handpiece
# 14	Lens Injector
# 15	I/A Handpiece
# 16	Secondary Knife
# 17	Micromanipulator
# 18	I/A Handpiece Handle
# 19	Capsulorhexis Forceps
# 20	Rycroft Cannula Handle
# 21	Phacoemulsifier Handpiece Handle
# 22	Capsulorhexis Cystotome Handle
# 23	Secondary Knife Handle
# 24	Lens Injector Handle
# 25	Suture Needle
# 26	Needle Holder
# 27	Charleux Cannula
# 28	Primary Knife Handle
# 29	Vitrectomy Handpiece
# 30	Mendez Ring
# 31	Marker
# 32	Hydrodissection Cannula Handle
# 33	Troutman Forceps
# 34	Cotton
# 35	Iris Hooks
