import numpy as np
import cv2
from matplotlib import pyplot as plt
bgr = cv2.imread('x0KZ8.jpg')
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
blur = cv2.bilateralFilter(bgr,9,75,75)
cv2.imwrite('clahe_bi.jpg',blur)