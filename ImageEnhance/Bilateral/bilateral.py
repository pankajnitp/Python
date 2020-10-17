import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('taj.jpg')
blur = cv.bilateralFilter(img,15,75,75)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
cv.imwrite('bilateral_output.jpg',blur)
plt.show()