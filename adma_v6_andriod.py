import cv2
import numpy as np
from matplotlib import pyplot as plt
f = cv2.imread('C:\\Users\\Lenovo\\Desktop\\16.png')
image_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
fg = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
I3 = cv2.filter2D(src=fg, ddepth=-1, kernel=kernel)
I4 = cv2.adaptiveThreshold(I3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 23)
I5 = cv2.bitwise_not(I4)
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
I6 = cv2.dilate(I5, se1)
plt.imshow(I6, cmap='gray')
plt.show()
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(I6, None, None, None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 30:
        res = np.where(labels == i+1)
        result[res[0], res[1]] = 255
cv2.imshow("cleaned", result)
cv2.waitKey(0)
cv2.imwrite('out_clean2.png',result)
