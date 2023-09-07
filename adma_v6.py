import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import BernoulliRBM as rbm

f1 = cv2.imread('D:\\OPTISOL\\textdetect\\data\\all challenges\\metal plates\\cropped\\12.png')
std_3 = cv2.mean(f1, mask=None)
mean3 = np.mean(std_3)
if mean3>100:
    th=1
else:
     th=2
f = cv2.fastNlMeansDenoisingColored(f1,None,20,20,7,th)
image_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
fg = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
I3 = cv2.filter2D(src=fg, ddepth=-1, kernel=kernel)
mean, std_1 = cv2.meanStdDev(I3, mask=None)
std_2 = int(std_1)
I4 = cv2.adaptiveThreshold(I3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, std_2)
I5 = cv2.bitwise_not(I4)
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
I6 = cv2.dilate(I5, se1)
plt.imshow(I6, cmap='gray')
plt.show()
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(I6, None, None, None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 5:
        result[labels == i + 1] = 255
#noiseless_image_bw = cv2.fastNlMeansDenoising(result, None, 20, 7, 1)
cv2.imshow("cleaned", result)
cv2.waitKey(0)
cv2.imwrite('metal_localmean12.png',result)
