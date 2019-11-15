import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.rectangle(img,(222,222),(510,128),(0,255,0),3)
cv2.imshow('line',img)
cv2.waitKey()