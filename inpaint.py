import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread('mask.png', 0)

dst_TELEA = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

plt.subplot(221), plt.imshow(img)
plt.title('img')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.title('mask')
plt.subplot(223), plt.imshow(dst_TELEA)
plt.title('TELEA')
plt.subplot(224), plt.imshow(dst_NS)
plt.title('NS')

plt.tight_layout()
plt.show()
