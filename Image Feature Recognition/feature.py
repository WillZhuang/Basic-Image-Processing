import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import copy

plt.figure(figsize=(12, 12))
# reading image file
img_gray = cv2.imread("logo.jpg", 1)

h, w = img_gray.shape[:2]

grad_img = np.asarray(img_gray)

for i in range(0, h):
    for j in range(0, w - 1):

        # applying gradient
        t1 = img_gray[i][j + 1]
        t2 = img_gray[i][j]
        a = min(t1.all(), t2.all())
        if a == t1.all():
            temp_arr = t2 - t1
        else:
            temp_arr = t1 - t2

        grad_img[i, j] = temp_arr

img = Image.fromarray(grad_img)
img.save("gradient.jpg")

#
# # function for sharpening filter
# def sharpenFiltering(img):
#     inputImg = copy.deepcopy(img.astype(np.float))
#     # converting color scale from BGR to GRAY
#     inputImg = cv2.cvtColor(img, cv2.COLOR_ .COLOR_BGR2GRAY)
#     # initialize black image of size equal to given image
#     outputImg = np.zeros(inputImg.shape)
#     # padding the image with zeros
#     inputImg = np.pad(inputImg, (1, 1), 'constant', constant_values=(0))
#     # creating two filters for horizontal and vertical edge detection
#     fh = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
#     fy = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
#     # looping through image pixels
#     for row in range(1, inputImg.shape[0] - 1):
#         for col in range(1, inputImg.shape[1] - 1):
#             dx, dy = 0.0, 0.0
#             # convolving both filters
#             for x_filter in range(3):
#                 for y_filter in range(3):
#                     dx += inputImg[row + x_filter - 1][col + y_filter - 1] * fh[x_filter][y_filter]
#                     dy += inputImg[row + x_filter - 1][col + y_filter - 1] * fy[x_filter][y_filter]
#
#             # magnitude of gradient (instead of just adding dx and dy. we calculate magnitude)
#             pixel = np.sqrt(dx * dx + dy * dy)
#             outputImg[row - 1][col - 1] = pixel
#     # normalizing pixels
#     outputImg *= 255.0 / np.max(outputImg)
#     return outputImg
#
#
# # applying sharpen filters
# output = sharpenFiltering(im)
# # writing image to image file
# cv2.imwrite("sharpen.jpg", output)
# # converting color scale from BGR to RGB
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# # plotting original image
# plt.subplot(211)
# plt.axis('off')
# plt.title("Original Image")
# plt.imshow(im)
# # plotting transformed image
# plt.subplot(212)
# plt.axis('off')
# plt.title("Transformed Image")
# plt.imshow(output, cmap="gray")
#
# plt.show()