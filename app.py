# Required modules
import cv2
import numpy as np


# Values are taken from: 'RGB-H-CbCr Skin Colour Model for Human Face Detection'
# (R > 95) AND (G > 40) AND (B > 20) AND (max{R, G, B} − min{R, G, B} > 15) AND (|R − G| > 15) AND (R > G) AND (R > B)
# (R > 220) AND (G > 210) AND (B > 170) AND (|R − G| ≤ 15) AND (R > B) AND (G > B)
def rgb_skin(r, g, b):
    """Rule for skin pixel segmentation based on the paper 'RGB-H-CbCr Skin Colour Model for Human Face Detection'"""

    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
    abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2


# Skin detector based on the RGB color space
def skin_detector_rgb(rgb_image):
    """Skin segmentation based on the RGB color space"""
    h = rgb_image.shape[0]
    w = rgb_image.shape[1]
    # We crete the result image with back background
    res = np.zeros((h, w, 1), dtype="uint8")

    # Only 'skin pixels' will be set to white (255) in the res image:
    for y in range(0, h):
        for x in range(0, w):
            (r, g, b) = rgb_image[y, x]
            if rgb_skin(r, g, b):
                res[y, x] = 255
    skinBGR = cv2.bitwise_and(rgb_image, rgb_image, mask=res)
    return skinBGR


def visualization(image):
    import matplotlib.pyplot as plt
    # path = 'E:\\ZZAX\\Dataset\\test\\Vascular Tumors\\hemangioma-infancy-21.jpg'
    # image = cv2.imread(path)
    plt.imshow(image)
    plt.show()
