import cv2
import numpy as np


class SkinExtractor:
    @staticmethod
    def extract_skin(image):
        # Make a copy of the image to avoid modifying the original
        img = image.copy()

        # Convert from BGR color space to HSV
        img_hsv = SkinExtractor.convert_to_hsv(img)

        # Define HSV thresholds for detecting skin color
        lower_threshold, upper_threshold = SkinExtractor.define_skin_thresholds()

        # Create a binary mask indicating the presence of skin colors
        skin_mask = SkinExtractor.create_skin_mask(img_hsv, lower_threshold, upper_threshold)

        # Clean up the mask using Gaussian blur
        skin_mask = SkinExtractor.clean_skin_mask(skin_mask)

        # Extract the skin from the threshold mask
        skin = cv2.bitwise_and(img, img, mask=skin_mask)

        # Convert the skin image back to BGR color space
        skin_bgr = SkinExtractor.convert_to_bgr(skin)

        return skin_bgr

    @staticmethod
    def convert_to_hsv(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def define_skin_thresholds():
        lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
        upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
        return lower_threshold, upper_threshold

    @staticmethod
    def create_skin_mask(image_hsv, lower_threshold, upper_threshold):
        return cv2.inRange(image_hsv, lower_threshold, upper_threshold)

    @staticmethod
    def clean_skin_mask(mask):
        return cv2.GaussianBlur(mask, (3, 3), 0)

    @staticmethod
    def convert_to_bgr(image_hsv):
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
