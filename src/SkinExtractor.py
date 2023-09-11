import cv2
import numpy as np


class SkinExtractor:
    """
    A class for extracting skin regions from an input image based on color analysis in the HSV color space.

    Methods:
        extract_skin(image) -> numpy.ndarray:
            Extracts the skin regions from the input image and returns the skin image in BGR color space.

    Static Methods:
        convert_to_hsv(image) -> numpy.ndarray:
            Converts an input image from the BGR color space to the HSV color space.

        define_skin_thresholds() -> (numpy.ndarray, numpy.ndarray):
            Defines the lower and upper HSV color thresholds for detecting skin color.

        create_skin_mask(image_hsv, lower_threshold, upper_threshold) -> numpy.ndarray:
            Creates a binary mask indicating the presence of skin colors in the HSV image.

        clean_skin_mask(mask) -> numpy.ndarray:
            Cleans up the binary mask using Gaussian blur.

        convert_to_bgr(image_hsv) -> numpy.ndarray:
            Converts an HSV image back to the BGR color space.

    Example Usage:
        # Create a SkinExtractor instance
        extractor = SkinExtractor()

        # Load an input image
        input_image = cv2.imread("input_image.jpg")

        # Extract skin regions from the input image
        skin_image = extractor.extract_skin(input_image)

        # Display or save the skin image as needed
        cv2.imshow("Skin Image", skin_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    @staticmethod
    def extract_skin(image):
        """
        Extracts skin regions from the input image.

        Args:
            image (numpy.ndarray): The input image in BGR color space.

        Returns:
            numpy.ndarray: The skin image in BGR color space.
        """
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
        skin = cv2.bitwise_and(img_hsv, img_hsv, mask=skin_mask)

        # Convert the skin image back to BGR color space
        skin_bgr = SkinExtractor.convert_to_bgr(skin)

        return skin_bgr

    @staticmethod
    def convert_to_hsv(image):
        """
        Converts an input image from the BGR color space to the HSV color space.

        Args:
            image (numpy.ndarray): The input image in BGR color space.

        Returns:
            numpy.ndarray: The input image converted to HSV color space.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def define_skin_thresholds():
        """
        Defines the lower and upper HSV color thresholds for detecting skin color.

        Returns:
            tuple: A tuple containing the lower and upper threshold numpy arrays.
        """
        lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
        upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
        return lower_threshold, upper_threshold

    @staticmethod
    def create_skin_mask(image_hsv, lower_threshold, upper_threshold):
        """
        Creates a binary mask indicating the presence of skin colors in the HSV image.

        Args:
            image_hsv (numpy.ndarray): The input image in HSV color space.
            lower_threshold (numpy.ndarray): The lower HSV color threshold for skin detection.
            upper_threshold (numpy.ndarray): The upper HSV color threshold for skin detection.

        Returns:
            numpy.ndarray: A binary mask indicating skin regions.
        """
        return cv2.inRange(image_hsv, lower_threshold, upper_threshold)

    @staticmethod
    def clean_skin_mask(mask):
        """
        Cleans up the binary mask using Gaussian blur.

        Args:
            mask (numpy.ndarray): A binary mask.

        Returns:
            numpy.ndarray: The cleaned binary mask.
        """
        return cv2.GaussianBlur(mask, (3, 3), 0)

    @staticmethod
    def convert_to_bgr(image_hsv):
        """
        Converts an HSV image back to the BGR color space.

        Args:
            image_hsv (numpy.ndarray): The input image in HSV color space.

        Returns:
            numpy.ndarray: The input image converted to BGR color space.
        """
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
