import os.path
import mtcnn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class FaceCropper:
    """
    A class for extracting and cropping a human face from an input image using MTCNN face detection.

    Attributes:
        image_path (str): The path to the input image.
        save_crop (str): The filename to save the cropped face image. Default is "face.jpg".

    Methods:
        extract_face_from_image(self) -> int:
            Extracts a face from the input image, saves it, and returns a status code.

    Example Usage:
        # Create a FaceCropper instance
        cropper = FaceCropper("input_image.jpg")

        # Perform face extraction and check the result code
        result_code = cropper.extract_face_from_image()

        if result_code == 1:
            print("Face extracted and saved successfully.")
        elif result_code == 0:
            print("No faces were detected in the input image.")
        elif result_code == 2:
            print("Multiple faces were detected in the input image.")
        else:
            print("An error occurred during face extraction.")
    """

    def __init__(self, image_path, save_crop="face.jpg"):
        """
        Initialize the FaceCropper object with the input image path and save_crop filename.

        Args:
            image_path (str): The path to the input image.
            save_crop (str, optional): The filename to save the cropped face image. Default is "face.jpg".
        """
        self.image_path = image_path
        self.save_crop = save_crop

    def extract_face_from_image(self) -> int:
        """
        Extracts a face from the input image using MTCNN and saves it as an image file.

        Returns:
            int: A status code indicating the result of the extraction process.
                - 1: Face successfully extracted and saved.
                - 0: No faces detected in the input image.
                - 2: Multiple faces detected in the input image.
                - -1: An error occurred during face extraction.
        """
        # Load and transpose the input image
        image = Image.open(self.image_path)
        image = ImageOps.exif_transpose(image)
        image.save(os.path.join(os.path.dirname(self.image_path), 'no_efix.jpeg'))
        image.close()

        # Load the transposed image
        image_path_no_efix = os.path.join(os.path.dirname(self.image_path), 'no_efix.jpeg')
        image = plt.imread(image_path_no_efix)

        # Initialize the face detector
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(image)

        if len(faces) == 1:
            face = faces[0]
            # Extract the bounding box 50% more from the detected face
            x, y, w, h = face['box']
            b = max(0, y - (h // 2))
            d = min(image.shape[0], (y + h) + (h // 2))
            a = max(0, x - (w // 2))
            c = min(image.shape[1], (x + w) + (w // 2))
            face_boundary = image[b:d, a:c, :]

            # Save the extracted face image
            plt.imsave(self.save_crop, face_boundary)
            return 1
        elif len(faces) == 0:
            return 0
        elif len(faces) > 1:
            return 2
        else:
            return -1
