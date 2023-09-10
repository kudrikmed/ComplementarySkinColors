import os.path
import mtcnn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class FaceCropper:

    def __init__(self, image_path, save_crop="face.jpg"):
        self.image_path = image_path
        self.save_crop = save_crop

    def extract_face_from_image(self):
        # Load and transpose the image
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
            # Extract the bounding box 50% more from the requested face
            x, y, w, h = face['box']
            b = max(0, y - (h // 2))
            d = min(image.shape[0], (y + h) + (h // 2))
            a = max(0, x - (w // 2))
            c = min(image.shape[1], (x + w) + (w // 2))
            face_boundary = image[b:d, a:c, :]

            # Save the face image
            plt.imsave(self.save_crop, face_boundary)
            return 1
        elif len(faces) == 0:
            return 0
        elif len(faces) > 1:
            return 2
        else:
            return "Unknown error while face detection"
