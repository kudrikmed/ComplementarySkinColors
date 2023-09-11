import numpy as np
import mediapipe as mp
import cv2
import pandas as pd


class FaceExtractor:
    """
    A class for extracting a face region from an input image using the mediapipe FaceMesh model.

    Methods:
        extract_face(image) -> numpy.ndarray:
            Extracts and returns the face region from the input image.

    Static Methods:
        None

    Example Usage:
        # Create a FaceExtractor instance
        extractor = FaceExtractor()

        # Load an input image
        input_image = cv2.imread("input_image.jpg")

        # Extract the face from the input image
        face_image = extractor.extract_face(input_image)

        # Display or save the extracted face image as needed
        cv2.imshow("Face Image", face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    @staticmethod
    def extract_face(image):
        """
        Extracts a face region from the input image using the FaceMesh model.

        Args:
            image (numpy.ndarray): The input image in BGR color space.

        Returns:
            numpy.ndarray: The extracted face region in BGR color space.
        """
        # Initialize the FaceMesh model
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

        # Process the image to extract landmarks
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]

        # Define the face oval landmarks
        face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
        df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])

        # Create a list of route indices
        routes_idx = []

        p1, p2 = df.iloc[0]["p1"], df.iloc[0]["p2"]

        for _ in range(df.shape[0]):
            obj = df[df["p1"] == p2]
            p1, p2 = obj["p1"].values[0], obj["p2"].values[0]
            route_idx = [p1, p2]
            routes_idx.append(route_idx)

        # Create a list of route coordinates
        routes = []

        for source_idx, target_idx in routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
            relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))

            routes.extend([relative_source, relative_target])

        # Create a mask based on the routes
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)

        # Apply the mask to the input image
        face = cv2.bitwise_and(image, image, mask=mask)

        return face
