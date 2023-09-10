import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
import mediapipe as mp
import pandas as pd


class SkinColor:

    def __init__(self, image):
        self.image = image

    @staticmethod
    def extract_face(image):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]
        face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
        df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])

        routes_idx = []

        p1 = df.iloc[0]["p1"]
        p2 = df.iloc[0]["p2"]

        for i in range(0, df.shape[0]):
            obj = df[df["p1"] == p2]
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]
            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)

        routes = []

        for source_idx, target_idx in routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
            relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))

            routes.append(relative_source)
            routes.append(relative_target)

        mask = np.zeros((image.shape[0], image.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(bool)

        out = np.zeros_like(image)
        out[mask] = image[mask]

        return out

    @staticmethod
    def extract_skin(image):
        # Taking a copy of the image
        img = image.copy()
        # Converting from BGR Colours Space to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Defining HSV Threadholds
        lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
        upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

        # Single Channel mask,denoting presence of colours in the about threshold
        skin_mask = cv2.inRange(img, lower_threshold, upper_threshold)

        # Cleaning up mask using Gaussian Filter
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        # Extracting skin from the threshold mask
        skin = cv2.bitwise_and(img, img, mask=skin_mask)

        # Return the Skin image
        return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

    @staticmethod
    def remove_black(estimator_labels, estimator_cluster):
        # Check for black
        has_black = False

        # Get the total number of occurance for each color
        occurance_counter = Counter(estimator_labels)

        # Quick lambda function to compare to lists
        compare = lambda x, y: Counter(x) == Counter(y)

        # Loop through the most common occuring color
        for x in occurance_counter.most_common(len(estimator_cluster)):

            # Quick List comprehension to convert each of RBG Numbers to int
            color = [int(i) for i in estimator_cluster[x[0]].tolist()]

            # Check if the color is [0,0,0] that if it is black
            if compare(color, [0, 0, 0]) == True:
                # delete the occurance
                del occurance_counter[x[0]]
                # remove the cluster
                has_black = True
                estimator_cluster = np.delete(estimator_cluster, x[0], 0)
                break

        return (occurance_counter, estimator_cluster, has_black)

    def get_color_information(self, estimator_labels, estimator_cluster, has_thresholding=False):
        # Output list variable to return
        color_information = []

        # Check for Black
        has_black = False

        # If a mask has be applied, remove black
        if has_thresholding:

            (occurance, cluster, black) = self.remove_black(estimator_labels, estimator_cluster)
            occurance_counter = occurance
            estimator_cluster = cluster
            has_black = black

        else:
            occurance_counter = Counter(estimator_labels)

        # Get the total sum of all the predicted occurances
        totalOccurance = sum(occurance_counter.values())

        # Loop through all the predicted colors
        for x in occurance_counter.most_common(len(estimator_cluster)):
            index = (int(x[0]))

            # Quick fix for index out of bound when there is no threshold
            index = (index - 1) if ((has_thresholding & has_black) & (int(index) != 0)) else index

            # Get the color number into a list
            color = estimator_cluster[index].tolist()

            # Get the percentage of each color
            color_percentage = (x[1] / totalOccurance)

            # make the dictionay of the information
            color_info = {"cluster_index": index, "color": color, "color_percentage": color_percentage}

            # Add the dictionary to the list
            color_information.append(color_info)

        return color_information

    def extract_dominant_color(self, image, number_of_colors=5, has_thresholding=False):
        # Quick Fix Increase cluster counter to neglect the black(Read Article)
        if has_thresholding:
            number_of_colors += 1

        # Taking Copy of the image
        img = image.copy()

        # Convert Image into RGB Colours Space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape Image
        img = img.reshape((img.shape[0] * img.shape[1]), 3)

        # Initiate KMeans Object
        estimator = KMeans(n_clusters=number_of_colors, random_state=0)

        # Fit the image
        estimator.fit(img)

        # Get Colour Information
        color_information = self.get_color_information(estimator.labels_, estimator.cluster_centers_, has_thresholding)
        return color_information

    def plot_color_bar(self, color_information):
        # Create a 500x100 black image
        color_bar = np.zeros((100, 500, 3), dtype="uint8")

        top_x = 0
        for x in color_information:
            bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

            color = tuple(map(int, (x['color'])))

            cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
            top_x = bottom_x
        return color_bar

    @staticmethod
    def plot_complementary_color_bar(color_information):
        # Create a 500x100 black image
        color_bar = np.zeros((100, 500, 3), dtype="uint8")

        top_x = 0
        for x in color_information:
            bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

            color = tuple(map(int, (x['color'])))
            # Calculate complementary colors
            complementary_color = list(color)
            complementary_color_1 = complementary_color.copy()
            complementary_color_2 = complementary_color.copy()
            complementary_color_3 = complementary_color.copy()
            complementary_color_1[1] = 255 - complementary_color_1[1]
            complementary_color_2[2] = 255 - complementary_color_2[2]
            complementary_color_3[1] = 255 - complementary_color_3[1]
            complementary_color_3[2] = 255 - complementary_color_3[2]

            cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), (color_bar.shape[0] // 4) * 1), color, -1)
            cv2.rectangle(color_bar, (int(top_x), (color_bar.shape[0] // 4) * 1), (int(bottom_x), color_bar.shape[0]),
                          complementary_color_1, -1)
            cv2.rectangle(color_bar, (int(top_x), (color_bar.shape[0] // 4) * 2), (int(bottom_x), color_bar.shape[0]),
                          complementary_color_2, -1)
            cv2.rectangle(color_bar, (int(top_x), (color_bar.shape[0] // 4) * 3), (int(bottom_x), color_bar.shape[0]),
                          complementary_color_3, -1)
            top_x = bottom_x
        return color_bar

    def draw_skin_colors_plot(self):
        # Get Image
        image = cv2.imread(self.image, cv2.COLOR_BGR2RGB)
        image = self.extract_face(image)

        # Apply Skin Mask
        skin = self.extract_skin(image)

        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
        dominant_colors = self.extract_dominant_color(skin, has_thresholding=True)
        colour_bar = self.plot_color_bar(dominant_colors)
        plt.axis("off")
        plt.imshow(colour_bar)
        plt.savefig('temp/skin_colors.png', bbox_inches='tight')
        plt.close()

    def draw_complentary_skin_colors_plot(self):
        # Get Image
        image = cv2.imread(self.image, cv2.COLOR_BGR2RGB)
        image = self.extract_face(image)

        # Apply Skin Mask
        skin = self.extract_skin(image)

        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
        dominant_colors = self.extract_dominant_color(skin, has_thresholding=True)
        # print(dominant_colors)

        # Show in the dominant color as bar
        complementary_colour_bar = self.plot_complementary_color_bar(dominant_colors)
        plt.axis("off")
        plt.imshow(complementary_colour_bar)
        plt.savefig('temp/complementary_skin_colors.png', bbox_inches='tight')
        plt.close()