import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
from src.SkinExtractor import SkinExtractor
from src.FaceExtractor import FaceExtractor


class SkinColor:
    """
    A class for extracting and analyzing dominant skin colors from an input image.

    Methods:
        extract_dominant_color(image, number_of_colors=5, has_thresholding=False) -> list:
            Extracts and returns dominant skin colors from the input image.

        plot_color_bar(color_information) -> numpy.ndarray:
            Creates a color bar representation based on color information.

        plot_complementary_color_bar(color_information) -> numpy.ndarray:
            Creates a complementary color bar representation based on color information.

        draw_skin_colors_plot(num_dominant_colors=5):
            Generates and saves a plot of dominant skin colors from the input image.

        draw_complementary_skin_colors_plot(num_dominant_colors=5):
            Generates and saves a plot of complementary skin colors from the input image.

    Static Methods:
        extract_face(image) -> numpy.ndarray:
            Extracts and returns the face region from the input image.

        extract_skin(image) -> numpy.ndarray:
            Extracts and returns the skin region from the input image.

        remove_black(estimator_labels, estimator_cluster) -> (Counter, numpy.ndarray, bool):
            Removes black color from the cluster and updates occurrence and cluster accordingly.

        get_color_information(estimator_labels, estimator_cluster, has_thresholding=False) -> list:
            Retrieves and returns color information based on cluster labels and cluster centers.

    Example Usage:
        # Create a SkinColor instance with an input image
        skin_color = SkinColor("input_image.jpg")

        # Generate a plot of dominant skin colors
        skin_color.draw_skin_colors_plot()

        # Generate a plot of complementary skin colors
        skin_color.draw_complementary_skin_colors_plot()
    """

    def __init__(self, image):
        self.image = image
        """
        Initializes a SkinColor instance with the input image.

        Args:
            image (str): The path to the input image.
        """

    @staticmethod
    def extract_face(image):
        face = FaceExtractor.extract_face(image)
        return face

    @staticmethod
    def extract_skin(image):
        skin = SkinExtractor.extract_skin(image)
        return skin

    @staticmethod
    def remove_black(estimator_labels, estimator_cluster):
        """
        Removes black color from the cluster and updates occurrence and cluster accordingly.

        Args:
            estimator_labels (numpy.ndarray): The cluster labels assigned by KMeans.
            estimator_cluster (numpy.ndarray): The cluster centers representing color values.

        Returns:
            Tuple[Counter, numpy.ndarray, bool]: A tuple containing:
                - Counter: Occurrence of each color label.
                - numpy.ndarray: Updated cluster centers after black color removal.
                - bool: Whether black color was found and removed.
        """
        # Get the total number of occurrences for each color
        occurrence_counter = Counter(estimator_labels)

        # Iterate over the most common occurring colors
        for color_index, (count, _) in enumerate(occurrence_counter.most_common(len(estimator_cluster))):
            color = estimator_cluster[color_index].astype(int)

            # Check if the color is [0, 0, 0] (black)
            if np.all(color == [0, 0, 0]):
                # Delete the occurrence
                del occurrence_counter[color_index]
                # Remove the cluster
                estimator_cluster = np.delete(estimator_cluster, color_index, 0)
                return occurrence_counter, estimator_cluster, True

        return occurrence_counter, estimator_cluster, False

    def get_color_information(self, estimator_labels, estimator_cluster, has_thresholding=False):
        """
        Retrieves color information based on cluster labels and cluster centers.

        Args:
            estimator_labels (numpy.ndarray): The cluster labels assigned by KMeans.
            estimator_cluster (numpy.ndarray): The cluster centers representing color values.
            has_thresholding (bool, optional): Whether to apply color thresholding to exclude black color.
                Default is False.

        Returns:
            list: A list of dictionaries containing information about dominant colors, including:
                - 'cluster_index' (int): Index of the color cluster.
                - 'color' (list): RGB color values.
                - 'color_percentage' (float): Percentage of pixels with the dominant color.

        Note:
            If `has_thresholding` is set to True and black color is removed, the 'cluster_index' values
            are adjusted accordingly to maintain consistency with the original cluster labeling.
        """
        color_information = []

        if has_thresholding:
            occurrence, cluster, has_black = self.remove_black(estimator_labels, estimator_cluster)
            occurrence_counter = occurrence
            estimator_cluster = cluster
        else:
            occurrence_counter = Counter(estimator_labels)

        total_occurrence = sum(occurrence_counter.values())

        for x in occurrence_counter.most_common(len(estimator_cluster)):
            index = int(x[0])

            if has_thresholding and has_black and index != 0:
                index -= 1

            color = estimator_cluster[index].tolist()
            color_percentage = x[1] / total_occurrence

            color_info = {
                "cluster_index": index,
                "color": color,
                "color_percentage": color_percentage
            }

            color_information.append(color_info)

        return color_information

    def extract_dominant_color(self, image, number_of_colors=5, has_thresholding=False):
        """
        Extracts and returns dominant skin colors from the input image.

        Args:
            image (numpy.ndarray): The input image in BGR color space.
            number_of_colors (int, optional): The number of dominant colors to extract. Default is 5.
            has_thresholding (bool, optional): Whether to apply color thresholding to exclude black color.
                Default is False.

        Returns:
            list: A list of dictionaries containing information about dominant skin colors.
        """
        if has_thresholding:
            number_of_colors += 1

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((-1, 3))

        estimator = KMeans(n_clusters=number_of_colors, random_state=0)
        estimator.fit(img)

        color_information = self.get_color_information(estimator.labels_, estimator.cluster_centers_, has_thresholding)

        return color_information

    @staticmethod
    def plot_color_bar(color_information):
        """
        Creates a color bar representation based on color information.

        Args:
            color_information (list): A list of dictionaries containing color information.

        Returns:
            numpy.ndarray: A color bar image.
        """
        color_bar = np.zeros((100, 500, 3), dtype=np.uint8)

        top_x = 0
        for info in color_information:
            color = tuple(map(int, info['color']))
            bottom_x = int(top_x + round(info["color_percentage"] * color_bar.shape[1], 0))

            cv2.rectangle(color_bar, (int(top_x), 0), (bottom_x, color_bar.shape[0]), color, -1)
            top_x = bottom_x

        return color_bar

    @staticmethod
    def make_complementary_color(color, scheme):
        """
        Calculate and return the complementary color of the given color based on the specified scheme.

        Args:
            color (list[int]): A list representing the RGB color to calculate the complementary color for.
                               It should contain three integer values between 0 and 255, inclusive, representing
                               the red, green, and blue components of the color.
            scheme (str): A string specifying the color transformation scheme. Supported schemes are:
                          - "invert_green": Inverts the green component of the color.
                          - "invert_blue": Inverts the blue component of the color.
                          - "invert_green_blue": Inverts both the green and blue components of the color.

        Returns:
            list[int]: A list representing the RGB values of the complementary color.
                       The returned list will have the same format as the input color.

        Note:
            If an unsupported or invalid scheme is provided, the function will return the original color unchanged.

        Example:
            >>> original_color = [255, 128, 64]
            >>> complementary_color = MyClass.make_complementary_color(original_color, "invert_green_blue")
            >>> print(complementary_color)
            [255, 127, 191]
        """
        schemes = {
            "invert_green": [color[0], 255 - color[1], color[2]],
            "invert_blue": [color[0], color[1], 255 - color[2]],
            "invert_green_blue": [color[0], 255 - color[1], 255 - color[2]],
        }

        return schemes.get(scheme, color)

    def plot_complementary_color_bar(self, color_information):
        """
        Creates a complementary color bar representation based on color information.

        Args:
            color_information (list): A list of dictionaries containing color information.

        Returns:
            numpy.ndarray: A complementary color bar image.
        """
        # Initialize the color bar image
        color_bar_height = 100
        color_bar_width = 500
        color_bar = np.zeros((color_bar_height, color_bar_width, 3), dtype=np.uint8)
        bar_height = color_bar_height // 4

        top_x = 0

        # Iterate through color information
        for info in color_information:
            color = tuple(map(int, info['color']))
            bottom_x = int(top_x + round(info["color_percentage"] * color_bar_width))

            # Define color schemes
            color_schemes = ["original", "invert_green", "invert_blue", "invert_green_blue"]

            for i, scheme in enumerate(color_schemes):
                start_y = i * bar_height
                end_y = (i + 1) * bar_height
                if scheme == "original":
                    comp_color = color
                else:
                    comp_color = self.make_complementary_color(color, scheme=scheme)
                cv2.rectangle(color_bar, (int(top_x), start_y), (bottom_x, end_y), comp_color, -1)

            top_x = bottom_x

        return color_bar

    def draw_skin_colors_plot(self, num_dominant_colors=5):
        """
        Generates and saves a plot of dominant skin colors from the input image.

        Args:
            num_dominant_colors (int, optional): The number of dominant colors to extract. Default is 5.
        """
        image = cv2.imread(self.image, cv2.COLOR_BGR2RGB)
        face = self.extract_face(image)
        skin = self.extract_skin(face)
        dominant_colors = self.extract_dominant_color(skin,
                                                      number_of_colors=num_dominant_colors,
                                                      has_thresholding=True)
        color_bar = self.plot_color_bar(dominant_colors)
        plt.axis("off")
        plt.imshow(color_bar)
        plt.savefig('temp/skin_colors.png', bbox_inches='tight')
        plt.close()

    def draw_complementary_skin_colors_plot(self, num_dominant_colors=5):
        """
        Generates and saves a plot of complementary skin colors from the input image.

        Args:
            num_dominant_colors (int, optional): The number of dominant colors to extract. Default is 5.
        """
        image = cv2.imread(self.image, cv2.COLOR_BGR2RGB)
        face = self.extract_face(image)
        skin = self.extract_skin(face)
        dominant_colors = self.extract_dominant_color(skin,
                                                      number_of_colors=num_dominant_colors,
                                                      has_thresholding=True)
        complementary_colour_bar = self.plot_complementary_color_bar(dominant_colors)
        plt.axis("off")
        plt.imshow(complementary_colour_bar)
        plt.savefig('temp/complementary_skin_colors.png', bbox_inches='tight')
        plt.close()

    def get_colors(self, num_dominant_colors=5):
        """
          Extract dominant and complementary colors from an image of a face with optional thresholding.

          Args:
              num_dominant_colors (int, optional): The number of dominant colors to extract. Default is 5.

          Returns:
              list: A list of dictionaries, each containing information about a dominant color and its complementary colors.
                  Each dictionary has the following keys:
                  - 'cluster_index': The cluster index of the dominant color.
                  - 'original_color': The original RGB color value as a tuple.
                  - 'color_percentage': The percentage of the original color in the image.
                  - 'complement_color_scheme_invert_green': Complementary color using the "invert_green" scheme as an RGB tuple.
                  - 'complement_color_scheme_invert_blue': Complementary color using the "invert_blue" scheme as an RGB tuple.
                  - 'complement_color_scheme_invert_green_blue': Complementary color using the "invert_green_blue" scheme as an RGB tuple.

          Note:
              This function processes an image of a face to extract dominant colors and their complementary colors.
              It uses various internal methods such as 'extract_face', 'extract_skin', 'extract_dominant_color', and 'make_complementary_color'.
              The 'num_dominant_colors' parameter controls the number of dominant colors to extract from the image.
              Complementary colors are calculated using three different color schemes: "invert_green," "invert_blue," and "invert_green_blue."

          Example:
              # Create an instance of the class
              face_color_extractor = FaceColorExtractor('path_to_image.jpg')

              # Get dominant and complementary colors
              dominant_and_complementary_colors = face_color_extractor.get_colors(num_dominant_colors=5)
          """
        image = cv2.imread(self.image, cv2.COLOR_BGR2RGB)
        face = self.extract_face(image)
        skin = self.extract_skin(face)
        dominant_colors = self.extract_dominant_color(skin,
                                                      number_of_colors=num_dominant_colors,
                                                      has_thresholding=True)

        complementary_colors = []

        def get_complementary_colors(color):

            schemes = ["invert_green", "invert_blue", "invert_green_blue"]
            complementary_colors_info = {
                'cluster_index': color['cluster_index'],
                'original_color': tuple(map(int, color['color'])),
                'color_percentage': color['color_percentage']
            }

            for scheme in schemes:
                complement_color = self.make_complementary_color(color['color'], scheme=scheme)
                complement_color = tuple((int(c) for c in complement_color))
                complementary_colors_info[f'complement_color_scheme_{scheme}'] = complement_color

            return complementary_colors_info

        for color_info in dominant_colors:
            complementary_colors.append(get_complementary_colors(color_info))

        return complementary_colors
