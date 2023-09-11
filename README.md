# Project Title: Complementary Skin Colors

## Description

This project is a Complementary Skin Colors Map Generator that can be used in makeup and related areas. It evaluates the colors present in a human face and generates a complementary color map. Makeup artists and enthusiasts can utilize this tool to discover complementary colors for makeup applications that harmonize with a person's skin tone.

## Features

- **Face Detection**: The project uses face detection to identify the presence of human faces in an uploaded image.
- **Color Evaluation**: After detecting a face, the system evaluates the skin colors within the face region.
- **Complementary Color Map**: Based on the skin colors detected, the project generates a complementary color map.
- **Error Handling**: Robust error handling is implemented to provide informative error messages to users.

Technologies Used
FastAPI: The project is built using FastAPI, a modern web framework for building APIs with Python.
OpenCV: OpenCV is used for image processing.
Mediapipe: Mediapipe is used for facial landmark detection, allowing for more precise face extraction.
MTCNN: MTCNN (Multi-task Cascaded Convolutional Networks) is utilized for robust face detection, particularly useful in handling images with multiple faces.
Logging: Logging is implemented to keep track of events and errors.
JSON: JSON is used to format and return complementary color information.
Matplotlib: Matplotlib is used to create the complementary color map.

## Endpoints

### 1. `/complementary-info` (POST)

This endpoint processes an uploaded image to extract complementary color information from detected human faces.

#### Request

- Accepts an image file in JPG format.

#### Response

- Returns a JSON response containing complementary color information for detected faces.
- In case of errors, it returns an HTTPException with an appropriate status code and detail message.

### 2. `/complementary-bar` (POST)

This endpoint handles the processing of uploaded images and generates complementary color maps.

#### Request

- Accepts an image file in JPG format.

#### Response

- Returns a PNG image file containing the generated complementary color map.
- In case of errors, it returns an error response.

## Usage

1. Upload a JPG image containing a human face to either endpoint.
2. The system will detect the face and evaluate the skin colors.
3. If a face is detected, it will generate a complementary color map based on the skin colors.
4. You can view the complementary color map or the JSON color information in the response.
5. More detailed info you can see in documentation (docs/_build/html/index.html).

## Setup

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the FastAPI application using `uvicorn app:app --host 0.0.0.0 --port 8000` or by running `run.py` file.
4. Access the API at `http://localhost:8000`.

## Author

[Dzmitry Kudrytski]

## Acknowledgments

- [FastAPI, Mediapipe, OpenCV, Scikit-learn, Pytest.]

## Support and Contribution

- Contributions to this project are welcome. Please submit bug reports, feature requests, or pull requests on [GitHub](https://github.com/kudrikmed/ComplementarySkinColors).
- For support or inquiries, contact [kudrikmed@gmail.com].