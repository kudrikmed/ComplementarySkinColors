from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import aiofiles
from src.FaceCropper import FaceCropper
from src.SkinColor import SkinColor
import logging
import tempfile
import json

app = FastAPI()


logger = logging.getLogger(__name__)
logger_file_path = "logs/app.log"
logging.basicConfig(
    filename=logger_file_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="a",
)


def error_response(message, status_code=400):
    """
    Creates a file response with an error message.

    Args:
        message (str): The error message.
        status_code (int, optional): The HTTP status code. Default is 400.

    Returns:
        FileResponse: A FileResponse with the error message.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(message)
    return FileResponse(temp_file.name, status_code=status_code, media_type="text/plain")


@app.post("/complementary-info")
async def complementary_colors_info(file: UploadFile):
    """
    Process an uploaded image to extract complementary colors information from detected human faces.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict or HTTPException: A dictionary containing complementary colors information for detected faces
        or an HTTPException with appropriate status code and detail message in case of errors.

    Raises:
        HTTPException: If the uploaded file is not a JPG image, is too large, or if there are errors during processing.

    Example:
        # Make a POST request to the '/complementary-info' endpoint with an uploaded JPG image file.
        # The response will contain complementary colors information or an error message.
    """
    try:
        # Check if the uploaded file is a JPG image
        if not file.filename.lower().endswith(('.jpg', '.jpeg')):
            return HTTPException(status_code=400, detail="Only JPG files are allowed.")
        # Check if the uploaded file is less than 10 MB
        file_size = file.file.tell()
        if file_size > 10 * 1024 * 1024:
            return HTTPException(status_code=400, detail="File is too large. Upload a file less than 10 MB.")
        # Save the uploaded file
        async with aiofiles.open('temp/image.jpg', 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.info("Image uploaded...")

        fc = FaceCropper(image_path="temp/image.jpg", save_crop="temp/face.jpg")
        fc_result = fc.extract_face_from_image()

        if fc_result == 1:
            logger.info("Face on image is found...")
            sc = SkinColor("temp/face.jpg")
            info = sc.get_colors()
            logger.info("Complementary colors info successfully created...")
            json_response = json.dumps(info, indent=2)
            return json_response

        elif fc_result == 0:
            logger.info("No human faces detected...")
            return HTTPException(status_code=400, detail="No human faces detected on uploaded image")

        elif fc_result == 2:
            logger.info("Several faces detected...")
            return HTTPException(status_code=400, detail="Several human faces detected on uploaded image")

        else:
            logger.info(f"{fc_result}")
            return HTTPException(status_code=400, detail="Error while face detection")
        return "ok"
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return HTTPException(status_code=500, detail="Internal server error")


@app.post("/complementary-bar", response_class=FileResponse)
async def complementary_colors_bar(file: UploadFile):
    """
    Handles the /complementary-bar endpoint for processing uploaded images and generating complementary colors.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        FileResponse: A FileResponse containing the generated complementary colors map or an error response.
    """
    try:
        # Check if the uploaded file is a JPG image
        if not file.filename.lower().endswith(('.jpg', '.jpeg')):
            return error_response("Only JPG files are allowed.", status_code=400)
        # Check if the uploaded file is less than 10 MB
        file_size = file.file.tell()
        if file_size > 10 * 1024 * 1024:
            return error_response("File is too large. Upload file less than 10 MB.", status_code=400)
        # Save the uploaded file
        async with aiofiles.open('temp/image.jpg', 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.info("Image uploaded...")

        fc = FaceCropper(image_path="temp/image.jpg", save_crop="temp/face.jpg")
        fc_result = fc.extract_face_from_image()

        if fc_result == 1:
            logger.info("Face on image is found...")
            sc = SkinColor("temp/face.jpg")
            sc.draw_complementary_skin_colors_plot()
            print(sc.get_colors())
            logger.info("Complementary colors map successfully created...")
            return FileResponse("temp/complementary_skin_colors.png")

        elif fc_result == 0:
            logger.info("No human faces detected...")
            return error_response("No human faces detected on uploaded image...")

        elif fc_result == 2:
            logger.info("Several faces detected...")
            return error_response("Several human faces detected on uploaded image...")

        else:
            logger.info(f"{fc_result}")
            return error_response("Unknown error while face detection...")
        return "ko"
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return HTTPException(status_code=500, detail="Internal server error")
