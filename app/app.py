from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import aiofiles
from src.FaceCropper import FaceCropper
from src.SkinColor import SkinColor
import logging
import tempfile

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


@app.post("/complementary", response_class=FileResponse)
async def draw_complementary_colors(file: UploadFile):
    """
    Handles the /complementary endpoint for processing uploaded images and generating complementary colors.

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

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return HTTPException(status_code=500, detail="Internal server error")
