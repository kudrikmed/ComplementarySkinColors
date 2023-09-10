from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import aiofiles
from src.FaceCropper import FaceCropper
from src.SkinColor import SkinColor
import logging


class SkinColorsResponse(BaseModel):
    skin_type: str
    short_info: str


app = FastAPI()

# logging
logger = logging.getLogger(__name__)
logger_file_path = ("logs/app.log")
logging.basicConfig(filename=logger_file_path,
                    level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    filemode="a")


@app.post("/complementary", response_class=FileResponse)
async def predict_macro(file: UploadFile):
    async with aiofiles.open('temp/image.jpg', 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
        logger.info("Image uploaded...")
    fc = FaceCropper(image_path="temp/image.jpg", save_crop="temp/face.jpg")
    fc_result = fc.extract_face_from_image()
    if fc_result == 1:
        logger.info("Face on image is found...")
        sc = SkinColor("temp/face.jpg")
        sc.draw_complentary_skin_colors_plot()
        logger.info("Complementary colors map successfully created...")
        response = "temp/complementary_skin_colors.png"
        return response
    elif fc_result == 0:
        logger.info("No human faces detected...")
        return "No human faces detected"
    elif fc_result == 2:
        logger.info("Several faces detected...")
        return "Several faces detected"
    else:
        logger.info(f"{fc_result}")
        return fc_result
