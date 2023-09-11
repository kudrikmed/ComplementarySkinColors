import pytest
from httpx import AsyncClient
from app.app import app
import os


@pytest.mark.asyncio
async def test_valid_jpg_upload():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        image_path = os.path.join(os.path.dirname(__file__), 'test.jpg')
        with open(image_path, 'rb') as image_file:
            image_content = image_file.read()

        response = await ac.post("/complementary-info", files={"file": ("test.jpg", image_content)})
        assert response.status_code == 200
        assert "cluster_index" in response.json()
        assert "original_color" in response.json()
        assert "color_percentage" in response.json()
        assert "complement_color_scheme_invert_green" in response.json()
        assert "complement_color_scheme_invert_blue" in response.json()
        assert "complement_color_scheme_invert_green_blue" in response.json()


@pytest.mark.asyncio
async def test_invalid_file_type():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        image_path = os.path.join(os.path.dirname(__file__), 'test.png')
        with open(image_path, 'rb') as image_file:
            image_content = image_file.read()
        response = await ac.post("/complementary-bar", files={"file": ("test.png", image_content)})
        assert response.status_code == 400
        assert "Only JPG files are allowed." in response.text
