from unittest.mock import AsyncMock, MagicMock, patch

import unittest
import pytest
from app.auth.jwt import get_current_user
from app.model.schema import PredictResponse
from fastapi import UploadFile
from fastapi.testclient import TestClient
from httpx import AsyncClient
from main import app

class TestRouterModel(unittest.TestCase):

    @pytest.mark.asyncio
    async def test_predict_fare_duration(self):
        data = {}
        mock_user = MagicMock()
        mock_user.id = 1

        mock_current_user = MagicMock()
        mock_current_user.return_value = "testtoken"

        app.dependency_overrides[get_current_user] = lambda: mock_current_user

        with patch("app.model.router.utils.get_file_hash", return_value="fakehash123"):
            
            with patch(
                "app.model.router.model_predict", new_callable=AsyncMock
            ) as mock_model_predict:
                mock_model_predict.return_value = (5.00, "00:10:00")
                    
                with patch("builtins.open", new_callable=MagicMock):
                    
                    async with AsyncClient(app=app, base_url="http://test") as ac:
                        response = await ac.post(
                            "/model/predict_fare_duration",
                            json=data,
                            headers={"Authorization": "Bearer testtoken"},
                        )

                        self.assertEqual(response.status_code, 200)

                        response_data = response.json()
                        self.assertEqual(response_data["success"], True)
                        self.assertEqual(response_data["fare"], 5.00)
                        self.assertEqual(response_data["duration"], "00:10:00")


