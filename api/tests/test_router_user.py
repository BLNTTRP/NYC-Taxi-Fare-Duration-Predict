from unittest.mock import MagicMock
import unittest
import pytest

from app.auth.jwt import create_access_token
from app.user.models import User
from app.user.schema import User as UserSchema
from app import db
from main import app
from httpx import AsyncClient
from sqlalchemy.orm import Session


class TestRouterUser(unittest.TestCase):
    
    @pytest.mark.asyncio
    async def test_all_users(self):
        mock_session = MagicMock(spec=Session)
        mock_user = User(
            name="John Doe", email="john@yahoo.com", password="123456", kwargs={"id": 1}
        )

        mock_session.query(User).all.return_value = [mock_user]

        app.dependency_overrides[db.get_db] = lambda: mock_session

        async with AsyncClient(app=app, base_url="http://test") as ac:
            user_access_token = create_access_token({"sub": "john@gmail.com"})
            response = await ac.get(
                "/user/", headers={"Authorization": f"Bearer {user_access_token}"}
            )
        self.assertEqual(response.status_code, 200)
        users = response.json()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["name"], "John Doe")


    @pytest.mark.asyncio
    async def test_create_user_registration_success(self):
        mock_session = MagicMock(spec=Session)
        request = UserSchema(
            id=0, name="John Doe", email="john@gmail.com", password="123456"
        )

        mock_session.query(User).filter.return_value.first.return_value = None

        app.dependency_overrides[db.get_db] = lambda: mock_session

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/user/", json=request.dict())

        self.assertEqual(response.status_code, 201)


    @pytest.mark.asyncio
    async def test_create_user_registration_fails(self):
        mock_session = MagicMock(spec=Session)
        mock_user = User(id=0, name="John Doe", email="john@gmail.com", password="123456")
        request = UserSchema(
            id=0, name="John Doe", email="john@gmail.com", password="123456"
        )

        mock_session.query(User).filter.return_value.first.return_value = mock_user

        app.dependency_overrides[db.get_db] = lambda: mock_session

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/user/", json=request.dict())

        self.assertEqual(response.status_code, 400)
