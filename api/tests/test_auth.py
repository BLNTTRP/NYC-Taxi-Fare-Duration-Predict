import unittest

from app.auth.jwt import create_access_token

class TestAuth(unittest.TestCase):
    
    def test_create_access_token(self):
        user_access_token = create_access_token({"sub": "john@gmail.com"})
        
        self.assertIsInstance(user_access_token, str)