from typing import Optional
import streamlit as st
import requests
from settings import API_BASE_URL

def login(username: str, password: str) -> Optional[str]:
    """This function calls the login endpoint of the API to authenticate the user
    and get a token.

    Args:
        username (str): email of the user
        password (str): password of the user

    Returns:
        Optional[str]: token if login is successful, None otherwise
    """

    url_login = f"{API_BASE_URL}/login"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "scope": "",
        "client_id": "string",
        "client_secret": "password"
    }
    response = requests.post(url_login, headers=headers, data=data)

    #Checking response status code
    if response.status_code == 200:
        # Extracting token from the response
        token = response.json().get("access_token")
        if token:
            return token
        else:
            st.error("Login failed. No token received.")    #error in streamlit

    return None


st.image("images/logo.png", width=200)
st.title("Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):

    token = login(username, password)
    if token:
        st.session_state["logged_in"] = True
        st.session_state.token = token
        st.session_state["username"] = username
        st.success("Login successful!")
    else:
        st.error("Login failed. Please check your credentials.")