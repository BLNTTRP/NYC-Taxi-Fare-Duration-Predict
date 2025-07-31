import streamlit as st
from settings import API_BASE_URL
import requests
from typing import Optional
import time

st.image("images/logo.png", width=200)
st.title("Welcome to New York Taxi Fare App")
st.markdown(
    """ 
This app predicts the fare of a taxi ride in New York City based on various parameters such as pickup and drop-off locations, time of day, and passenger count.
You can input the details of your ride below to get an estimated fare.
    """
)

st.title("Create New User")
st.session_state.clear_form = False

for key in ["username", "email", "password", "repeat_password", "clear_form"]:
    if key not in st.session_state:
        st.session_state[key] = ""

def create_user(username: str, email:str, password: str, repeat_password: str) -> Optional[str]:
    """This function calls the create user endpoint of the API

    Args:
        username (str): name of the user
        email (str): email of the user
        password (str): password of the user
        repeat_password (str): confirm password of the user

    Returns:
        Optional[str]: token if login is successful, None otherwise
    """
    return_value = False

    if(password != repeat_password):
        st.error("Sorry, The passwords inserted is not match")
    else:
        url = f"{API_BASE_URL}/user/create_user"

        headers = {
            "accept" : "application/json",
            "Content-Type" : "application/json"
        }

        data = {
            "name": username,
            "email": email,
            "password": password
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            return_value = True
            
    return return_value

if st.session_state.clear_form:
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.password = ""
    st.session_state.repeat_password = ""
    st.session_state.clear_form = False
    st.experimental_rerun()

if "token" not in st.session_state:
    username = st.text_input("User Name", key="username")
    email = st.text_input("Email", key="email")
    password = st.text_input("Password", type="password", key="password")
    repeat_password = st.text_input("Confirm Password", type="password", key="repeat_password")

    if st.button("Create"):
        token = create_user(username, email, password, repeat_password)
        if token:
            time.sleep(1)
            st.session_state.clear_form = True
            st.success("New User Created Successfull!")
        else:
            st.error("Sorry, Have an error on api. Try later")
else:
    st.success("New User Created Successfull!")
