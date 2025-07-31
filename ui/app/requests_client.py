import requests
import utils
from settings import API_BASE_URL, ENDPOINT
import streamlit as st
import json

API_URL = 'api' if utils.running_in_docker() else 'localhost'

def autocomplete(data) -> requests.Response:
    """ Make the request to endpoint autocomplete """
    response = None
    # Encabezado para la autenticacion
    try:
        response = requests.post(f"{API_BASE_URL}/route_service/autocomplete", json=data)

    except Exception as e:
        st.text(f"Error on request autocomplete {e}")
        response = requests.Response()
        response.status_code = 400
        return response
        
    return response

def coordinates(data) -> requests.Response:
    """ Make the request to endpoint coordinates """
    response = None
    # Encabezado para la autenticacion
    try:
        response = requests.post(f"{API_BASE_URL}/route_service/coordinates", json=data)

    except Exception as e:
        st.text(f"Error on request coordinates {e}")
        response = requests.Response()
        response.status_code = 400
        return response
        
    return response

def getroute(data) -> requests.Response:    
    """ Make the request to endpoint get_route """
    response = None
    # Encabezado para la autenticacion
    try:
        response = requests.post(f"{API_BASE_URL}/route_service/get_route", json=data)

    except Exception as e:
        st.text(f"Error on request get_route {e}")
        response = requests.Response()
        response.status_code = 400
        return response
        
    return response




def get_distance_duration(data) -> requests.Response:
    """ Make the request to endpoint get_distance """
    response = None
    # Encabezado para la autenticacion
    try:
        response = requests.post(f"{API_BASE_URL}/route_service/get_distance_duration", json=data)

    except Exception as e:
        st.text(f"Error on request get_distance {e}")
        response = requests.Response()
        response.status_code = 400
        return response
        
    return response

def predict(data) -> requests.Response:
    """ Make the request to endpoint predict_fare_duration """
    response = None
    # Encabezado para la autenticacion
    """ headers = {
        "Authorization": f"Bearer {token}",  # Pasando el token en el encabezado
    } """
        
    try:
        response = requests.post(f"{API_BASE_URL}/model/predict_fare_duration", json=data)
        
    except Exception as e:
        print(f"Error on request predict {e}")
        response = requests.Response()
        response.status_code = 400
        return response
        
    return response