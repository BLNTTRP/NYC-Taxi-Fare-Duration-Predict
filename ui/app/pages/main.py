import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
import requests_client as rq
import datetime
import requests
from settings import ENDPOINT


# Functions to get the current date and time
def get_pickup_date():
    return datetime.datetime.now().date()
def get_pickup_time():
    return datetime.datetime.now().time().replace(microsecond=0)

# Function to autocomplete location names using the API
def autocomplete_location(query):
    
    suggestions = []
    if query:
        res = rq.autocomplete({"text": query})
        if res.status_code == 200:
            suggestions = res.json()
    return suggestions

# Function to get coordinates for a given location using the API
def get_location_coordinates(location):
    res = rq.coordinates({"text": location})
    if res and res.status_code == 200:
        return res.json()
    return None

# Function to plot the route and points on the map
def plot_route_and_points(pickup_coords, dropoff_coords):
    coords = [
        [pickup_coords["longitude"], pickup_coords["latitude"]],
        [dropoff_coords["longitude"], dropoff_coords["latitude"]],
    ]
    route_res = rq.getroute({"coordinates": coords})
    st.write("Route Response:", route_res)
    if route_res and route_res.status_code == 200:
        route_geo = route_res.json()
        route_coords = route_geo.get("coordinates", [])
        route_points = pd.DataFrame(
            [{"lat": lat, "lon": lon} for lon, lat in route_coords]
        )
        route_points = pd.concat(
            [
                pd.DataFrame([
                    {"lat": pickup_coords["latitude"], "lon": pickup_coords["longitude"]},
                    {"lat": dropoff_coords["latitude"], "lon": dropoff_coords["longitude"]},
                ]),
                route_points,
            ],
            ignore_index=True,
        )
        st.map(route_points)
        return coords
    else:
        st.warning("Could not retrieve route from API.")
        return None

# Function to get distance and duration between two coordinates using the API
def get_distance_duration(pickup_coords, dropoff_coords):
    res = rq.get_distance_duration({
        "coordinates": [
            [pickup_coords["longitude"], pickup_coords["latitude"]],
            [dropoff_coords["longitude"], dropoff_coords["latitude"]],
        ]
    })
    if res and res.status_code == 200:
        return res.json()
    else:
        st.error("Could not retrieve distance and duration. Please check your locations.")
        return None

# Function to get the weather at given coords

def get_city_weather():
    """
    Consulta el estado del clima actual para la ciudad de New York.
    """

    try:
        response = requests.get(ENDPOINT)
        response.raise_for_status()
        data = response.json()
        print(data)
        
        data = json.dumps(
        {
            "weather": data.get("weather", [{}])[0].get("description"),
            "icon_url": f"https://openweathermap.org/img/wn/{data.get('weather', [{}])[0].get('icon')}@2x.png"
        })
        return data
    except requests.RequestException as e:
        raise ValueError(status_code=500, detail=f"Error al consultar el clima: {str(e)}")

# Function to predict fare using the API
def predict_fare(input_fields):
    if st.button(f"Confirm the trip info to calculate", key="calculate_fare_button"):
        res = rq.predict(input_fields)
        if res.status_code == 200:
            st.success(
                f"Predicted Fare: ${res.json()['fare']:.2f}. Predicted Duration: {res.json()['duration']} minutes."
            )
        else:
            st.write("Input Fields for Fare Prediction:", input_fields)
            st.error("API error. Please try again.")


# Streamlit Frontend

# User side bar 
st.sidebar.image("images/logo.png", width=100)
if "username" in st.session_state:
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")


# Trip Fare Prediction App 

# Main content header

#st.markdown('<div style="text-align: center;"><h1>NY City Taxi Fare Calculator</h1></div>', unsafe_allow_html=True)
#st.write("Welcome to our app. This app predicts the fare of a taxi ride in New York City based on various parameters such as pickup and drop-off locations, time of day, and passenger count. You can input the details of your ride below to get an estimated fare.")

# Display current trip data in a container with a border
with st.container():

    # First row with Local Time and Weather
    st.markdown('<div style="text-align: center;"><h1>Current Trip Data</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown('<div style="text-align: center;"><h3>Local Time</h3></div>', unsafe_allow_html=True)
            # HTML + JS clock, updates every second without refreshing the app
            st.components.v1.html("""
                <div style="text-align: center; padding: 0px !important; color: FFFFFF">
                  <div><h4 id="clock"></h4></div>                              
                </div>
                <script>
                function updateClock() {
                  var now = new Date();
                  document.getElementById('clock').innerHTML =
                    now.toLocaleTimeString();
                }
                setInterval(updateClock, 1000);
                updateClock();
                </script>
            """, height=50, scrolling=False)
        with col2:
            st.markdown('<div style="text-align: center;"><h3>Date</h3></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align: center; padding: 0px !important"><h5>{get_pickup_date()}</h5></div>', unsafe_allow_html=True)     
        with col3:
            st.markdown('<div style="text-align: center;"><h3>NY Weather</h3></div>', unsafe_allow_html=True)
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.empty()
                with col2:                    
                    weather_info = get_city_weather()
                    data = json.loads(weather_info)
                    if weather_info:
                        st.write(data["weather"])
                        st.image(data["icon_url"])
                with col3:
                    st.empty()
            

    # Second row with Pickup and Dropoff Location Inputs
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            # --- Autocomplete for pickup_location ---
            st.markdown('<div style="text-align: center;"><h3>Pickup Location</h3></div>', unsafe_allow_html=True)
            pickup_query = st.text_input("1. Enter your Pickup Address", key="pickup_location_query")
            pickup_suggestions = autocomplete_location(pickup_query)
            pickup_options = [pickup_query] + [
                item["name"] for item in pickup_suggestions if item["name"] != pickup_query
            ]
            pickup_location = st.selectbox(
                "2. Select the exact Pickup Location",
                pickup_options,
                key="pickup_location_select",
            )

        with col2:
            # --- Autocomplete for dropoff_location ---
            st.markdown('<div style="text-align: center;"><h3>Dropoff Location</h3></div>', unsafe_allow_html=True)
            dropoff_query = st.text_input(
                "3. Enter your Dropoff Address", key="dropoff_location_query"
            )
            dropoff_suggestions = autocomplete_location(dropoff_query)
            dropoff_options = [dropoff_query] + [
                item["name"] for item in dropoff_suggestions if item["name"] != dropoff_query
            ]
            dropoff_location = st.selectbox(
                "4. Select the exact Dropoff Location",
                dropoff_options,
                key="dropoff_location_select",
            )

    pickup_location_coordinates = get_location_coordinates(pickup_location)
    dropoff_location_coordinates = get_location_coordinates(dropoff_location)

    # third row with Passenger Count
    st.markdown('<div style="text-align: center;"><h3>Passenger Count</h3></div>', unsafe_allow_html=True)
    passenger_count = st.slider("Number of Passengers", min_value=1, max_value=4, value=1)

    # Prepare trip data for API call
    trip_data = {
        "pickup_location": pickup_location,
        "dropoff_location": dropoff_location,
        "pickup_date": get_pickup_date(),
        "pickup_time": get_pickup_time(),
        "passenger_count": passenger_count,
        "weather_description": data["weather"],
    }
        
    if pickup_location_coordinates and dropoff_location_coordinates:
        coords = plot_route_and_points(pickup_location_coordinates, dropoff_location_coordinates)
        if coords:
            distance_duration = get_distance_duration(pickup_location_coordinates, dropoff_location_coordinates)
            if distance_duration:
                input_fields = {
                    "start_location": {
                        "latitude": pickup_location_coordinates["latitude"],
                        "longitude": pickup_location_coordinates["longitude"],
                    },
                    "end_location": {
                        "latitude": dropoff_location_coordinates["latitude"],
                        "longitude": dropoff_location_coordinates["longitude"],
                    },
                    "pickup_hour": str(get_pickup_time()),
                    "trip_distance": distance_duration.get("distance", "N/A"),
                    "pickup_date": str(get_pickup_date()),
                    "weather": data["weather"]
                }
                
    else:
        st.info("Enter valid pickup and dropoff locations to plot the route.")

@st.dialog("Please, confirm your trip data")
def inputs_submit(trip_data):

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header(f"Pickup Location:")
        st.subheader(f"{pickup_location}")
        st.write(f"{pickup_location_coordinates}")
        
    with col2:
        st.header(f"Dropoff Location:")
        st.subheader(f"{dropoff_location}")
        st.write(f"{dropoff_location_coordinates}")

    st.header(f"Passenger Count:")
    st.write(f"{passenger_count}")

    predict_fare(input_fields)
        

if "inputs_submit" not in st.session_state:
    st.write("Select your trip data and click the button below to calculate fare and distance.")
    if st.button("Calculate Fare and Distance"):
        inputs_submit(trip_data)
else:
    f"You voted for {st.session_state.inputs_submit['trip_data']}"