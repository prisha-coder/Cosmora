#!/usr/bin/env python
# coding: utf-8

import streamlit as st

# -------------------------------
# 0. Page config - MUST be first
# -------------------------------
st.set_page_config(page_title="Cosmora Classifier", layout="wide")

# -------------------------------
# 1. Imports
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
import warnings
import requests
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
import httpx

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# 2. Load and prepare data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("koi_data.csv")
    needed_columns = [
        "koi_period", "koi_prad", "koi_srad", "koi_steff", "koi_slogg",
        "koi_time0bk", "koi_impact", "koi_duration", "koi_teq", "koi_disposition"
    ]
    df = df[needed_columns].dropna()
    le = LabelEncoder()
    df["koi_disposition"] = le.fit_transform(df["koi_disposition"])
    X = df.drop("koi_disposition", axis=1)
    y = df["koi_disposition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
    return X_train, X_test, y_train, y_test, le

X_train, X_test, y_train, y_test, le = load_data()

# -------------------------------
# 3. Train models
# -------------------------------
gb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
hgb_clf = HistGradientBoostingClassifier(max_iter=200, max_depth=10, random_state=42)

clf = VotingClassifier(
    estimators=[('gb', gb_clf), ('hgb', hgb_clf)],
    voting='soft'
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = round(accuracy_score(y_test, predictions) * 100)

# -------------------------------
# 4. Azure OpenAI client
# -------------------------------
client = AzureOpenAI(
    api_version="2024-02-01",
    api_key="6sZsHJT5v9yxCzgx5W0Gx9GXKvQnbWWo0eN7D9XXRZiAwCijvsl8JQQJ99BJACfhMk5XJ3w3AAAAACOGOC1b",  # Replace with your key
    azure_endpoint="https://prish-mgdunwfl-swedencentral.cognitiveservices.azure.com/"
)

def generate_exoplanet_image(prompt, size="1024x1024"):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt + ", space illustration, realistic, cinematic",
            size=size,
            n=1
        )
        image_url = response.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# -------------------------------
# 5. Helper functions
# -------------------------------
labels = {0: "Exoplanet Candidate", 1: "Exoplanet", 2: "Not an Exoplanet"}

def describe_exoplanet(koi_prad, koi_teq, koi_period):
    description = []
    if koi_prad <= 1:
        description.append("a rocky planet smaller than Earth")
    elif koi_prad <= 2:
        description.append("a rocky super-Earth")
    elif koi_prad <= 6:
        description.append("a gas dwarf or mini-Neptune")
    else:
        description.append("a large gas giant, like Jupiter")

    if koi_teq <= 250:
        description.append("likely an icy pale blue and frozen")
    elif koi_teq <= 500:
        description.append("cold, blue and dim")
    elif koi_teq <= 1000:
        description.append("moderately warm, possibly with liquid water")
    elif koi_teq <= 2000:
        description.append("very hot, possibly molten orange")
    else:
        description.append("extremely hot, glowing red and hostile")

    if koi_period <= 10:
        description.append("and orbits its star extremely closely")
    elif koi_period <= 100:
        description.append("with a relatively short orbit")
    elif koi_period <= 500:
        description.append("with an Earth-like year length")
    else:
        description.append("and takes centuries to orbit its star")

    return "is " + ", ".join(description) + "."

def plot_transit_light_curve(name, koi_period, koi_prad, koi_srad, koi_duration):
    transit_duration_days = koi_duration / 24
    time = np.linspace(-0.1, 0.1, 1000) * koi_period
    depth = (koi_prad / (koi_srad * 109))**2
    flux = np.ones_like(time)
    in_transit = np.abs(time) < (transit_duration_days / 2)
    flux[in_transit] -= depth

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, flux, color='darkblue')
    ax.set_xlabel("Time from transit center (days)")
    ax.set_ylabel("Normalized Brightness")
    ax.set_title(f"Transit Light Curve of {name}")
    ax.grid(True)
    st.pyplot(fig)

# -------------------------------
# 6. Streamlit App Interface
# -------------------------------

st.title("Welcome to Cosmora Classifier! 🌌")
st.write("This interactive classifier analyzes planetary and stellar parameters to predict whether a candidate is a **confirmed exoplanet**, a **candidate**, or a **false positive**.")

st.sidebar.header("Enter Exoplanet Parameters")

# Sidebar inputs
koi_period = st.sidebar.number_input("Orbital period (days)", 0.0, 10000.0, 0.0)
koi_prad = st.sidebar.number_input("Planet radius (Earth radii)", 0.0, 30.0, 0.0)
koi_impact = st.sidebar.number_input("Impact parameter (0=center, 1=edge)", 0.0, 1.0, 0.0)
koi_time0bk = st.sidebar.number_input("Time of first transit (days)", 0.0, 3000.0, 0.0)
koi_duration = st.sidebar.number_input("Transit duration (hours)", 0.0, 50.0, 0.0)
koi_teq = st.sidebar.number_input("Equilibrium temperature (K)", 0.0, 3000.0, 0.0)
koi_steff = st.sidebar.number_input("Star effective temperature (K)", 0.0, 10000.0, 0.0)
koi_srad = st.sidebar.number_input("Star radius (Solar radii)", 0., 10.0, 0.0)
koi_slogg = st.sidebar.number_input("Star surface gravity", 0.0, 10.0, 0.0)
planet_name = st.sidebar.text_input("Give your planet a name", "Kepler-51b")

# Prediction button
if st.sidebar.button("Classify Exoplanet"):
    features = np.array([[koi_period, koi_prad, koi_srad, koi_steff, koi_slogg,
                          koi_time0bk, koi_impact, koi_duration, koi_teq]])
    prediction = clf.predict(features)[0]
    label_name = labels[prediction]

    st.subheader("Prediction~")
    st.write(label_name)

    if label_name in ["Exoplanet Candidate", "Exoplanet"]:
        description = describe_exoplanet(koi_prad, koi_teq, koi_period)
        st.markdown("### Planet Description~")
        st.write(f"{planet_name} {description}")

        st.markdown("### Transit Light Curve~")
        st.write("A transit light curve shows a star dimming as a planet passes. Deeper dips mean bigger planets and wider dips mean longer crossings.")
        plot_transit_light_curve(planet_name, koi_period, koi_prad, koi_srad, koi_duration)

        # AI-Generated Image
        st.markdown("### AI-Generated Exoplanet Image~")
        img = generate_exoplanet_image(f"{planet_name}, {description}")
        if img:
            st.image(img, caption="AI-Generated Exoplanet", use_column_width=True)
    else:
        st.warning("Sorry! No description available for false positives 🥺")

st.sidebar.markdown("---")
st.sidebar.info("[Learn More About The Transit Method](https://academic.oup.com/mnras/article/513/4/5505/6472249?login=false)")
