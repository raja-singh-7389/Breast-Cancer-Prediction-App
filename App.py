import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


def get_clean_data():
    try:
        data = pd.read_csv("data.csv")
        # Drop unnecessary columns if they exist
        if 'Unnamed: 32' in data.columns and 'id' in data.columns:
            data = data.drop(['Unnamed: 32', 'id'], axis=1)
        # Handle missing values
        data = data.dropna()
        # Map diagnosis to binary values
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        max_value = float(data[key].max())
        mean_value = float(data[key].mean())
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=max_value,
            value=mean_value,
        )
    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        if max_val == min_val:  # Handle cases with no variation
            scaled_value = 0
        else:
            scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    try:
        model = joblib.load("model/model.joblib")
        scaler = joblib.load("model/scaler.joblib")
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure they exist in the 'model/' directory.")
        return
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    probability = model.predict_proba(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {probability[0][0]:.2f}")
    st.write(f"Probability of being malignant: {probability[0][1]:.2f}")

    st.warning("This app can assist medical professionals in making a diagnosis but should not replace a professional's evaluation.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styles if available
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Sidebar inputs
    input_data = add_sidebar()

    # Main content
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts whether a breast mass is benign or malignant using machine learning.")
    
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
