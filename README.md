# Breast-Cancer-Diagnosis-Prediction-App

Overview

The Breast Cancer Diagnosis App is a machine-learning-powered application designed to support medical professionals in diagnosing breast cancer. By analyzing specific measurements, the app predicts whether a breast mass is benign or malignant. The app also provides a visual representation of the input data using a radar chart and outputs the diagnosis along with the probability of malignancy or benignity.

Users can manually input measurements or connect the app to a cytology lab to receive data directly from laboratory equipment (note: the integration with lab machines is not part of this app).

This project was created as part of a machine learning exercise using the publicly available Breast Cancer Wisconsin (Diagnostic) Data Set. It is important to note that the dataset and predictions are for educational purposes only and should not be used for professional medical diagnosis.

A live version of the app is available on Streamlit Community Cloud.

Installation

To use the app, it is recommended to set up a virtual environment to manage dependencies. For this, Conda is highly recommended.

Steps:
	1.	Create a new Conda environment:

conda create -n breast-cancer-diagnosis python=3.10


	2.	Activate the environment:

conda activate breast-cancer-diagnosis


	3.	Install the required packages:

pip install -r requirements.txt



This will install all necessary dependencies, including Streamlit, OpenCV, and scikit-learn.

Usage

To launch the app:
	1.	Run the following command:

streamlit run app/main.py


	2.	The app will open in your default web browser.
	3.	Use the interface to:
	•	Upload measurement data or input values manually.
	•	Visualize data with radar charts.
	•	Receive predictions for benign or malignant masses.
	4.	Export results to a CSV file for further analysis, if required.
