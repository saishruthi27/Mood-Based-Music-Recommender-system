# Mood-Based-Music-Recommender-system

Author : Sai Shruthi Cherukuri

 Documentation: Mood-Based Music Recommender
This project makes song recommendations based on the user's emotional state in real time using facial expression analysis. It combines deep learning models and computer vision techniques to identify emotions from camera footage, and it works with a Streamlit web application to provide an interactive user experience.
Step 1: Install Visual Studio Code
This link will take you to the official website where you may download Visual Studio Code. Link: https://code.visualstudio.com/download
Step 2: Install Python
When you open Visual Studio Code, the extension is visible on the left. You may install it by searching for "search python."
Step 3: Install Dependencies
● OpenCV: A library for computer vision tasks.
● NumPy: A library for numerical computing.
● MediaPipe: A library for real-time solutions in computer vision.
● Keras: A deep learning library for building and training neural networks.
● Streamlit: A library for building interactive web applications.
● Streamlit-WebRTC: A Streamlit extension for integrating real-time video streams.
● Web browser: A module for opening web browsers from Python code.
Step 4: Access Streamlit and Web UI
After you run all the code files, open Power shell ( the extension can be downloaded from Visual Studio code and run as administrator)
- Open the Directory on the C drive where the .py files are saved and run to access the “model.h5”
- Create a virtual environment by using the command “.\venv-tutorial1\Scripts\Activate.ps1”. The virtual environment gets activated.
- Run using the command “streamlit run Group9_music.py”
Usage:
● Your default browser will launch the Streamlit web interface when you run the application.
● Input your desired language and vocalist using the text input fields provided.
 
 ● To start the emotion detection and song recommendation process, click the "Recommend me songs" button.
● Give your webcam permission to record your expressions.
● The suggested songs will appear in the online browser based on your selected language,
singer, and the emotion that was detected.
Note: Make sure your webcam is set up and connected correctly to reliably capture facial expressions. Additionally, when the browser prompts you to grant access to the webcam, make sure you do so.
