#Accidently killed the live link here's the new one: https://07baecf4225f11a42e.gradio.live
# Sports Deepfake Detector

Sports Deepfake Detector is a browser based image authenticity checker built for Pepperdine University hackathon project. It is designed to help evaluate whether a sports related image is likely authentic, possibly synthetic, or inconclusive, withing the given abilities. 

## Features

- Pretrained deepfake image classifier
- Face detection for face centered analysis
- Blur and resolution checks
- Metadata inspection for suspicious editing or missing EXIF clues
- Inconclusive result when evidence is weak

## How It Works

The app analyzes uploaded sports images using a combination of:
- a pretrained image-classification model
- face detection
- image quality checks
- metadata-based forensic clues

It then returns:
- a prediction
- model confidence
- image quality notes
- metadata warnings
- an explanation of the result

## Tech Stack

- Python
- Gradio
- Transformers
- OpenCV
- Pillow
- NumPy

## Run Locally

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
python3 app.py
