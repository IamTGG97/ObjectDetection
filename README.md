# Real-Time Scene Detection System

A real-time object detection system built with YOLOv8 and OpenCV that processes live webcam footage and generates contextual natural language descriptions of human-object interactions across 80+ detectable object classes.

# Demo

Point your webcam at any scene and the system will detect objects, draw bounding boxes, and display a natural language description of what it sees at the bottom of the screen in real time.

# Features

Real-time object detection using YOLOv8

Natural language scene descriptions generated from detected objects

Spatial relationship analysis that identifies when a person is holding or near an object

Transparent overlay text bar displaying live scene context

Multithreaded pipeline that decouples frame capture from model inference for improved responsiveness

FPS counter displayed live on the video feed

# Tech Stack

Python
OpenCV — webcam access, frame rendering, overlay drawing

YOLOv8 (Ultralytics) — pretrained object detection model

PyTorch — underlying deep learning framework for YOLOv8

Python Threading — multithreaded pipeline for real-time performance

# Prerequisites

Python 3.10, 3.11, or 3.12

A webcam

# Installation
Clone the repo

git clone https://github.com/IamTGG97/ObjectDetection.git

cd ObjectDetection

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install ultralytics opencv-python
Run python detect.py

The YOLOv8 model will download automatically on the first run. Press Q to quit.

# How It Works
Multithreaded Pipeline

The main thread handles reading frames from the webcam and displaying the output as fast as possible. A separate background thread runs YOLOv8 inference continuously on the latest frame. This decoupling prevents the heavy model computation from blocking the video display, resulting in smoother real-time performance.

# Scene Description

After each detection pass, the system analyzes the spatial relationships between detected objects. If an object's bounding box falls significantly inside a person's bounding box, it is classified as being held by that person. Objects outside any person's bounding box are described as nearby. The result is a natural language sentence rendered in a transparent bar at the bottom of the video feed.

# Examples:

A person holding a cell phone

A person holding a cup and a cell phone near a laptop

2 people near a chair

A laptop is visible
