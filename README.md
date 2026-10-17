# Qhali Ñawi: Virtual Reality Perimetry System to Support Glaucoma Screening

Qhali Ñawi is a prototype designed to support glaucoma screening through a visual field examination experience. It combines a mobile application, camera-based eye tracking, and a virtual reality (VR) environment.

The system integrates a Flask-based eye-tracking service, an ESP32-CAM configured as a camera server, and a React Native mobile application. The Unity VR project is part of the complete system but is not included in this repository due to its size.

## System Overview

Qhali Ñawi consists of three main software components:

1. **Eye-Tracking Service (`App/`):** A Python application built with Flask that processes images obtained from the ESP32-CAM to perform eye tracking.
2. **Camera Server (`CameraWebServer/`):** Firmware that configures the ESP32-CAM to serve camera images over a network.
3. **Mobile Application (`QhaliNawi/`):** A React Native application that provides the interface for conducting the glaucoma screening examination.

The complete prototype also incorporates a Unity-based virtual reality environment, which is not included in this repository.

## Repository Structure

```text
Qhali-Nawi/
├── App/
│   └── Flask server and eye-tracking implementation
├── CameraWebServer/
│   └── ESP32-CAM camera server firmware
└── QhaliNawi/
    └── React Native mobile application
```

## How It Works

At a high level, the system follows this workflow:

1. **Camera streaming:** The ESP32-CAM runs a camera server and makes images available over the network.
2. **Eye tracking:** The Flask service retrieves images from the camera server and processes them to track eye movements.
3. **Examination interface:** The React Native application guides the user through the screening examination.
4. **Virtual reality:** The Unity component provides the VR environment used during the examination.
5. **Screening support:** The components work together to support a perimetry-style visual field examination intended for glaucoma screening.

The exact communication protocols, calibration process, examination procedure, and data flow depend on the implementation in the source code.

## Technologies

### App — Eye-Tracking Service

The `App/` directory contains the Python application responsible for image processing and eye tracking.

- Python
- Flask
- Video-based eye tracking
- Network communication with the ESP32-CAM

### CameraWebServer — ESP32-CAM

The `CameraWebServer/` directory contains the firmware used to configure the ESP32-CAM as a camera server.

- ESP32-CAM
- Camera web server
- Network connectivity

Network credentials and device-specific settings should be configured locally.

### QhaliNawi — Mobile Application

The `QhaliNawi/` directory contains the React Native application used to interact with the screening workflow.

- React Native
- Mobile user interface
- Examination workflow

Refer to the project files for the exact dependencies and platform-specific setup instructions.
