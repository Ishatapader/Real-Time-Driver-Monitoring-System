 # Real-Time Driver Monitoring System (DMS)

A webcam-based system that watches a driver's eyes, mouth, and head position in real time, detects signs of drowsiness, plays an alert, and logs every event for later review.



---

## Problem Statement
Driver fatigue is one of the leading causes of road accidents, and it's hard to catch in the moment, drivers often don't realize their eyes have closed for a second too long or their head has started to drop until it's already a problem. Most vehicles have no built-in way to detect this in real time, and after a drive there's usually no data to look back on to understand when or how often it happened. This project aims to close that gap with a low-cost, webcam-only monitoring solution.

## Approach
- Uses MediaPipe's face mesh model to extract facial landmarks from each webcam frame
- Turns landmarks into three measurable signals: eye closure (Eye Aspect Ratio), yawning (Mouth Aspect Ratio), and head pitch
- Head pitch is estimated by solving a Perspective-n-Point problem with OpenCV's `solvePnP` against a 3D face model, then compared against a baseline captured during a short calibration step
- To avoid false alerts from a normal blink or a quick glance away, each signal has to stay past its threshold for several consecutive frames before it counts as a real warning
- Once triggered, the system plays an audio alert on a separate thread so the video feed never freezes
- Logs every event with a timestamp to a CSV file; a Streamlit dashboard then reads that file to summarize and chart what happened across a session

## Key Features
- Real-time face and posture tracking from a webcam
- Auto-calibration to the driver's normal sitting position
- False-alarm filtering using consecutive-frame checks
- Non-blocking audio alerts (runs on a separate thread)
- CSV logging and a Streamlit dashboard for reviewing past sessions

## Tech Stack

**Python | OpenCV | MediaPipe | NumPy | Streamlit | Pandas | Plotly**

## Project Structure
```
.
├── main.py                  # Webcam capture, landmark detection, alert logic, CSV logging
├── dashboard.py             # Streamlit dashboard for reviewing logged sessions
├── config.json              # Externalized thresholds and runtime settings
└── driver_telemetry.csv     # Auto-generated log of detected events
```

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ishatapader/Real-Time-Driver-Monitoring-System.git
cd Real-Time-Driver-Monitoring-System
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe numpy streamlit pandas plotly
```

## Usage

**Run the monitoring system:**
```bash
python main.py
```
Sit normally during the brief calibration phase at startup. Press `m` to mute/unmute the alert, `q` to quit.

**Review session data:**
```bash
streamlit run dashboard.py
```
Opens a browser dashboard showing total warnings and time-series charts for each metric.

---

## Author

**Isha Tapader**

🔗 GitHub: [github.com/Ishatapader](https://github.com/Ishatapader)  
💼 LinkedIn: [linkedin.com/in/isha-tapader-116680247](https://www.linkedin.com/in/isha-tapader-116680247/)
