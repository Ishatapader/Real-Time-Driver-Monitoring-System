# Real-Time Driver Monitoring System (DMS)

A system that watches a driver through a webcam, notices when they look sleepy, are yawning, or are nodding off, and warns them with an alarm. Every warning is saved to a file so it can be reviewed later on a simple dashboard.

## Problem Statement

Many road accidents happen because a driver falls asleep or loses focus for just a few seconds without realizing it. There is usually no way for a driver to know in the moment that they look sleepy, and no record afterward to look back on and understand what happened during the drive.

## Solution Strategy

This project uses a webcam and face-tracking to watch the driver's eyes, mouth, and head position in real time:

- **Eyes** — checks if the eyes stay closed for too long
- **Mouth** — checks if the driver is yawning
- **Head position** — checks if the head is dropping forward, like nodding off

If any of these happens for a few seconds in a row, the system plays an alarm sound so the driver wakes up and pays attention. Every time this happens, it's written down with the time and details, so later you can open a dashboard and see how many times it happened and when.

## Project Structure

```
.
├── main.py                  # Watches the webcam, detects warning signs, plays the alarm, saves logs
├── dashboard.py              # Shows the saved logs as charts and numbers
├── config.json                # Settings used by main.py
└── driver_telemetry.csv  # The saved log of all warnings
```

## Requirements

```bash
pip install opencv-python mediapipe numpy streamlit pandas plotly
```

## Usage

**Start monitoring:**
```bash
python main.py
```
Sit normally while it calibrates at the start. Press `m` to mute or unmute the alarm. Press `q` to quit.

**View the dashboard:**
```bash
streamlit run dashboard.py
```
This opens a browser window showing how many warnings happened and a chart of each one over time.
