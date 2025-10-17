# AI Agent Tasks

This project includes a set of automation tasks written in JSON format. These task are used for the AI agent research. Each task performs a specific action related to collecting, organizing, and saving visual data from online sources.

---

## 1. cli1_task_details

This task turns your microphone input into a visual display inside the terminal.  
It uses the built in tool **arecord** and **sox** to live record, process, and generate spectrogram image.

---

## 2. cli2_task_details

This task transforms your terminal into a live visualization of CPU activity.  
It uses installable tools such as **sar**, **mpstat**, and **htop** to enable system statistics, capture CPU load, and display them as numeric columns in the terminal.

---

## 3. cli3

This CLI task is an image processing and manipulation pipeline.  
It pulls any image (in this example a black square from wiki) and resizes, grayscales and blurs the image. Then it blends the desaturated and blurred image together for a final product.  
It uses the **jp2a** tool to display the final product in your terminal.

---

## 4. traffic_screenshot_sd

This task opens Google Chrome and navigates to the **California QuickMap** website.  
It enables road condition overlays (Fast, Lane Closures, Full Closures, CHP Incidents), zooms into **San Diego**, switches to full-screen mode, and uses `Alt+Print` to capture and save a live traffic snapshot in the user’s Documents directory.

---

## 5. weather_screenshot_sd

This task opens **Google Chrome** and navigates to the **National Weather Service San Diego** page to view live weather and radar data. After ensuring the page is fully visible, it captures a window `Alt+Print` of the weather map and saves it automatically to the user’s Documents folder for later reference.

---

## 6. copy_to

This task opens a terminal and runs a command that creates a folder at  
`~/Documents/traffic_and_weather_data/` (if missing) and copies the **two most recent screenshots** into it.  
It ensures both the traffic and weather images are stored together in one organized directory for analysis or documentation.

---

## 7. Task_7

Task details goes here..

---

## 8. Task_8

Task details goes here..

---

## 9. Task_9

Task details goes here..

---

## 10. Task_10

Task details goes here..

---

### Notes
- All tasks are designed for **Linux systems** using GNOME desktop environment.  
- Screenshots are taken using `Alt+Print` screenshot commands and file management is handled via standard shell utilities.  
- These automations are intended for **educational and research use** only.

---
