## PaddleOCR + YOLO Screen Parser (CPU)

This code grabs your screen from primary monitor.  
Runs PaddleOCR for text and YOLO for regions/icons.  
Then returns an array of elements for LLM planner.  

### Required Packages
Here is a list of required packages: `paddlepaddle paddleocr ultralytics mss opencv-python`

### How to run
You can initalize `ScreenParserCPU()` and call `.parse_fullscreen()` to return a result dictionary of window size, parse time and all elements captured from the primary window.  
I provided standalone code to run a small pipeline and print out results. Run `python .\vision_CPU.py` to run mini pipeline. 

### Speed Improvements
Drop: `max_side_for_ocr=1600` to a smaller number (1280)  
Set: `use_angle_cls=False` in PaddleOCR init code  
Increase: `rec_batch_num` in PaddleOCR init code  
These will make the code run faster for a trade off in accuracy, also how "busy" or "noisy" your screen is plays a major factor in processing speed (more elements to crop/parse)