# Geometric Shape Detector

Python + OpenCV application that detects geometric shapes in images or live video,
classifies them (triangle, square, rectangle, quadrilateral, pentagon, hexagon, circle, polygon),
draws contours and calculates area, perimeter and bounding boxes.

## Pipeline
1. Preprocessing: grayscale, Gaussian blur, Otsu threshold, Canny edges
2. Contour detection with small-noise filtering
3. Polygon approximation (`approxPolyDP`) and classification
4. Measurements: area, perimeter, bounding box, centroid

## Setup
```
pip install -r requirements.txt
```

## Usage
```
python shape_detector.py --demo
python shape_detector.py --image shapes.png --save result.png
python shape_detector.py --video 0
```
Press `q` to quit and `d` to toggle debug windows in video mode.

## Demo output
![demo](demo_output.png)