**A program that uses two usb-connected webcams to measure distances.**
- Camera used
  - Logitech Webcam C270 720P https://www.logicool.co.jp/ja-jp/products/webcams/hd-webcam-c270n.html
  - Logitech Webcam C270 HD 720P https://amzn.asia/d/bOuo2Yy



### Folder
- **「calibration_720」,「calibration_HD720」**・・・
  - **00.getimages.py**・・・Using the camera you wish to calibrate, take pictures of the checkerboard from various angles. The captured images are stored in the “images”folder.
  - **01.calibration.py**・・・The camera distortion is calibrated by detecting checkerboard intersections in the images taken with 00. The calibrated results are saved in pkl file format.


 ### File
- **main_capture.py**・・・A program that takes an image with each camera and measures the distance from that image. Calibrate each camera using the pkl files generated by “calibration_720” and “calibration_HD720”. 
It detects a red object and its center of gravity, and calculates the distance using the following formula based on the idea of triangulation.  <br>`Distanse = (Distance between cameras * Camera focal length) / Parallax`

- **.jpg**・・・Image taken by “main_capture.py"

- **main_raltime.py**・・・A program that measures distances in real time. The distance measurement method is the same as in “main_capture.py”.
