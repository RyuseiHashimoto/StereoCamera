A program that uses two usb-connected webcams to measure distances.

### Folder
- **「calibration_720」,「calibration_HD720」**
  - **00.getimages.py**・・・Using the camera you wish to calibrate, take pictures of the checkerboard from various angles. The captured images are stored in the “images”folder.
  - **01.calibration.py**・・・The camera distortion is calibrated by detecting checkerboard intersections in the images taken with 00. The calibrated results are saved in pkl file format.
