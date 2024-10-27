# Real-Time Laser Point Detector

This script uses OpenCV and scikit-image to capture real-time video from a camera, detect bright spots (such as laser points) in the image, and mark them accordingly.

## Features

- Real-Time Video Processing: Captures frames from the camera and processes them in real-time.
- Bright Spot Detection: Highlights bright spots in the image through grayscale conversion, Gaussian blur, and thresholding.
- Noise Removal: Utilizes morphological operations (erosion and dilation) to eliminate noise from the image.
- Connected Component Analysis: Identifies and filters bright spot regions that meet certain criteria.
- Contour Drawing： Draws the location and information of detected bright spots on the original image.
- Adjustable Parameters: Provides a GUI interface to adjust detection parameters in real-time for optimal results.

#How to Adjust the Sliders to Optimize Detection

  (1)Enable Manual Mode:
    **Set the manual slider to 1.**
    This_ activates the pixels_num slider for manual adjustment._
  (2)Pixel Count Threshold (_**pixels_num):Represents the maximum number of pixels a bright region**_ can have to be considered a valid detection. 
     And The script retains regions smaller than this threshold.Since a laser point is typically small, we want to focus on smaller regions and exclude larger, irrelevant bright areas.

## Requirements

Ensure you have the following Python libraries installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- scikit-image (`skimage`)
- imutils

### Installation

Use `pip` to install the required libraries:


pip install opencv-python numpy scikit-image imutils


## Usage

1.Modify Camera Index (Optional):

   Depending on your camera device, modify the `CAM` variable in the script:
   CAM = 0  # Default camera index is 0
  
2. Run the Script:

   python laser_point_detector.py
   
3. Adjust Parameters:

   - The program will open a window named `params`, containing two adjustable parameters:
     - `manual`: Manual mode switch (0: Off, 1: On).
     - `pixels_num`: Pixel count threshold (effective when `manual` is on).
   - Adjust the sliders as needed to optimize detection.

5. View Detection Results:

   - `image` Window: Displays the original image with detection markings.
   - `mask` Window: Displays the processed binary image.

6. Exit the Program:

   Press the `ESC` key on your keyboard to exit.

## Code Explanation

- Initialize the Camera:
  
    cap = cv2.VideoCapture(CAM)
  
- Create Parameter Adjustment Window:
 
  cv2.namedWindow("params")
  cv2.createTrackbar("manual", "params", 0, 1, nothing)
  cv2.createTrackbar("pixels_num", "params", 0, 2000, nothing)

- Image Preprocessing:

  - Convert to Grayscale:

      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  - Apply Gaussian Blur:

      blurred = cv2.GaussianBlur(gray, (11, 11), 0)
   
- Thresholding:

    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
  
- Morphological Operations:
 
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
  
- Connected Component Analysis:

    labels = measure.label(thresh, connectivity=2, background=0)

- Region Filtering and Drawing:

  - Get Slider Parameters:
   
      MANUAL = cv2.getTrackbarPos("manual", "params")
      PIXEL_THRESH = cv2.getTrackbarPos("pixels_num", "params") if MANUAL == 1 else 500

  - Iterate Over Connected Regions and Filter:

      for label in np.unique(labels):
          # Filter out small regions
    
  - Draw the Largest Contour:

    cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 255, 0), 3)
    cv2.putText(image, "#{} at ({},{})".format(i + 1, int(cX), int(cY)), ...)

## Notes

- Camera Connection: Ensure that the camera is properly connected and the appropriate drivers are installed.
- Ambient Lighting: Strong ambient light may affect detection results. It's recommended to use in a darker environment.
- Parameter Adjustment: Adjust threshold values and pixel counts according to the actual situation to achieve the best detection results.

## Troubleshooting

- Unable to Read Camera:

  If you encounter the following error:

    Unable to read the camera. Please check the device connection.
    Please check if the camera is properly connected or try changing the `CAM` index.

- No Contours Detected:

  If no bright spots are detected due to strong ambient light or improper parameter settings, try adjusting the parameters or improving the ambient lighting conditions.

## License

  MIT License

## Contact

  If you have any questions or suggestions, please contact (2111545820@qq.com).
