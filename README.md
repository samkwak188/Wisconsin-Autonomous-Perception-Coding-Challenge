# Cone Detection Algorithm
Wisconsin Autonomous Perception Coding Challenge Solution

## Overview
This project implements a computer vision algorithm to detect traffic cones and draw boundary lines for a path defined by these cones. The algorithm processes images through multiple stages of analysis to ensure accurate cone detection and reliable path boundary visualization.

![Sample Output](answer.png)

## Methodology

### 1. Color Detection
- Converts image from BGR to HSV color space for more reliable color segmentation
- Implements dual-range HSV thresholding to accurately detect orange-red cones
  - First range: 0-15 in Hue channel
  - Second range: 160-180 in Hue channel
- Applies morphological operations (opening and closing) to clean up noise and fill gaps in detected regions

### 2. Shape Analysis
- Detects contours in the binary mask for shape processing
- Implements comprehensive filtering criteria:
  - Dynamic size thresholds that adapt based on vertical position in image (compensates for perspective)
  - Height to width ratio analysis to identify cone-like shapes
  - Advanced shape properties analysis including:
    - Solidity (area to convex hull area ratio)
    - Circularity (perimeter to area relationship)
    - Extent (contour area to bounding box area ratio)

- Calculates weighted confidence scores using multiple shape metrics:
  - Size contribution (30% weight): Ensures detected objects are appropriate size
  - Shape ratio (20% weight): Verifies cone-like proportions
  - Solidity (20% weight): Checks shape completeness
  - Circularity (15% weight): Validates shape regularity
  - Extent (15% weight): Confirms proper shape filling

### 3. Lane Detection
- Splits detected cones into left and right groups based on image center
- Applies RANSAC regression to find pattern and remove false detections
- Implements polynomial fitting for smooth boundary line visualization

## Implementation Challenges and Solutions

### Successful Approaches
1. **HSV Color Space**: Chosen over RGB for its superior color-intensity separation, providing more reliable cone detection across varying lighting conditions
2. **Dynamic Thresholds**: Implemented adaptive size thresholds based on vertical position, significantly improving detection accuracy for cones at different distances
3. **RANSAC Implementation**: Successfully eliminated false positives and created stable boundary lines through robust pattern analysis

### Unsuccessful Attempts
1. **Initial Color Range Selection**: Early attempts with broader color ranges captured too many non-cone objects. Solved by implementing precise dual-range thresholding
2. **Basic Color Thresholding**: Simple brightness-based thresholding proved unreliable due to varying lighting conditions in the image
3. **Linear Regression**: Initial attempts with DBSCAN clustering and simple linear regression failed to handle the perspective effects properly. Resolved by implementing RANSAC with perspective compensation

## Libraries Used
- OpenCV (cv2): Primary computer vision operations and image processing
- NumPy: Efficient numerical operations and array manipulations
- scikit-learn: RANSAC implementation for robust line fitting

## Future Improvements
- Implementation of deep learning approaches like YOLO could potentially improve detection accuracy
- Additional perspective correction could enhance distance estimation
- Integration of temporal tracking could improve stability in video streams

## References
[Similar Github project](https://gist.github.com/razimgit/d9c91edfd1be6420f58a74e1837bde18)

[OpenCV tutorial](https://www.youtube.com/watch?v=bPSfyK_DJAg&list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn&index=4)

[OpenCV tutorial](https://www.youtube.com/watch?v=ddSo8Nb0mTw&list=PLzMcBGfZo4-lUA8uGjeXhBUUzPYc6vZRn&index=5)

[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12212)

[OpenCV official documentations](https://docs.opencv.org/4.x/)

Numerous Geeks for Geeks and Stack Overflow forums
