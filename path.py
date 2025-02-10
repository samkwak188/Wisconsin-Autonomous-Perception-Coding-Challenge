import cv2  # OpenCV library for image processing
import numpy as np  # For numerical operations
from sklearn.linear_model import RANSACRegressor  # For analyzing detected data and find pattern
import os  

def cone_detection(image_path):

    # First, load the image. The .any() check makes sure the image loaded correctly
    img = cv2.imread(image_path)
    if not img.any():
        print(f"Couldn't open {image_path}")
        return
    
    # Get image height - needed later for size calculations that depend on vertical position
    # Objects appear smaller when they're further away 
    img_height = img.shape[0]
    
    # Convert image to HSV for better color detection
    # HSV is better than RGB for color detection because it separates color (Hue) from brightness (Value) and color intensity (Saturation)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define orange-red color ranges in HSV
    # Need two ranges because red wraps around the HSV color wheel
    red_low1, red_high1 = np.array([0, 135, 135]), np.array([15, 255, 255])
    red_low2, red_high2 = np.array([159, 135, 135]), np.array([179, 255, 255])
    
    # Create a binary mask where white pixels (255) represent the orange-red colors
    mask = cv2.bitwise_or(
        cv2.inRange(hsv_img, red_low1, red_high1),
        cv2.inRange(hsv_img, red_low2, red_high2)
    )

    # Clean up the mask to remove noise and fill small holes
    # MORPH_OPEN removes small white spots (noise)
    # MORPH_CLOSE fills small black holes
    kernel = np.ones((3,3), np.uint8)
    for i in range(2):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours (outlines) of potential cones in the mask
    cone_shapes = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cones_detected = []

    # For each detected shape, check if it could be a cone
    for shape in cone_shapes:
        # cv2.contourArea is a built-in OpenCV function that calculates the area of the shape by summing up the pixels inside the contour
        # This gives us a measure of how big the detected shape is
        area = cv2.contourArea(shape)  
        
        # cv2.boundingRect returns the minimal upright rectangle that contains the entire shape
        # x, y: coordinates of the top-left corner of the rectangle
        # w: width of the rectangle, h: height of the rectangle
        x, y, w, h = cv2.boundingRect(shape)
        
        # Calculate the relative vertical position of the shape in the image, as objects get further away (higher in the given image), they appear smaller due to perspective
        # y_ratio will be 0 at the top of the image and 1 at the bottom
        y_ratio = y / img_height
        
        # Dynamic size thresholds that adapt based on vertical position
        # Objects further away in the image need smaller thresholds
        # The linear scaling (y_ratio * n) adjusts thresholds smoothly based on position
        min_size = 50 + (y_ratio * 100)    
        max_size = 2000 + (y_ratio * 8000) 
        
        # Filter out shapes that are too small or too large
        if not (min_size < area < max_size):
            continue
        
        # Cones are taller than they are wide.
        # 0.8 factor means height should be at least 1.25 times the width, giving cone shape.
        if h <= w * 0.8:
            continue
        
        # Calculate height-to-width ratio to check if the shape matches cone proportions
        # This ratio helps identify cone-like shapes (taller than wide)
        height_width_ratio = h/w

        # Allow more variation in the ratio for objects higher in the image, due to perspective distortion
        max_ratio = 2.0 + (y_ratio * 0.6)
        
        #Try and error to find the correct ratio; cone height/width ratio is: Ratio = h/2r
        if not (0.6 < height_width_ratio < max_ratio):
            continue
        
        # convexHull finds the smallest convex polygon that contains all points in the shape
        # This function helps analyze how "complete" or "solid" the shape is
        hull = cv2.convexHull(shape)
        hull_area = cv2.contourArea(hull)
        
        # Solidity is the ratio of contour area to its convex hull area
        # A perfect solid shape would have solidity = 1
        # This helps distinguish solid cone shapes from irregular or broken shapes
        solidity = area/hull_area if hull_area else 0
        
        # Circularity measures how circular a shape is
        # Formula: (4π * area )/ perimeter^2
        # A perfect circle would have circularity = 1
        # More irregular shapes have lower values
        perimeter = cv2.arcLength(shape, True)  # True means closed contour
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        
        # Extent is the ratio of actual contour area to bounding rectangle area
        # It measures how much of the bounding rectangle is filled by the actual detected shape
        # Try and error to find the correct ratio
        extent = area/(w*h)
        if not (0.35 < extent < 0.9):
            continue

        # Calculate a weighted confidence score combining multiple shape metrics
        # Each metric contributes differently based on its importance:
        # - Size (30%): Normalized area, capped at 1.0
        # - Shape (20%): How close the height/width ratio is to ideal (1.5)
        # - Solidity (20%): How well the shape fills its convex hull
        # - Circularity (15%): How round/smooth the shape is
        # - Extent (15%): How well the shape fills its bounding box
        score = (
            min(1.0, area / 1000) * 0.3 +        
            (1.0 - abs(1.5 - height_width_ratio) / 1.5) * 0.2 +  
            solidity * 0.2 +                      
            circularity * 0.15 +                  
            extent * 0.15                         
        )
        
        # Only process shapes with high confidence scores
        if score > 0.6:
            # cv2.moments calculates all spatial moments of the shape
            # These can be used to find the centroid (center of mass)
            M = cv2.moments(shape)
            if M["m00"]:  # m00 is the zero-th moment (area), avoid division by zero
                # Calculate centroid coordinates using spatial moments
                # cx = M10/M00, cy = M01/M00 (standard centroid formulas)
                cx = int(M["m10"] / M["m00"])  
                cy = int(M["m01"] / M["m00"])  
                cones_detected.append((cx, cy, score))

    # Need at least 4 cones to form meaningful lanes
    if len(cones_detected) < 4:
        return

    # Split cones into left and right lanes based on their x-position
    # relative to the middle of the image
    points = np.array(cones_detected)
    mid_x = np.mean(points[:, 0])
    left_cones = points[points[:, 0] < mid_x]
    right_cones = points[points[:, 0] >= mid_x]

    # Find a pattern between datas
    def fitLine(points):
        # Return empty list if there aren't enough points to form a line
        if len(points) < 2:
            return []
        
        # Sort points by their y-coordinate (vertical position), since the path in the sample image starts from the bottom and ends at the top.
        points = points[points[:, 1].argsort()]

        # Separate x and y coordinates for RANSAC
        x_coords = points[:, 0].reshape(-1, 1)
        y_coords = points[:, 1]
        
        # Create RANSAC regressor with specific inlier threshold
        # residual_threshold=40 means:
        # - Points within 40 pixels of the fitted line are considered inliers
        # - Points further than 40 pixels are considered outliers
        # This helps remove false cone detections that don't align with the pattern of the path.
        model = RANSACRegressor(residual_threshold=40)

        # Fit the line using RANSAC algorithm
        model.fit(x_coords, y_coords)

        # Return the points that fits the pattern
        return points[model.inlier_mask_]

    # Use the fitLine function to process both lanes
    # left_cones and right_cones contain (x, y, confidence) for each detected cone
    left_valid = fitLine(left_cones) 
    right_valid = fitLine(right_cones) 

    # Create a copy of original image to draw results
    # Use copy() to avoid modifying the original image
    result = img.copy()
    
    # Draw a line through the filtered points
    def drawLine(points, color):
        # Draw circles at each detected cone position
        # x, y are the coordinates
        # z is the confidence score from earlier detection
        for x, y, z in points:
            # cv2.cricle is used to draw the circles. 
            cv2.circle(result, (int(x), int(y)), 5, color, -1)
            
        # Prepare points for polynomial fitting
        # Reshape x coordinates into column vector
        # All x coordinates
        x = points[:, 0].reshape(-1, 1)

        # All y coordinates
        y = points[:, 1]            
        
        # Use numpy's polyfit to fit a line through points
        # Parameters:
        # y: y-coordinates (independent variable)
        # x: x-coordinates (dependent variable)
        # 1: degree of polynomial (1 = linear fit)
        # Returns coefficients [slope, intercept]
        coeffs = np.polyfit(y, x, 1) 
        
        # Extract slope and intercept from coefficients
        # ravel() flattens the coefficient array
        slope = coeffs.ravel()[0]      # First coefficient is slope
        intercept = coeffs.ravel()[1]  # Second coefficient is y-intercept
        
        # Find the endpoints of the line
        # Get minimum and maximum y-coordinates to define line length
        y_min, y_max = np.min(y), np.max(y)
        
        # Calculate corresponding x-coordinates using line equation: x = my + b
        # where m is slope and b is intercept
        x_min = int(slope * y_min + intercept)  # x at minimum y
        x_max = int(slope * y_max + intercept)  # x at maximum y
        
        # Draw the line on the image using cv2.line
        cv2.line(result, (x_min, int(y_min)), (x_max, int(y_max)), (0, 0, 255), 4)
        
        # Return the line endpoints for potential further use
        # Returns tuple of tuples: ((x1,y1), (x2,y2))
        return ((x_min, int(y_min)), (x_max, int(y_max)))

    
    # mark detected cones in blue
    if len(left_valid) >= 2:
        drawLine(left_valid, (255,0, 0))  
    if len(right_valid) >= 2:
        drawLine(right_valid, (255,0,0))  

    # Save the result
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(script_dir, 'answer.png'), result)


cone_detection('/Users/samkwak/Desktop/test/sample.png')