import cv2
import numpy as np

# Load the saved image
image_path = "tennisballtestphoto.jpg"  # Replace with your image file path
frame = cv2.imread(image_path)

# Resize the image if needed (optional, for better visualization)
frame = cv2.resize(frame, (800, 600))  # Adjust the dimensions as necessary

# Convert the image to HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the HSV range for the tennis ball color
lower_bound = np.array([25, 50, 50])  # Adjust based on your tennis ball's HSV color
upper_bound = np.array([40, 255, 255])

# Create a mask for the tennis ball
mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

# Optional: Clean up the mask using morphological operations
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Find contours of the tennis ball
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours to identify and mark the tennis ball
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Filter out small objects
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a rectangle around the detected tennis ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw a circle at the center of the tennis ball
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

# Display the results
cv2.imshow("Tennis Ball Tracker", frame)
cv2.imshow("Mask", mask)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
