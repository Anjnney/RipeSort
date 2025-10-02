import cv2
import numpy as np
import matplotlib.pyplot as plt

def classify_fruit(frame):
    """
    Classify fruit as Ripe or Unripe based on color (HSV thresholds).
    """

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for green (unripe) and red/yellow (ripe)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    # Create masks
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = mask_red1 | mask_red2
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Count pixels
    green_pixels = np.sum(mask_green > 0)
    ripe_pixels = np.sum(mask_red > 0) + np.sum(mask_yellow > 0)

    # Decide classification
    if ripe_pixels > green_pixels:
        label = "Ripe"
        color = (0, 0, 255)  # Red text
    else:
        label = "Unripe"
        color = (0, 255, 0)  # Green text

    # Add label to frame
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)

    return frame, label, mask_green, mask_red | mask_yellow

# ---------------------- MAIN ----------------------

cap = cv2.VideoCapture(0)  # Use webcam (change index if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    classified_frame, label, mask_green, mask_ripe = classify_fruit(frame)

    # Show original + classification
    cv2.imshow("Fruit Classification", classified_frame)
    cv2.imshow("Green Mask (Unripe)", mask_green)
    cv2.imshow("Ripe Mask (Red+Yellow)", mask_ripe)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
