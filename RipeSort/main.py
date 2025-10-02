# main.py
import cv2
import numpy as np

def classify_fruit(frame):
    """
    Detect multiple fruits based on HSV color segmentation.
    Returns (label, masks_dict) where:
    - label = detected fruit name(s)
    - masks_dict = dictionary of masks for visualization
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = {}

    # --- Apples (Ripe - Red) ---
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    masks["Apple (Ripe)"] = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # --- Apples (Unripe - Green) ---
    green_lower = np.array([35, 50, 40])
    green_upper = np.array([85, 255, 255])
    masks["Apple (Unripe)"] = cv2.inRange(hsv, green_lower, green_upper)

    # --- Bananas (Yellow) ---
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    masks["Banana"] = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # --- Oranges (Orange) ---
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([25, 255, 255])
    masks["Orange"] = cv2.inRange(hsv, orange_lower, orange_upper)

    # --- Watermelons (Dark Green shell + Red inside) ---
    dark_green_lower = np.array([35, 80, 20])
    dark_green_upper = np.array([85, 255, 100])
    masks["Watermelon Shell"] = cv2.inRange(hsv, dark_green_lower, dark_green_upper)

    masks["Watermelon Flesh"] = masks["Apple (Ripe)"].copy()  # reuse red detection

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for key in masks:
        masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_OPEN, kernel)
        masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_CLOSE, kernel)

    # Decide which fruits are present (based on pixel count)
    detected = []
    for fruit, mask in masks.items():
        count = cv2.countNonZero(mask)
        if count > 5000:  # threshold
            detected.append(fruit)

    if not detected:
        detected.append("Unknown")

    return detected, masks


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, masks = classify_fruit(frame)

        # Overlay results on frame
        overlay = frame.copy()
        cv2.putText(overlay, " + ".join(detected), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show original + masks in separate windows
        cv2.imshow("RipeSort - MultiFruit", overlay)

        for i, (fruit, mask) in enumerate(masks.items()):
            cv2.imshow(fruit, mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
