# main.py
import cv2
import numpy as np

def classify_fruit(frame, red_thresh=5000, green_thresh=5000):
    """
    Input: BGR frame
    Returns: (label, green_mask, red_mask, red_count, green_count)
    - label: "Apple (Ripe)", "Unripe/Green", or "Unknown"
    - green_mask / red_mask: single-channel masks (0 or 255)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Green range (typical green apples / unripe areas) ---
    green_lower = np.array([35, 50, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # --- Red range (ripe apple red) ---
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # --- Morphological clean-up to remove noise ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # --- Pixel counts ---
    green_count = cv2.countNonZero(green_mask)
    red_count = cv2.countNonZero(red_mask)

    # --- Simple decision logic ---
    if red_count > red_thresh and red_count > green_count:
        label = "Apple (Ripe)"
    elif green_count > green_thresh and green_count > red_count:
        label = "Unripe / Green"
    else:
        label = "Unknown"

    return label, green_mask, red_mask, red_count, green_count


def main():
    cap = cv2.VideoCapture(0)  # change to filepath if you want to test a video file
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    # window names
    win_orig = "Original"
    win_green = "Green Mask"
    win_red = "Red Mask"
    cv2.namedWindow(win_orig)
    cv2.namedWindow(win_green)
    cv2.namedWindow(win_red)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optional: resize for faster processing (uncomment if needed)
        # frame = cv2.resize(frame, (640, 480))

        label, green_mask, red_mask, red_count, green_count = classify_fruit(frame)

        # overlay label and counts on the original frame
        overlay = frame.copy()
        cv2.putText(overlay, f"{label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Red: {red_count}  Green: {green_count}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # show windows: original video + masks
        cv2.imshow(win_orig, overlay)
        cv2.imshow(win_green, green_mask)
        cv2.imshow(win_red, red_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):         # press 'q' to quit
            break
        elif key == ord('s'):       # press 's' to save a snapshot (optional)
            cv2.imwrite("snapshot.png", frame)
            print("Saved snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
