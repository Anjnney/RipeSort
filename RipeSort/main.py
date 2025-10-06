import cv2
import numpy as np

def is_fruit_like_object(contour):
    area = cv2.contourArea(contour)
    if area < 3000 or area > 50000:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < 0.4:
        return False
    return True

def is_in_valid_detection_zone(contour, frame_shape):
    x, y, w, h = cv2.boundingRect(contour)
    frame_h, frame_w = frame_shape[:2]
    center_x = x + w // 2
    center_y = y + h // 2
    if (0.15 * frame_w < center_x < 0.85 * frame_w and
        0.15 * frame_h < center_y < 0.85 * frame_h):
        return True
    return False

def get_fruit_confidence(mask, contour):
    area = cv2.contourArea(contour)
    mask_pixels = cv2.countNonZero(mask)
    color_consistency = mask_pixels / area if area > 0 else 0
    size_score = min(area / 10000, 1.0)
    confidence = (color_consistency * 0.6 + size_score * 0.4)
    return confidence

def classify_fruit(frame):
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

    detected = []
    fruit_regions = {}

    for fruit, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_confidence = 0

        for cnt in contours:
            if (is_fruit_like_object(cnt) and
                is_in_valid_detection_zone(cnt, frame.shape)):
                confidence = get_fruit_confidence(mask, cnt)
                if confidence > 0.5 and confidence > best_confidence:
                    best_contour = cnt
                    best_confidence = confidence

        if best_contour is not None:
            detected.append(f"{fruit} ({best_confidence:.1f})")
            fruit_regions[fruit] = best_contour

    if not detected:
        detected.append("No fruit detected")

    return detected, masks, fruit_regions

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, masks, fruit_regions = classify_fruit(frame)

        overlay = frame.copy()
        msg_color = (0, 255, 0) if "No fruit detected" not in detected else (0, 0, 255)
        cv2.putText(overlay, " + ".join(detected), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, msg_color, 2, cv2.LINE_AA)

        for fruit, contour in fruit_regions.items():
            cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(overlay, fruit, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("RipeSort - MultiFruit", overlay)

        for i, (fruit, mask) in enumerate(masks.items()):
            cv2.imshow(fruit, mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
