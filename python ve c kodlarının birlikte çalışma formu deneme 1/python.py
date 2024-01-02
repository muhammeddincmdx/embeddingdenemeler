import cv2
import numpy as np
import ctypes

# Shared library'yi yükle
lib = ctypes.CDLL('./libprintvalues.so')

# C işlevine erişim
print_values = lib.print_values

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_blue = np.array([100, 100, 100])
    high_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        x_center = int(x + w / 2)
        y_center = int(y + h / 2)

        scale = max(w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 2, (0, 255, 0), -1)
        cv2.putText(frame, f"Center: ({x_center}, {y_center}), Scale: {scale}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # C işlevine x, y, scale değerlerini aktar
        print_values(x_center, y_center, scale)

        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()