import cv2
import numpy as np

cap = cv2.VideoCapture("videos/conveyor.mp4")

roi_x = 120
roi_y = 400
roi_w = 200
roi_h = 220

kernel = np.ones((5, 5), np.uint8)
total_count = 0
counted_centers = []


count_line_y = roi_h // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI
    cv2.rectangle(frame,
                  (roi_x, roi_y),
                  (roi_x + roi_w, roi_y + roi_h),
                  (255, 0, 0), 2)

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

  
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contours
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

   
    cv2.line(roi, (0, count_line_y), (roi_w, count_line_y), (0, 0, 255), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

      
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)

        
        if (count_line_y - 5) < cy < (count_line_y + 5):
            if cx not in counted_centers:
                total_count += 1
                counted_centers.append(cx)

    # Display running total
    cv2.putText(frame,
                f"Total Count: {total_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255), 2)

    cv2.imshow("Yellow Object Counter", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

