import cv2

cap = cv2.VideoCapture(1)
count = 0

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    key = cv2.waitKey(3) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'{count}.png', img)
        count += 1