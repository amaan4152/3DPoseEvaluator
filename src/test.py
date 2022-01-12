import cv2

cap = cv2.VideoCapture("parallel_rail_hurdle_Without EXO_VIEWPORT_1.mp4")
f_num = 3300
while cap.isOpened():
    s, f = cap.read()
    if not s:
        print("ERROR!")
        break
    cv2.imshow("WINDOW", f)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    f_num += 1
print(f_num)
cap.release()
cv2.destroyAllWindows()
