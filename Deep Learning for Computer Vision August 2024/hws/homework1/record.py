import cv2

rtsp_path = 'rtsp://admin:Test@1234@10.43.64.61/axis-media/media.amp'

filename = 'record.avi'

cap = cv2.VideoCapture(rtsp_path)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 30.0, (1920, 1080))

while True:
    ret, frame = cap.read()

    out.write(frame)

    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
