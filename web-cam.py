#This is a start seperate the webcam from the inference.
#Finish or dump this code


import cv2

cap = cv2.VideoCapture(1)
count = 0

while (True):
    count = count + 1

    ret, frame = cap.read()
    coin = Coin(frame)
    rotated = coin.rotated_image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rotated, coin.predicted_date, (300,240),font, 1,(0,0,0),2)
    cv2.imshow('rotated', rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
