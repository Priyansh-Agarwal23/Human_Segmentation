import cv2
import time
import numpy as np
from rembg import remove
from PIL import Image

cap = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

while True:
    check, frame = cap.read()
    if not check:
        print("No Webcam Access")

    bg_frame = np.full((400, 400, 3), (0, 0, 0), np.uint8)

    default_bg = cv2.imread('background/bg_3.jpg')

    frame_count += 1

    output = remove(frame)

    image = cv2.resize(output, (400, 400))

    alpha_channel = image[:, :, 3]

    _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)

    # Extract the RGB channels from the image
    rgb_channels = image[:, :, :3]

    # Apply the inverted mask to the frame
    background = cv2.bitwise_and(bg_frame, bg_frame, mask=inverted_mask)

    # Apply the mask to the image
    foreground = cv2.bitwise_and(rgb_channels, rgb_channels, mask=mask)

    # Combine the foreground and background
    result = cv2.add(background, foreground)

    clr_cnt = 0
    
    for i in range(400):
        for j in range(400):
            if [0, 0, 0] in result[i][j]:
                clr_cnt += 1

    print(clr_cnt)

    if clr_cnt < 128000:
        default_bg = cv2.imread('background/bg_4.jpg')
        default_bg = cv2.resize(default_bg, (1920, 1080))
        default_bg[0:400, 0:400] = np.where(result == (0, 0, 0), default_bg[0:400, 0:400], result)

    cv2.namedWindow('background', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('background', default_bg)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()


