import cv2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)


def display_frame(frame,image):   
        #draw points
        img2 = cv2.drawKeypoints(image,frame.kps, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("image",img2)
        cv2.waitKey(1)