'''
License:
    Pushpak.AI
    
Description:
This is the integration script that takes the first frame of the given video to mark the points for zebra crossing and traffic light.

Usage:
    from get_line.py import roi2
'''





from typing import overload
import cv2 
import numpy as np
from PIL import Image




mouse_pts = []
# two_points = []
six_point = []
# scale_w = 1.2 / 2
# scale_h = 4 / 2

# circles1 = np.zeros((2,2),np.int)
circles1 = np.zeros((6,2),np.int)
counter =0



#Get mouse clicks
def get_mouse_points2(event, x, y, flags, param):
    # Used to mark 2 points on the frame zero of the video that will be warped
   
    global mouseX, mouseY, mouse_pts, counter
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x,y)
        circles1[counter]= x,y
        counter+=1
        mouseX, mouseY = x, y
        print(circles1)
        #cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)

num_mouse_points = 0
first_frame_display = True 



def roi2(frame):
    global six_points
    # if frame_num == 0:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
    while True:
        image = frame
        # cv2.namedWindow("Resized Window", cv2.WINDOW_NORMAL)
        cv2.imshow('new Window', image)
        cv2.setMouseCallback("new Window", get_mouse_points2)
        cv2.waitKey(1) 
        #cv2.circle(image, mouse_pts,10, (0, 255, 255), 10)
        
        # for x in range(0,2):
        for x in range(0,6):
            cv2.circle(image,(circles1[x][0],circles1[x][1]),3,(0,255,0),cv2.FILLED)   # to show circles over the frame.   
        if len(mouse_pts) == 6:
        # if len(mouse_pts) == 2:
            #cv2.destroyWindow("image")
            cv2.destroyWindow("new Window")
            break
        first_frame_display = False
    # two_points = mouse_pts
    six_points = mouse_pts
    print(six_points)
    # print(two_points)

    # draw polygon of ROI
    pts = np.array(
        [six_points[0], six_points[1]], np.int32
    )

    # pts = np.array(
    #     [two_points[0], two_points[1]], np.int32
    # )

    # print("The first point cordinates: ", six_points[0])
    # print("The second point cordinates: ", six_points[1])

    y1 = six_points[0][1]
    y2 = six_points[1][1]

    x1 = six_points[0][0]
    x2 = six_points[1][0]

    # y1 = two_points[0][1]
    # y2 = two_points[1][1]

    # x1 = two_points[0][0]
    # x2 = two_points[1][0]
    # print("The x value is ", x1, x2)
    # print("The y value is ", y1, y2)
    
    y1_2 = six_points[2][1]
    y2_2 = six_points[3][1]

    x1_2 = six_points[2][0]
    x2_2 = six_points[3][0]
    # # print("The x value is ", x1, x2)
    # # print("The y value is ", y1, y2)

    # widthlight = x2lght - x1lght
    # heightlight = y2lght - y1lght
    y1_3 = six_points[4][1]
    y2_3 = six_points[5][1]

    x1_3 = six_points[4][0]
    x2_3 = six_points[5][0]

    # return x1, x2, y1, y2, x1lght, y1lght, widthlight, heightlight
    return x1, x2, y1, y2, x1_2, x2_2, y1_2, y2_2, x1_3, x2_3, y1_3, y2_3, 
    # return x1, x2, y1, y2