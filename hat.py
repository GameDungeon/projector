import time
from pykinect import nui
import numpy as np
import cv2
from imutils.perspective import four_point_transform

video = np.empty((480,640,4),np.uint8)

def video_handler_function(frame):  
    global video   
    frame.image.copy_bits(video.ctypes.data)

depth_to_color = nui.Camera.get_color_pixel_coordinates_from_depth_pixel
skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image

#cv2.namedWindow('Kinect Video Stream', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Transform', cv2.WINDOW_NORMAL)
cv2.namedWindow('Projector', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

kinect = nui.Runtime()

kinect.video_frame_ready += video_handler_function
kinect.video_stream.open(nui.ImageStreamType.Video,2,nui.ImageResolution.Resolution640x480,nui.ImageType.Color)

def get_transform(cont):
    # convert img to grayscale
    gray = cv2.cvtColor(cont, cv2.COLOR_BGR2GRAY)
    gray = 255-gray

    # blur image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
    thresh = 255-thresh

    # apply morphology
    kernel = np.ones((5,5), np.uint8)
    rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

    # thin
    kernel = np.ones((5,5), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    big_contour = None
    area_thresh = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > area_thresh:
            area_thresh = area
            big_contour = approx

    return big_contour
    

pnts = None
screen_size = (1080, 1920, 3)
white = False
border = False

face_cascade = cv2.CascadeClassifier('face.xml')
fist_cascade = cv2.CascadeClassifier('fist.xml')

hat = cv2.imread("hats/top_hat.png")

while True:
    if pnts is None:
        if not white:
            cv2.imshow("Projector", np.full(screen_size, 255, np.uint8))
            white = True
            cv2.waitKey(1000)
            continue

        pnts = get_transform(video.copy())
        white = False
        if pnts is not None:
            pnts = pnts.reshape(4,2)
        else:
            continue

    trans = four_point_transform(video, pnts)
    project = np.full(screen_size, 0, np.uint8)
    ratio = tuple(x / y for x, y in zip(screen_size[:2], trans.shape[:2]))

    gray = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 4)
    fists = fist_cascade.detectMultiScale(gray, 1.05, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(trans,   (x, y), (x+w, y+h), (255, 0, 0), 2)
        w = round(w*ratio[0])
        x = round(x*ratio[0])
        y = round(y*ratio[1])
        p_hat = cv2.resize(hat, (w, round(hat.shape[1] * (w / hat.shape[0]))))

        # get the number of rows & columns of matrices
        proj_m, proj_n   = project.shape[:2]
        p_hat_m, p_hat_n = p_hat.shape[:2]

        # form an empty, large array
        enlarged = np.empty((p_hat_m + proj_m, proj_n + p_hat_n, 3), np.uint8)

        # put the `project` inside
        enlarged[p_hat_m:, :-p_hat_n, :] = project

        # paste `p_hat` in without worries
        enlarged[y: y + p_hat_m, x: x + p_hat_n, :] = p_hat

        # crop the excess parts to get `project` back
        project = enlarged[p_hat_m:, :-p_hat_n, :]

    for (x, y, w, h) in fists:
        cv2.rectangle(trans,   (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(project, (x, y), (x+w, y+h), (255, 0, 0), 2)

    project = cv2.flip(project, 1)

    if border:
        project = cv2.copyMakeBorder(
                 project, 
                 75, 
                 100, 
                 100, 
                 100, 
                 cv2.BORDER_ISOLATED, 
                 value=(0, 255, 255)
              )


    cv2.imshow("Projector", project)
    cv2.imshow("Transform", trans)
    
    #esc to quit
    key = cv2.waitKey(1)
    if key == 114:  pnts = None
    elif key == 98: border = not border
    elif key == 27: break


kinect.close()
cv2.destroyAllWindows()
