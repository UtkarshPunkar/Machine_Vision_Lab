import cv2
import numpy as np


# ==========================================
# VIDEO PATH
# ==========================================

video_path = r"C:\Users\utkar\Desktop\Vehicle_Movement_Analysis\traffic.mp4"


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Video not found")
    exit()


# ==========================================
# BACKGROUND SUBTRACTION
# ==========================================

bg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=60,
    detectShadows=True
)


# ==========================================
# VEHICLE TRACKING VARIABLES
# ==========================================

vehicle_count = 0

vehicle_id = 0

vehicles = {}


# Counting line position

line_y = 300



# ==========================================
# PROCESS VIDEO
# ==========================================

while True:


    ret, frame = cap.read()


    if not ret:
        break


    frame = cv2.resize(frame,(900,500))


    # ======================================
    # BACKGROUND SUBTRACTION
    # ======================================

    mask = bg.apply(frame)


    # Remove shadows

    _,mask=cv2.threshold(
        mask,
        200,
        255,
        cv2.THRESH_BINARY
    )


    # Noise removal

    kernel=np.ones((5,5),np.uint8)

    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )



    # ======================================
    # FIND VEHICLES
    # ======================================

    contours,_=cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    current_centers=[]



    for cnt in contours:


        area=cv2.contourArea(cnt)


        # Ignore small objects

        if area > 1000:


            x,y,w,h=cv2.boundingRect(cnt)


            cx=int(x+w/2)
            cy=int(y+h/2)


            current_centers.append(
                (cx,cy)
            )


            # Draw vehicle box

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )


            cv2.circle(
                frame,
                (cx,cy),
                5,
                (0,0,255),
                -1
            )


            cv2.putText(
                frame,
                "Vehicle",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )



    # ======================================
    # COUNT VEHICLES CROSSING LINE
    # ======================================


    for center in current_centers:


        cx,cy=center


        if cy > line_y:


            vehicle_count += 1



    # ======================================
    # DRAW COUNTING LINE
    # ======================================

    cv2.line(
        frame,
        (0,line_y),
        (900,line_y),
        (255,0,0),
        3
    )


    cv2.putText(
        frame,
        "Vehicle Count : "+str(vehicle_count),
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        3
    )


    # ======================================
    # SHOW OUTPUT
    # ======================================


    cv2.imshow(
        "Vehicle Detection and Counting",
        frame
    )


    cv2.imshow(
        "Motion Mask",
        mask
    )



    if cv2.waitKey(30)==ord('q'):
        break



cap.release()

cv2.destroyAllWindows()