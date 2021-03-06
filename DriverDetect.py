#!/usr/bin/env python

'''
Daniel Velazquez - May 2018
Driver Detection and Sleepy Warning
based on OpenCV Sample facedetect.py

face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function
from timeit import default_timer as timer


import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

font = cv2.FONT_HERSHEY_SIMPLEX

# Warning signs
Triangle = cv2.imread('Warning.svg')

FrameCount= 1.0
DriverAway= 1.0
DriverPresent= 1.0
DriverSleepy= 1.0

Vertical= 0

GeneralCounter_start = timer()
DriverOKCounter_start = timer()
SleepyCounter_start = timer()
DriverNGCounter_start = timer()
GeneralCounter_end = timer()
DriverOKCounter_end = timer()
SleepyCounter_end = timer()
DriverNGCounter_end = timer()


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def ScanFace(img, rects):
	while Vertical < y2:
		cv2.line(img,(x1,y1),(x2,Vertical),(255,255,255),1)
		Vertical += 1



if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = detect(gray, cascade)
        vis = img.copy()
        output = vis.copy()
        overlay = vis.copy()
        draw_rects(vis, rects, (0, 255, 0))					# GREEN Square on Face BGR
        for x1, y1, x2, y2 in rects:
			Vertical=y1
			while Vertical < y2:
				cv2.line(vis,(x1,Vertical),(x2,Vertical),(255,0,0),1)
				Vertical+=5
        if (len(rects)) == 0:
			DriverAway=DriverAway+1.0
			cv2.putText(vis,'Driver Away',(10,60), font, 2,(255,255,255),2,cv2.LINE_AA)

        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                #cv2.rectangle(overlay, (0,0),(1200,y1), (0, 255, 0), -1)   # TOP GREEN Line
                #cv2.rectangle(overlay, (0,y2),(1200,600), (0, 255, 0), -1)   # GREEN RED Line
                #cv2.rectangle(overlay, (0,0),(x1,600), (0, 255, 0), -1)   # LEFT GREEN Line
                #cv2.rectangle(overlay, (x2,0),(1200,600), (0, 255, 0), -1)   # RIGHT GREEN Line
                cv2.addWeighted(overlay, 0.2, vis, 1 - 0.0, 0, vis)
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                #draw_rects(vis_roi, subrects, (255, 0, 0))
                cv2.putText(vis,'Driver Present',(10,60), font, 2,(255,255,255),2,cv2.LINE_AA)
                DriverPresent=DriverPresent+1.0
                if DriverPresent>=15:
					if (len(subrects)) < 2:
						cv2.putText(vis,'Driver Sleepy',(10,120), font, 2,(255,255,255),2,cv2.LINE_AA)
						cv2.rectangle(overlay, (x1,y1),(x2,y2), (0, 0, 255), -1)   # RED Square
						cv2.addWeighted(overlay, 0.5, vis, 1, 1, vis)
						DriverSleepy=DriverSleepy+1.0
						DriverPresent=DriverPresent-1.0

		
		
        FrameCount=FrameCount+1
        #print("Frame Count",FrameCount)
        #print("Driver Away",DriverAway)
        #print("Driver Present",DriverPresent)
        #print("Driver Sleepy",DriverSleepy)
        #cv2.putText(vis,'Frame Count',(785,490), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Driver Away',(785,505), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Driver Present',(785,520), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis,'Driver Sleepy',(785,535), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        #cv2.putText(vis, "{:.0f}".format(FrameCount),(910,490), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis, "{:.1f}".format((DriverAway/FrameCount)*100.0),(910,505), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis, "{:.1f}".format(((DriverPresent-1)/FrameCount)*100.0),(910,520), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(vis, "{:.1f}".format((DriverSleepy/FrameCount)*100.0),(910,535), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        #cv2.putText(vis,'Total:',(785,475), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        #cv2.putText(vis, "{:.1f}".format( ((DriverSleepy/FrameCount)*100.0)+((DriverPresent/FrameCount)*100.0)+((DriverAway/FrameCount)*100.0) ),(910,475), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        cv2.imshow('Driver Detection', vis)
        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
