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

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str


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
        draw_rects(vis, rects, (0, 255, 0))					# GREEN Square on Face
        print(len(rects))
        if (len(rects)) == 0:
			draw_str(vis, (100,100), 'No Driver!')
			
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                cv2.rectangle(overlay, (0,0),(1200,y1), (0, 255, 0), -1)   # TOP GREEN Line
                cv2.rectangle(overlay, (0,y2),(1200,600), (0, 255, 0), -1)   # GREEN RED Line
                cv2.rectangle(overlay, (0,0),(x1,600), (0, 255, 0), -1)   # LEFT GREEN Line
                cv2.rectangle(overlay, (x2,0),(1200,600), (0, 255, 0), -1)   # RIGHT GREEN Line
                cv2.addWeighted(overlay, 0.2, vis, 1 - 0.0, 0, vis)
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
                draw_str(vis, (x2+10,y2), 'Driver Detected!')
                print(len(subrects))
                if (len(subrects)) < 2:
					draw_str(vis, (x2+10,y2+13), 'Driver Sleepy!')
					cv2.rectangle(overlay, (x1,y1),(x2,y2), (0, 0, 255), -1)   # RED Square
					cv2.addWeighted(overlay, 0.2, vis, 1 - 0.0, 0, vis)
        cv2.imshow('Driver Detection', vis)
        
        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

'''
				#for x1_2, y1_2, x2_2, y2:2 in subrects:
				#if len(subrects) > 0:				# if >1
				#draw_str(vis, (x2+10,y2), 'Eyes Open Detected!')
				#print(len(subrects))
				#break;
				#else:
'''
