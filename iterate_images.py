#! /usr/bin/env python3

import os
import cv2
import math
import numpy as np
from pprint import pprint

def next_image(start=0, stop=math.inf):
    n = len(os.listdir('./dataset')) // 2
    for i in range(n):
        if i < start or i > stop:
            continue
        else:
            rgb = cv2.imread('./dataset/rgb_%i.png' % i, cv2.IMREAD_COLOR)
            depth = cv2.imread('./dataset/depth_%i.png' % i, cv2.IMREAD_GRAYSCALE)
            yield rgb, depth

def resize(image, dimensions=(720, 480)):
    return cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)

def iterate():
    filter = True
    for rgb, depth in next_image(7, 1000):
        f_rgb = filter_rgb(rgb)
        f_depth = filter_depth(depth)
        cv2.namedWindow('iterate', cv2.WINDOW_GUI_EXPANDED)
        while True:
            # cv2.imshow('iterate', resize(f_rgb if filter else rgb, (1280, 720)))
            cv2.imshow('iterate', f_rgb if filter else rgb)
            key = cv2.waitKey(1) & 0xFF
            if key  == ord('q'):
                return
            if key == ord('f'):
                filter = not filter
            if key == ord(' '):
                break

def filter_rgb(source):

    im = source.copy()
    im = cv2.bilateralFilter(im,3,30,30)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    def rgb_hsv(color):
        pixel = np.zeros((1,1,3), dtype=np.uint8)
        pixel[0,0,:] = np.array(color, dtype=np.uint8)
        hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
        return hsv[0,0,:]

    def tolerances(color_hsv, margin):
        h_min, h_max =  abs((color_hsv[0] - margin [0]) % 180), \
                        abs((color_hsv[0] + margin [0]) % 180)
        if h_min > h_max:
            h_min, h_max = h_max, h_min
        tolerance_low = np.array((
            h_min,
            max(color_hsv[1] - margin [1], 0),
            max(color_hsv[2] - margin [2], 0)
        ), dtype=np.uint8)
        tolerance_high = np.array((
            h_max,
            min(color_hsv[1] + margin [1], 255),
            min(color_hsv[2] + margin [2], 255)
        ), dtype=np.uint8)
        return tolerance_low, tolerance_high

    def mask(color_hsv, tolerance):
        mask_color = cv2.inRange(
            im_hsv,
            *tolerances(color_hsv, tolerance)
        )
        kernel = np.ones((7,7),np.uint8)
        mask_morph = cv2.dilate(mask_color.copy(),kernel,iterations=3)
        mask_morph = cv2.erode(mask_morph,kernel,iterations=3)
        return mask_morph

    # extracted_red = cv2.bitwise_and(im, im, mask=mask(color_red_hsv, tolerance_red_hsv))
    # extracted_green = cv2.bitwise_and(im, im, mask=mask(color_green_hsv, tolerance_green_hsv))

    def contours(mask):
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def euclidian_distance(a, b):
        square_sum = 0
        for a_val, b_val in zip(a, b):
            square_sum += (a_val - b_val) ** 2
        return math.sqrt(square_sum)

    def contour_corner(contour, line_treshold=30):

        edges = contour.copy()
        edges = np.resize(edges, (edges.shape[0] + edges.shape[1], 2)).tolist()
        
        edges.sort(key=lambda s: s[1]) # sort by y location

        bottom = []
        top = []

        for point in edges:
            if point[1] >= edges[-1][1] - line_treshold:
                bottom.append(tuple(point))
            elif point[1] <= edges[0][1] + line_treshold:
                top.append(tuple(point))
            
        n = sorted(top, key=lambda s: s[1])[0][1]
        s = sorted(bottom, key=lambda s: s[1])[-1][1]
        
        top.sort(key=lambda s: s[0])
        bottom.sort(key=lambda s: s[0])
        nw, ne = (top[0][0], n), (top[-1][0], n)
        sw, se = (bottom[0][0], s), (bottom[-1][0], s)

        # for point in bottom:
        #     cv2.circle(im, point, 3,(150, 50, 255), thickness=-1)
        # for point in top:
        #     cv2.circle(im, point, 3,(150, 50, 255), thickness=-1)

        return nw, ne, sw, se

    def bounding_box(
        color,
        color_tolerance=(90, 255, 255),
        height_range=(0, math.inf),
        width_range=(0, math.inf),
        hw_ratio_range=(0, math.inf),
        orientation_range=(0, 90)
    ):

        vote_list = []

        for cnt in contours(mask(rgb_hsv(color), color_tolerance)):
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 4:

                # Orientation
                rows,cols = im.shape[:2]
                [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                angle = abs(math.atan(vy/vx) * 180 / math.pi)

                if angle < orientation_range[0] or \
                    angle > orientation_range[1]:
                    continue

                # Shape
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                height = euclidian_distance(box[0], box[1])
                width = euclidian_distance(box[1], box[2])
                if width > height:
                    width, height = height, width

                if height == 0 or width == 0:
                    continue

                if (height < height_range[0] or height > height_range[1]) or \
                    (width < width_range[0] or width > width_range[1]) or \
                    (height / width < hw_ratio_range[0] and \
                     height / width > hw_ratio_range[1]):
                    continue

                vote_list.append((
                    height,
                    width,
                    angle,
                    approx,
                    box
                ))

                # Color
                mask_box = np.zeros(im.shape[:2], dtype=im.dtype)
                cv2.drawContours(mask_box, [approx], 0, (255), -1)
                color_mean_hsv = cv2.mean(im_hsv, mask=mask_box)
                color_difference = euclidian_distance(color, color_mean_hsv)

        vote_list.sort(key=lambda s: s[0], reverse=True)

        for index in range(len(vote_list)):
            shape = vote_list[index]
            # Visualise
            if index == 0:
                # cv2.drawContours(im, [shape[3]], 0, (color[2], color[1], color[0]), 3)
                # cv2.drawContours(im, [shape[4]], 0, (255, 255, 255), 1)
                # cv2.drawContours(im, [shape[3]], 0, (255, 0, 255), 3)
                cv2.putText(
                    im, '%ix%ipx %ideg' % (shape[1], shape[0], shape[2]),
                    tuple(shape[4][2]), cv2.QT_FONT_NORMAL, 0.7,
                    (color[2], color[1], color[0]), thickness=2
                )
                nw, ne, sw, se = contour_corner(shape[3])
                box = np.array([nw, ne, se, sw], dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(im,[box],True,(255,255,255), thickness=2)

                return box

    color_red = (145, 35, 30)
    color_green = (50, 111, 67)

    # tolerance_red_hsv = (5, 40, 50)
    # tolerance_green_hsv = (25, 60, 50)

    tolerance_red_hsv = (5, 40, 50)
    tolerance_green_hsv = (25, 50, 50)

    bb_red = bounding_box(
        color=color_red,
        color_tolerance=tolerance_red_hsv,
        height_range=(480, 580),
        width_range=(40, 60),
        hw_ratio_range=(10, 14),
        orientation_range=(70, 90)
    )
    bb_green = bounding_box(
        color=color_green,
        color_tolerance=tolerance_green_hsv,
        height_range=(400, 600),
        width_range=(30, 40),
        hw_ratio_range=(11, 15),
        orientation_range=(55, 90)
    )

    return im, bb_red, bb_green

def filter_depth(source):
    return source

iterate()