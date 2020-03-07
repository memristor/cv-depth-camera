#! /usr/bin/env python3

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import math
# import plot
import matplotlib.pyplot as plt

pipeline = rs.pipeline

align = rs.align

depth_scale = 0

mfilt = False

clc_pixel_angles = False
pixel_angles = np.array
pxa_x = np.array
pxa_y = np.array

width, height = 1280, 720

def init():
    
    global pipeline, align, depth_scale

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

def stop():

    global pipeline

    pipeline.stop()

init()

def get_frame_rgbd():

    global pipeline, align, depth_scale, mfilt, \
           clc_pixel_angles, pixel_angles, pxa_x, pxa_y

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Intrinsic properties
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

    fov = depth_intrin.ppx / 10, depth_intrin.ppy / 10

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        return None

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    # grey_color = 153
    # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    distance_clip_cm = 100
    depth_image_cm = depth_image.copy()
    depth_image_cm = depth_image_cm * depth_scale * 100
    depth_image_cm[depth_image_cm > distance_clip_cm] = 255
    depth_image_cm = np.array(depth_image_cm, dtype=np.uint8)
    # if mfilt:
    #     depth_image_cm = cv2.medianBlur(depth_image_cm, 7)

    # angleX = (np.arange(-width / 2, width / 2) / width * fov[0])
    # angleY = (np.arange(-height / 2, height / 2) / height * fov[1])
    # x = np.abs(depth_image_cm[0, :] * np.sin(np.pi * angleX / 180))
    # y = np.abs(depth_image_cm[:, 0] * np.sin(np.pi * angleY / 180))

    diagonal_length = math.sqrt((height / 2)**2 + (width / 2)**2)
    diagonal_fov = math.sqrt((fov[0] / 2)**2 + (fov[1] / 2)**2)
    pinhole_depth_px = abs((diagonal_length) / math.tan(diagonal_fov))

    center = [width // 2, height // 2]
    if not clc_pixel_angles:

        pixel_angles = np.zeros((height, width), dtype=np.float)
        for y in range(height):
            for x in range(width):
                from_origin_px = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                pixel_angles[y, x] = math.asin(from_origin_px / pinhole_depth_px) * 180 / math.pi // 2
        clc_pixel_angles = True

        pxa_x = np.zeros((height, width), dtype=np.float)
        for y in range(height):
            pxa_x[y,:] = pixel_angles[height//2, :] * 1 if y < height // 2 else -1
        
        pxa_y = np.zeros((height, width), dtype=np.float)
        for x in range(width):
            pxa_y[:,x] = pixel_angles[:, width//2] * 1 if x < width // 2 else -1

    z = depth_image_cm
    d = z.copy() * np.cos(pixel_angles * math.pi / 180)
    x = np.array(d.copy() * np.tan(pxa_x * math.pi / 180), dtype=np.float)
    y = np.array(d.copy() * np.tan(pxa_y * math.pi / 180), dtype=np.float)

    return color_image, depth_image_cm
    # return x, y, z, color_image


# x, y, z = get_frame_rgbd()

# points = np.zeros((1280 * 720, 3), dtype=np.uint8)
# points[:, 0] = x.transpose().reshape((1280 * 720))
# points[:, 1] = y.transpose().reshape((1280 * 720))
# points[:, 2] = z.transpose().reshape((1280 * 720))

# plot.points(points)
# plot.show()

# dataset_index = 

import os
filecount =  len([f for f in os.listdir('./dataset/')if os.path.isfile(os.path.join('./dataset/', f))])

dataset_file_index = filecount // 2

# Streaming loop
try:
    while True:
        
        # x, y, z, rgb = get_frame_rgbd()
        # xyz = np.zeros((720, 1280, 3), dtype=np.uint8)
        # xyz[:,:,0] = x
        # xyz[:,:,1] = y
        # xyz[:,:,2] = z
        # cv2.namedWindow('Z', cv2.WINDOW_GUI_EXPANDED)
        # cv2.imshow('Z', z)
        # cv2.namedWindow('X', cv2.WINDOW_GUI_EXPANDED)
        # cv2.imshow('X', x)
        # cv2.namedWindow('Y', cv2.WINDOW_GUI_EXPANDED)
        # cv2.imshow('Y', y)
        rgb, d = get_frame_rgbd()

        # plt.imshow(frame)
        # plt.show()
        # break

        cv2.namedWindow('frame', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('frame', rgb)
        cv2.namedWindow('d', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('d', d)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key == ord('x'):
            cv2.imwrite('./dataset/rgb_%i.png' % dataset_file_index, rgb)
            cv2.imwrite('./dataset/depth_%i.png' % dataset_file_index, d)
            dataset_file_index += 1

        
finally:
    
    stop()