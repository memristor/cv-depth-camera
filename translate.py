#! /usr/bin/env python3

import numpy as np
import plot

table_origin = np.array([0,0,0])
table_base = np.array([[1,0,0],[0,1,0],[0,0,1]])

camera_origin = np.array([5,0,1])
camera_base = np.array([[0,1,0],[-0.287348, 0, 0.957826],[-0.957826,0,-0.287348]])

plot.table()
plot.point(table_origin, color='blue')
plot.vectors(table_base, color='blue')
plot.point(camera_origin, color='orange')
plot.vectors(camera_base, camera_origin, color='orange')

def convert_point(point, rotation_matrix, relative_translation=np.array([0,0,0])):
  return np.linalg.inv(rotation_matrix).dot(point) + relative_translation

def rotation_matrix(original_points, converted_points, relative_translation=np.array([0,0,0])):
  return np.linalg.inv(original_points).dot(converted_points - relative_translation)


o1 = np.array([1,1,1])
c1 = np.linalg.inv(camera_base).dot(o1) + camera_origin

o2 = np.array([2,5,7])
c2 = np.linalg.inv(camera_base).dot(o2) + camera_origin

o3 = np.array([-3,4,0])
c3 = np.linalg.inv(camera_base).dot(o3) + camera_origin

print(rotation_matrix(np.array([o1,o2,o3]), np.array([c1,c2,c3]), camera_origin).round(decimals=3))


plot.point(o1, color='purple', label='table')
plot.point(c1, color='red', label='camera')

plot.show()