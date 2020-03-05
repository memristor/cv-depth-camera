#! /usr/bin/env python3

import numpy as np
import plot

table_origin = np.array([0,0,0])
table_base = np.array([[1,0,0],[0,1,0],[0,0,1]])
table_size = np.array([30, 20])

camera_origin = np.array([15.5,0,5])
camera_base = np.array([[0,1,0],[-0.287348, 0, 0.957826],[-0.957826,0,-0.287348]])

plot.table(table_size)
plot.point(table_origin, color='blue', label='table_origin')
plot.vectors(table_base, scale=3, color='blue')
plot.point(camera_origin, color='magenta', label='camera_origin')
plot.vectors(camera_base, camera_origin, scale=3, color='magenta')

plot.line(np.array([15,3,0]), np.array([10.5,3,0]), color='red')
plot.line(np.array([15,-3,0]), np.array([10.5,-3,0]), color='green')
plot.line(np.array([10.5,-3,0]), np.array([10.5,3,0]), color='orange')

plot.line(np.array([-15,3,0]), np.array([-10.5,3,0]), color='green')
plot.line(np.array([-15,-3,0]), np.array([-10.5,-3,0]), color='red')
plot.line(np.array([-10.5,-3,0]), np.array([-10.5,3,0]), color='orange')

def convert_point_from(point, rotation_matrix, relative_translation=np.array([0,0,0])):
  return np.linalg.inv(rotation_matrix).dot(point) + relative_translation

def rotation_matrix(original_points, converted_points, relative_translation=np.array([0,0,0])):
  return np.linalg.inv(original_points).dot(converted_points - relative_translation)


o1 = np.array([1,1,1])
c1 = convert_point_from(o1, camera_base, camera_origin)
o2 = np.array([2,5,7])
c2 = convert_point_from(o2, camera_base, camera_origin)
o3 = np.array([-3,4,0])
c3 = convert_point_from(o2, camera_base, camera_origin)

# print(rotation_matrix(
#   np.array([o1,o2,o3]), np.array([c1,c2,c3]), camera_origin
# ).round(decimals=3))

plot.cylinder(np.array([12,2,0]), color='red')
plot.cylinder(np.array([11,-3,0]), color='red')
plot.cylinder(np.array([14,1,0]), color='green')

plot.show()