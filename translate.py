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

camera_point = np.array([1,1,1])

camera_point_ = np.linalg.inv(camera_base).dot(camera_point) + camera_origin

print(np.linalg.norm(camera_point_ - camera_origin))

plot.point(camera_point, color='blue', label='originalna')
plot.point(camera_point_, color='green', label=str(camera_point_))

plot.show()