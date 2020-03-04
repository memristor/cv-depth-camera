#! /usr/bin/env python3

import numpy as np
import plot

table_origin = np.array([0,0,0])
table_base = np.array([[1,0,0],[0,1,0],[0,0,1]])

camera_origin = np.array([5,0,1])
camera_base = np.array([[0,1,0],[-0.287,0,0.958],[-0.958,0,-0.287]])

plot.table()
plot.point(table_origin, color='blue')
plot.vectors(table_base, color='blue')
plot.point(camera_origin, color='orange')
plot.vectors(camera_base, camera_origin, color='orange')

camera_point = np.array([2, 2, 0])
plot.point(camera_point)

plot.show()