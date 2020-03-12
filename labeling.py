#! /usr/bin/env python3

import cv2
import numpy as np
import math
from time import sleep
import copy
import iterate_images

im = np.array
cmask = np.array

# im = cv2.imread('depth_images/slika1_Color.png')
# cmask = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

initial_tags = {
  'buoy': {
    'red': [],
    'green': []
  },
  'dock': {
    'red': [],
    'green': []
  }
}

initial_color = 'red'
initial_action = 'dock'

tags = copy.deepcopy(initial_tags)

color = initial_color
action = initial_action
done = False

def toggle_color():
  global color
  color = 'green' if color == 'red' else 'red'

def toggle_action():
  global action
  action = 'dock' if action == 'buoy' else 'buoy'

def controls() -> np.array:
  global color, action, tags
  controls = np.full([100, im.shape[1], 3], (32, 32, 32), dtype=np.uint8)
  if not done:
    count = len(tags[action][color])
    c = (16, 30,187) if color == 'red' else (67, 111, 0)
    controls = cv2.circle(controls, (60, 50), 40, c, thickness=-1)
    controls = cv2.putText(
      controls, '%s [next: %i]' % (action, count),
      (120, 70), cv2.QT_FONT_NORMAL, 2,
      (255, 255, 255), thickness=2
    )
  else:
    controls = cv2.putText(
      controls,
      'done, press [V] for next picture',
      (10, 70), cv2.QT_FONT_NORMAL, 2,
      (255, 255, 255), thickness=2
    )
  return controls

undo_stack = []
def undo():
  global tags
  if len(undo_stack) == 0:
    return
  last = undo_stack.pop()
  if len(tags[last[0]][last[1]]) == 0:
    return
  tags[last[0]][last[1]].pop()
  auto_change()
  render_tags()

def clicker(event, x, y, flags, param):
  global tags
  if event == cv2.EVENT_LBUTTONDOWN and not done:
    tags[action][color].append((x, y))
    undo_stack.append((action, color))
  elif event == cv2.EVENT_RBUTTONDOWN and not done:
    tags[action][color].append(None)
    undo_stack.append((action, color))
  auto_change()
  render_tags()

def auto_change():
  return
  global tags, action, color, done
  done = False
  if len(tags['dock']['red']) < 4:
    action = 'dock'
    color = 'red'
  elif len(tags['dock']['red']) == 4:
    action = 'dock'
    color = 'green'
    if len(tags['dock']['green']) < 4:
      action = 'dock'
      color = 'green'
    elif len(tags['dock']['green']) == 4:
      action = 'buoy'
      color = 'red'
      done = True
      # if len(tags['buoy']['red']) < 2:
      #   action = 'buoy'
      #   color = 'red'
      # elif len(tags['buoy']['red']) == 2:
      #   action = 'buoy'
      #   color = 'green'
      #   if len(tags['buoy']['green']) < 2:
      #     action = 'buoy'
      #     color = 'green'
      #   else:
      #     done = True

output = np.array
def render_tags():
  global tags, cmask
  cmask = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)
  for dkey in tags:
    for ckey, color in tags[dkey].items():
      for tag in range(len(color)):
        if color[tag] is None:
          continue
        x, y = color[tag]
        c = (16, 30,187) if ckey == 'red' else (67, 111, 0)
        if dkey == 'buoy':
          cmask = cv2.circle(cmask, (x, y), 20, c, thickness=-1)
          cmask = cv2.circle(cmask, (x, y), 15, (255, 255, 255), thickness=2)
        elif dkey == 'dock':
          cmask = cv2.rectangle(cmask, (x - 17, y - 17), (x + 17, y + 17), c, thickness=-1)
          cmask = cv2.rectangle(cmask, (x - 12, y - 12), (x + 12, y + 12), (255, 255, 255), thickness=2)
        cmask = cv2.putText(
          cmask,
          str(tag),
          (x - 5 - math.floor(math.log(tag if tag > 0 else 1, 10)) * 8, y + 5),
          cv2.QT_FONT_NORMAL,
          0.5, (255, 255, 255), thickness=2
        )

cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('image', clicker)

if __name__ == '__main__':

  for rgb, depth in iterate_images.next_image(7, 1000):

    im = rgb
    cmask = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)
    output = np.zeros([im.shape[0] + 100, im.shape[1], 3], dtype=np.uint8)
    undo_stack = []

    while True:

      output[0:im.shape[0],:,:] = im.copy()
      output[im.shape[0]:,:,:] = controls()

      bmask = cv2.add(cmask[:,:,0], cv2.add(cmask[:,:,1], cmask[:,:,2]))
      bmask[bmask > 0] = 255
      bmask = cv2.bitwise_not(bmask)

      output[0:im.shape[0],:,0] = cv2.bitwise_and(output[0:im.shape[0],:,0], bmask)
      output[0:im.shape[0],:,1] = cv2.bitwise_and(output[0:im.shape[0],:,1], bmask)
      output[0:im.shape[0],:,2] = cv2.bitwise_and(output[0:im.shape[0],:,2], bmask)

      output[0:im.shape[0],:,:] = cv2.add(output[0:im.shape[0],:,:], cmask)
    
      cv2.imshow('image', output)
      key = cv2.waitKey(1) & 0xFF

      if key == ord('c'):
        toggle_color()
      if key == ord('x'):
        toggle_action()
      if key == ord('z'):
        undo()
      if key == ord('v') or done:
        print(tags)
        tags = copy.deepcopy(initial_tags)
        print(initial_tags)
        color = initial_color
        action = initial_action
        done = False
        break
      if key == ord('q'):
          exit()