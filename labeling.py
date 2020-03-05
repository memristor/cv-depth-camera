#! /usr/bin/env python3

import cv2
import numpy as np
import math

im = cv2.imread('../blender/terrain-hd.png')
cmask = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)

tags = {
  'buoy': {
    'red': [],
    'green': []
  },
  'dock': {
    'red': [],
    'green': []
  }
}

color = 'red'
action = 'dock'
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
  tags[last[0]][last[1]].pop()
  auto_change()
  render_tags()

def clicker(event, x, y, flags, param):
  global tags
  if event == cv2.EVENT_LBUTTONDOWN and not done:
    print('mdown', x, y)
    tags[action][color].append((x, y))
    undo_stack.append((action, color))
    auto_change()
    render_tags()

def auto_change():
  global tags, action, color, done
  done = False
  if len(tags['dock']['red']) < 2:
    action = 'dock'
    color = 'red'
  elif len(tags['dock']['red']) == 2:
    action = 'dock'
    color = 'green'
    if len(tags['dock']['green']) < 2:
      action = 'dock'
      color = 'green'
    elif len(tags['dock']['green']) == 2:
      action = 'buoy'
      color = 'red'
      if len(tags['buoy']['red']) < 2:
        action = 'buoy'
        color = 'red'
      elif len(tags['buoy']['red']) == 2:
        action = 'buoy'
        color = 'green'
        if len(tags['buoy']['green']) < 2:
          action = 'buoy'
          color = 'green'
        else:
          done = True

output = np.zeros([im.shape[0] + 100, im.shape[1], 3], dtype=np.uint8)
def render_tags():
  global tags, cmask
  cmask = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.uint8)
  for dkey in tags:
    for ckey, color in tags[dkey].items():
      for tag in range(len(color)):
        x, y = color[tag]
        c = (16, 30,187) if ckey == 'red' else (67, 111, 0)
        if dkey == 'buoy':
          cmask = cv2.circle(cmask, (x, y), 40, c, thickness=-1)
          cmask = cv2.circle(cmask, (x, y), 30, (255, 255, 255), thickness=5)
        elif dkey == 'dock':
          cmask = cv2.rectangle(cmask, (x - 35, y - 35), (x + 35, y + 35), c, thickness=-1)
          cmask = cv2.rectangle(cmask, (x - 25, y - 25), (x + 25, y + 25), (255, 255, 255), thickness=5)
        cmask = cv2.putText(
          cmask,
          str(tag),
          (x - 10 - math.floor(math.log(tag if tag > 0 else 1, 10)) * 13, y + 10),
          cv2.QT_FONT_NORMAL,
          1, (255, 255, 255), thickness=3
        )

cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('image', clicker)

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
    # print(bmask.shape)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
      toggle_color()
    if key == ord('x'):
      toggle_action()
    if key == ord('z'):
      undo()
    if key == ord('v') and done:
      print(tags)
    if key == ord('q'):
        break