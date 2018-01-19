#!/usr/bin/env python
# coding:utf-8
# Copyright (C) dirlt

from __future__ import (absolute_import, division, print_function, unicode_literals)
import cv2
import numpy as np
import play

state = cv2.imread('state.png')
resolution = state.shape[:2]
scale = state.shape[1] / 720.
state = cv2.resize(state, (720, int(state.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
print('state shape = {}'.format(state.shape))
if state.shape[0] > 1280:
    s = (state.shape[0] - 1280) // 2
    state = state[s:(s+1280),:,:]
elif state.shape[0] < 1280:
    s1 = (1280 - state.shape[0]) // 2
    s2 = (1280 - state.shape[0]) - s1
    pad1 = 255 * np.ones((s1, 720, 3), dtype=np.uint8)
    pad2 = 255 * np.ones((s2, 720, 3), dtype=np.uint8)
    state = np.concatenate((pad1, state, pad2), 0)

player = cv2.imread('resource/player.png', 0)
gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
pos = play.multi_scale_search(player, gray_state, 0.3, 10)
print('multi scale search. res = {}'.format(pos))
player_pos = int((pos[0] + 13 * pos[2])/14.), (pos[1] + pos[3])//2

def get_target_position_fast(state, player_pos):
    if True:
        state_cut = state[:player_pos[0],:,:]
        m1 = (state_cut[:, :, 0] >= 240)
        m2 = (state_cut[:, :, 1] >= 240)
        m3 = (state_cut[:, :, 2] >= 240)
        m = np.uint8(np.float32(m1 * m2 * m3) * 255)
        b1, b2 = cv2.connectedComponents(m)
        print(b1, b2.shape, m.shape, np.max(b2))
        for i in range(1, np.max(b2) + 1):
            x, y = np.where(b2 == i)
            print(len(x), len(y))
            # print('fast', len(x))
            if len(x) > 280 and len(x) < 310:
                r_x, r_y = x, y
        h, w = int(r_x.mean()), int(r_y.mean())
        return np.array([h, w])

# print(get_target_position_fast(state, pos))
current_state = state
cv2.circle(current_state, (player_pos[1], player_pos[0]), 10, (0,255,0), -1)
cv2.imwrite('draw_state.png', current_state)
