#!/usr/bin/python3

import colorsys
import cv2
import numpy as np

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])



def rgb_to_hsl(r, g, b):
    """
    Converte RGB (0-255) para HSL.
    Retorna: H (0-360), S (0-1), L (0-1)
    """
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    h, l, s = colorsys.rgb_to_hls(r_, g_, b_)  # note: HLS no colorsys
    return h * 360, s, l


def hsl_to_rgb(h, s, l):
    """
    Converte HSL para RGB (0-255).
    Entrada: H (0-360), S (0-1), L (0-1)
    Retorna: r, g, b (0-255)
    """
    h_ = h / 360.0
    r_, g_, b_ = colorsys.hls_to_rgb(h_, l, s)  # note: HLS no colorsys
    return int(round(r_ * 255)), int(round(g_ * 255)), int(round(b_ * 255))


def rgb_to_lab(r, g, b):
    """
    Converte uma cor de RGB para CIELAB.
    Entrada: r, g, b (0-255)
    Saída: L, a, b (valores Lab)
    """
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)  # precisa de forma (1,1,3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0, 0, 0], lab[0, 0, 1], lab[0, 0, 2]


def lab_to_rgb(L, a, b):
    """
    Converte uma cor de CIELAB para RGB.
    Entrada: L, a, b (valores Lab como retornados pela função anterior)
    Saída: r, g, b (0-255)
    """
    lab = np.array([[[L, a, b]]], dtype=np.uint8)  # precisa de forma (1,1,3)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb[0, 0, 0], rgb[0, 0, 1], rgb[0, 0, 2]


