#!/usr/bin/env python3

""" Rasterization

Usage:
    ./rasterization.py [shape]

Args:
    shape: json file containing grafics description
"""

import sys

import json
import numpy as np
# import scipy as sp
# from google.colab.patches import cv2_imshow
# from google.colab import files
# from matplotlib import patheffects
# import matplotlib.pyplot as plt

def inside(x, y, primitive):
    """
    Check if point (x,y) is inside the primitive

    Args:
        x (float): horizontal point position
        y (float): vertical point position
    Returns:
        True if (x,y) is inside the primitive, False case contrary
    """

    # You should implement your inside test here for all shapes
    # for now, it only returns a false test

    return False

class Screen:
    """ Creates a virtual basic screen

    Args:
        gdata (dict): dictionary containing screen size and scene description
    """

    def __init__(self, gdata):
        self._width = gdata.get("width")
        self._height = gdata.get("height")
        self._scene = self.preprocess(gdata.get("scene"))
        self.create_image()


    def preprocess(self, scene):
        """ ?????????????

        Args:
            scene (dict): Scene containing the graphic primitives

        Returns:
            scene (dict): Scene containing the graphic primitives with additional info
        """

        # Possible preprocessing with scene primitives, for now we don't change anything
        # You may define bounding boxes, convert shapes, etc
        preprop_scene = []

        for primitive in scene:
            # do some processing
            # for now, only copies each primitive to a new list

            preprop_scene.append(primitive)

        return preprop_scene

    def create_image(self):
        """ Creates image with white background

        Returns
            image (numpy array): White image with R, G, B channels
        """

        self._image = 255 *  np.ones((self._height, self._width, 3), np.uint8)

        def rasterize(self):
            """ Rasterize the primitives along the Screen
            """

            for primitive in self._scene:
                # Loop through all pixels
                # You MUST use bounding boxes in order to speed up this loop
                for w in range(0, self._width):
                    x = w + 0.5
                    for h in range(0, self._height):
                        y = h + 0.5
                        # First, we check if the pixel center is inside the primitive
                        if (inside(x, y, primitive)):
                            im_x, im_y = w, self._height - (h + 1)
                            self._image[im_y, im_x] = primitive["color"][::-1]


    def show(self, exec_rasterize = False):
        """ Show the virtual Screen
        """

        if (exec_rasterize):
            self.rasterize()

        cv2_imshow(self._image)

def main():
    try:
        with open(sys.argv[1]) as json_f:
            graphic_data = json.load(json_f)

        # screen = Screen(graphic_data)
        # screen.show(True)

    except:
        print(__doc__)

if __name__ == "__main__":
    main()
