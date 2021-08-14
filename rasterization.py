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
from PIL import Image

def inside(x, y, primitive):
    """
    Check if point (x,y) is inside the primitive

    Args:
        x (float): horizontal point position
        y (float): vertical point position
        primitive (dict): primitive shape
    Returns:
        True if (x,y) is inside the primitive, False case contrary
    """

    # You should implement your inside test here for all shapes
    # for now, it only returns a false test

    if primitive["shape"] == "circle":
        dist_sqr = ((primitive["center"][0] - x) ** 2 +
                    (primitive["center"][1] - y) ** 2)

        return dist_sqr <= primitive["radius"] ** 2
    else:


    return False

def bounding_box(primitive):
    """ Creates a bounding box for a given circle ou convex polygon

    Args:
        primitive (dict): primitive shape
    Returns:
        [[x1, y1], [x2, y2]] corresponding do the bounding box, where

        (x1, y1) X-----+
                 |     |
                 +-----X (x2, y2)
    """

    if primitive["shape"] == "circle":
        bbox = [[primitive["center"][0] - primitive["radius"],
                 primitive["center"][1] - primitive["radius"]],
                [primitive["center"][0] + primitive["radius"],
                 primitive["center"][1] + primitive["radius"]]]
    else:
        x_coords, y_coords = zip(*primitive["vertices"])
        bbox = [[min(x_coords), min(y_coords)],
                [max(x_coords), max(y_coords)]]

    primitive["bounding_box"] = bbox
    return primitive


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
            preprop_scene.append(bounding_box(primitive))

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
            print(primitive)
            bbox = primitive["bounding_box"]
            print(bbox)
            # Loop through all pixels
            # You MUST use bounding boxes in order to speed up this loop
            for w in range(bbox[0][0], bbox[1][0]):
                x = w + 0.5
                for h in range(bbox[0][1], bbox[1][1]):
                    y = h + 0.5
                    # First, we check if the pixel center is inside the primitive
                    im_x, im_y = w, self._height - (h + 1)
                    if inside(x, y, primitive):
                        self._image[im_y, im_x] = primitive["color"][::-1]
                    else:
                        self._image[im_y, im_x] = [100, 0, 0]


    def show(self, exec_rasterize = False):
        """ Show the virtual Screen
        """

        if (exec_rasterize):
            self.rasterize()

        Image.fromarray(self._image).show()

def main():
    with open(sys.argv[1]) as json_f:
        graphic_data = json.load(json_f)

    screen = Screen(graphic_data)
    screen.show(True)

if __name__ == "__main__":
    main()
