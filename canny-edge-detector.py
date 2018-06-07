#!/usr/bin/python -O
# -*- coding: utf-8 -*-

import string, sys
import os.path
import numpy
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
import cv2
from cv2 import imread, imwrite

class CannyEdgeDetector:
	def __init__(self):

		return

	#### Apply Gaussian filter to smooth the image in order to remove the noise

	# the gaussian blur is used to reduce noise.
	# the inputs to the gaussian blur are the image and the standard deviation, σ (sigma), of the curve.
	# A nd gaussian (2d) can be obtained by simply performing a 1d gaussian on each axis

	# The shape of a gaussian curve (or a normal distribution) is primarily defined by the standard deviation, σ (sigma).  a small sigma gives a curve that is tall and thin, and large sigma gives a curve that is short and fat.
	# The 1d gaussian distribution is described by: (1/(sqrt(2pi)σ))exp(-x²/(2σ²))
	# A gaussian kernel is the discrete matrix repersenting the gaussian distribution used in convolution with the image.
	# In theory, the gaussian distribution is non-zero everywhere, which would require an infinitely large convolution kernel, but in practice it is effectively zero more than about three standard deviations from the mean, and so we can truncate the kernel at this point.
	# The discrete kernel for the gaussian used in the convolution is sized based on on a few factors including standard deviation.  (say a 5x5 matrix or 10x10 matrix)
	# It is not obvious how to pick the values of the matrix to approximate a Gaussian.  One could use the value of the Gaussian at the centre of a pixel in the mask, but the Gaussian varies non-linearly across the pixel and the center would not capture that.  Using a little calculus for each pixel, we can easily fix this.
	def gaussian(self, image, sigma):
		return gaussian_filter(image, sigma)

	#### Find the intensity gradients of the image

	# Going to use a sobel filter for gradient calculations.
	# The soble operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical
	def gradients(self, image):
		sobel_kernel_x = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], numpy.int32)
		sobel_kernel_y = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], numpy.int32)

		g_x = convolve(image, sobel_kernel_x)
		g_y = convolve(image, sobel_kernel_y)

		# Equivalent tp sqrt(x² + y²), element-wise
		g = numpy.hypot(g_x, g_y)

		# get the direction of g
		theta = numpy.arctan2(g_y, g_x)

		return g, theta

	# Apply non-maximum suppression to get rid of spurious response to edge detection
	# Non-maximum suppression is applied to "thin" the edge.  We desire to have one accurate response to the edge. Thus non-maximum suppression can help to suppress all the gradient values (by setting them to 0) except the local maxima, which indicate locations with the sharpest change of intensity value. The algorithm for each pixel in the gradient image is:
	# 1) Compare the edge strength of the current pixel with the edge strength of the pixel in the positive and negative gradient directions.
	# 2) If the edge strength of the current pixel is the largest compared to the other pixels in the mask with the same direction (i.e., the pixel that is pointing in the y-direction, it will be compared to the pixel above and below it in the vertical axis), the value will be preserved. Otherwise, the value will be suppressed.
	def suppression(self, image, theta):
		y_length, x_length = image.shape

		s = numpy.zeros((y_length, x_length), dtype=numpy.int32)

		bound_index = lambda i, max_i: max(min(i, max_i), 0)

		for y in range(y_length):
			for x in range(x_length):
				# get the angle between 0 and pi
				pixel_index = int(math.floor((theta[y, x]%math.pi)/(math.pi/4.0)))

				x_index = 1 # if(pixel_index == 0)
				y_index = 0 # if(pixel_index == 0)
				if pixel_index == 1:
					x_index = 1
					y_index = 1
				elif pixel_index == 2:
					x_index = 0
					y_index = 1
				elif pixel_index > 2: # also account for case when theta is pi
					x_index = -1
					y_index = 1

				image_yx = image[y, x]
				# check for local maximum
				if (image_yx >= image[bound_index(y + y_index, y_length-1), bound_index(x + x_index, x_length-1)] and image_yx >= image[bound_index(y - y_index, y_length-1), bound_index(x - x_index, x_length-1)]):
					s[y, x] = image_yx
		return s


	# Some edge pixels remain that are caused by noise and color variation. In order to account for these spurious responses, it is essential to filter out edge pixels with a weak gradient value and preserve edge pixels with a high gradient value.
	# Apply double threshold to determine potential edges
	def threshold(self, image, lower_t, upper_t):
		none_x, none_y = numpy.where(image < lower_t)
		weak_x, weak_y = numpy.where((image >= lower_t) & (image <= upper_t))
		strong_x, strong_y = numpy.where(image > upper_t)

		image[none_x, none_y] = numpy.int32(0)
		image[weak_x, weak_y] = numpy.int32(50)
		image[strong_x, strong_y] = numpy.int32(255)
		return image

	# Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
	def hysteresis(self, image):
		bound_index = lambda i, max_i: max(min(i, max_i), 0)

		y_length, x_length = image.shape
		for y in range(y_length):
			for x in range(x_length):
				if image[y, x] == numpy.int32(50):
					for y_t in [-1, 0, 1]:
						for x_t in [-1, 0, 1]:
							if (image[bound_index(y + y_t, y_length-1), bound_index(x + x_t, x_length-1)] == numpy.int32(255)):
								image[y, x] = numpy.int32(255)
								break
					if image[y, x] == numpy.int32(50):
						image[y, x] = numpy.int32(0)
		return image

	def execute(self, image, sigma, lower_t, upper_t):
		image = self.gaussian(image, sigma)
		image, theta = self.gradients(image)
		image = self.suppression(image, theta)
		image = self.threshold(image, lower_t, upper_t)
		image = self.hysteresis(image)

		return image

# main (DRIVER)
def main():
	if len(sys.argv) != 2:
		print "Wrong number of arguments."
		print "Usage: " + sys.argv[0] + " <input.jpg>\n"
		return 1
	else:
		inputFileName = sys.argv[1]
		if not os.path.isfile(inputFileName):
			print "The file \""+ inputFileName + "\" does not exist.\n"
			return 2
		if not inputFileName.endswith(".jpg"):
			print "This is not an .jpg file.\n"
			return 3
		outputFileName = "output.jpg"

		image = imread(inputFileName, cv2.IMREAD_GRAYSCALE).astype("int32")
		# TODO: could these be passed-in or progrmaticly determined?
		sigma = 1.4
		lower_t = 20
		upper_t = 40

		image = CannyEdgeDetector().execute(image, sigma, lower_t, upper_t)
		imwrite(outputFileName, image)
	return 0

# call to main
if __name__ == "__main__":
	main()
