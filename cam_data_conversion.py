import numpy as np
import matplotlib.pyplot as plt
import cv2

###############################
##### DOCUMENT EVERYTHING #####
###############################
# You aren't bound to any of these function headers, I just figured these would
# be the most natural inputs to manipulate.

def reduce_angle(input_image, initial_angle, final_angle=60):
	"""
	Inputs:
		input_image		:	### YOUR FORMAT CHOICE ###. Image to be manipulated.
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
	Returns:
		### YOUR FORMAT CHOICE ###. input_image, modified to have a viewing 
		angle of final_angle. If final_angle > initial_angle, don't modify.

	From Lydia:
		For image cropping, I recommend keeping the center of the output image
		the same as the input image.
	"""
	### YOUR CODE HERE ###
	return 

def rgb2grey(input_image):
	"""
	Inputs:
		input_image	:	### YOUR FORMAT CHOICE ###. RGB image to be manipulated.
	Returns:
		### YOUR FORMAT CHOICE ###. Greyscaled image of rgb_image.

	From Lydia:
		You don't need to reinvent the wheel for this one, it's been implemented
		at least a dozen times in a bunch of different libraries.
	"""
	### YOUR CODE HERE ###
	return

def reduce_bits(input_image, final_bits=8):
	"""
	Inputs:
		input_image	:	### YOUR FORMAT CHOICE ###. Image to be manipjulated.
		final_bits	:	Integer. The number of bits used to define an individual
						pixel in the output image, i.e. 2 bits => 2^2 = 4
						possible intensity values for any pixel.
	Returns:
		### YOUR FORMAT CHOICE ###. input_image, modified where each pixel
		is represented by final_bits number of bits. Assume an unchanged
		dynamic range.
	"""
	### YOUR CODE HERE ###
	return

def reduce_resolution(input_image, final_rows=224, final_cols=224):
	"""
	Inputs:
		input_image	:	### YOUR FORMAT CHOICE ###. Image to be manipulated.
		final_rows	:	Integer. The number of rows in the output image.
		final_cols	:	Integer. The number of columns in the output image.

	Returns:
		### YOUR FORMAT CHOICE ###. input_image, downsampled to have final_rows
		rows and final_cols columns. If final_rows or final_cols are larger than
		the number of rows or columns in input_image, respectively, pad with
		zeros.
	"""
	### YOUR CODE HERE ###
	return

def conv2research(input_image, initial_angle, final_angle,
					final_bits, final_rows, final_cols):
	"""
	Inputs:
		input_image		:	Input from RPi. Image to be manipulated.
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
		final_bits		:	Integer. The number of bits used to define an
							individual pixel in the output image.
		final_rows		:	Integer. The number of rows in the output image.
		final_cols		:	Integer. The number of columns in the output image.

	Returns:
		### YOUR FORMAT CHOICE ###. input_image, modified to have 
		a new viewing angle of final_angle, final_bits to represent each pixel
		intensity, and final_rows/cols for the new dimensions of the image.
	"""
	### YOUR CODE HERE ###
	return

def rpi2research(initial_angle, final_angle,
				final_bits, final_rows, final_cols):
	"""
	Inputs:
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
		final_bits		:	Integer. The number of bits used to define an
							individual pixel in the output image.
		final_rows		:	Integer. The number of rows in the output image.
		final_cols		:	Integer. The number of columns in the output image.

	Returns:
		None.
	"""
	# Set up the connection to the Raspberry Pi
	### YOUR CODE HERE ###

	# Display the pretty input from the Raspberry Pi's camera
	### YOUR CODE HERE ###

	# Display the converted video. The conversion should be done in real-time.
	# Save the converted frames as you go if you'd like.
	### YOUR CODE HERE ###

	pass