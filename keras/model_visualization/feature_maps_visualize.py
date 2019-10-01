# Visualizing Activation Maps

#  Firstly, look for the conv feature maps available in your model
# summarize feature map size for each conv layer
# from keras.applications.vgg16 import VGG16
# from matplotlib import pyplot
# model = VGG16()
# # summarize feature map shapes
# for i in range(len(model.layers)):
# 	layer = model.layers[i]
# 	# check for convolutional layer
# 	if 'conv' not in layer.name:
# 		continue
# 	# summarize output shape
# 	print(i, layer.name, layer.output.shape)

# 1 block1_conv1 (?, 224, 224, 64)
# 2 block1_conv2 (?, 224, 224, 64)  --> Block 1
# 4 block2_conv1 (?, 112, 112, 128)
# 5 block2_conv2 (?, 112, 112, 128) --> Block 2 
# 7 block3_conv1 (?, 56, 56, 256)
# 8 block3_conv2 (?, 56, 56, 256)
# 9 block3_conv3 (?, 56, 56, 256)   --> Block 3
# 11 block4_conv1 (?, 28, 28, 512)
# 12 block4_conv2 (?, 28, 28, 512)
# 13 block4_conv3 (?, 28, 28, 512)  --> Block 4
# 15 block5_conv1 (?, 14, 14, 512)
# 16 block5_conv2 (?, 14, 14, 512)
# 17 block5_conv3 (?, 14, 14, 512)  --> Block 5


# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
# Can load any imagent pre-trained architecture
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
# load the model
model = VGG16()
# redefine model to output right after the first hidden layer
# These are specifically for VGG 16 model
# You have to search these particularly for each from the piece of code above
ixs = [2, 5, 9, 13]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# print(len(feature_maps))
# plot the output from each block
square = 4
c=0
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	pyplot.figure()
	print(c)
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	
	pyplot.savefig('layer_' + str(ixs[c]) + '.png')
	pyplot.close()
	
	c+=1
	# show the figure
	# pyplot.show()

# plot feature map of first conv layer for given image
# # redefine model to output right after the first hidden layer
# model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# model.summary()
# # load the image with the required shape
# img = load_img('18.jpeg', target_size=(224, 224))
# # convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# # get feature map for first hidden layer
# feature_maps = model.predict(img)
# # plot all 64 maps in an 8x8 squares
# square = 5
# ix = 1
# for _ in range(square):
# 	for _ in range(square):
# 		# specify subplot and turn of axis
# 		ax = pyplot.subplot(square, square, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
# 		ix += 1
# # show the figure
# pyplot.show()	


