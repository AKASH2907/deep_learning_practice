# # summarize filters in each convolutional layer
# from keras.applications.vgg16 import VGG16
# from matplotlib import pyplot
# # load the model
# model = VGG16(include_top=False)
# print(model.summary())
# # summarize filter shapes
# i = 0

# for i in range(len(model.layers)):
# 	layer = model.layers[i]
# 	# check for convolutional layer
# 	if 'conv' not in layer.name:
# 		continue
	
# 	filters, biases = layer.get_weights()
# 	print(i, layer.name, filters.shape)

# We can visualize the layrs which are conv in nature
# That are 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17 in case of VGG-16 

# 1 block1_conv1 (3, 3, 3, 64)
# 2 block1_conv2 (3, 3, 64, 64)
# 4 block2_conv1 (3, 3, 64, 128)
# 5 block2_conv2 (3, 3, 128, 128)
# 7 block3_conv1 (3, 3, 128, 256)
# 8 block3_conv2 (3, 3, 256, 256)
# 9 block3_conv3 (3, 3, 256, 256)
# 11 block4_conv1 (3, 3, 256, 512)
# 12 block4_conv2 (3, 3, 512, 512)
# 13 block4_conv3 (3, 3, 512, 512)
# 15 block5_conv1 (3, 3, 512, 512)
# 16 block5_conv2 (3, 3, 512, 512)
# 17 block5_conv3 (3, 3, 512, 512)



# Visualizing Filters

# cannot easily visualize filters lower down
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
# load the model
model = VGG16()
# retrieve weights from the fifth hidden layer
filters, biases = model.layers[5].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
# Create a new figure to save the Vizualization
pyplot.figure()

for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# save the figure
pyplot.savefig('wts.png')


# show the figure
# pyplot.show()


