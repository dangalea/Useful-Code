import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os, glob
import multiprocessing as mp
import cv2
import keras.backend as K
import tensorflow.keras as keras
tf.compat.v1.enable_eager_execution()

def get_heatmap(model, layer_name, x):

	x = np.expand_dims(x, axis=0)
	conv_layer = model.get_layer(layer_name)
	heatmap_model = tf.keras.models.Model([model.inputs[0]], [conv_layer.output, model.output])

	with tf.GradientTape() as tape:
		conv_outputs, predictions = heatmap_model(x)
		loss = predictions[0]

	output = conv_outputs[0]
	grads = tape.gradient(loss, conv_outputs)[0]

	gate_f = tf.cast(output > 0, 'float32')
	gate_r = tf.cast(grads > 0, 'float32')
	guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

	weights = tf.reduce_mean(guided_grads, axis=(0, 1))

	cam = np.ones(output.shape[0: 2], dtype = np.float32)

	for i, w in enumerate(weights):
		cam += w * output[:, :, i]

	cam = cv2.resize(cam.numpy(), (43, 29))
	cam = np.maximum(cam, 0)
	heatmap = (cam - cam.min()) / (cam.max() - cam.min())

	return heatmap

def plot_heatmap(heatmap, x, pred, variables=None, name=None, save=True, show=False):

	plt.ioff()

	fig, axs = plt.subplots(x.shape[-1], 2, figsize=(15, 5*x.shape[-1]))
	axs = axs.ravel()

	for i in range(x.shape[-1]):

		arr_to_plt = x[:, :, i]
		im = axs[i*2].contourf(arr_to_plt)
		fig.colorbar(im, ax=axs[i*2])

		if variables != None:
			axs[i*2].set_title(variables[i])

		arr_to_plt = heatmap
		im = axs[i*2 + 1].contourf(arr_to_plt)
		fig.colorbar(im, ax=axs[i*2+1])
		plt_title = "Heatmap"
		axs[i*2+1].set_title(plt_title)

	if name == None:
		fig.suptitle(pred)
	else:
		fig.suptitle(name + " " + str(pred))
	if show == True:
		plt.show()
	if save == True:
		plt.savefig(name+".pdf")
