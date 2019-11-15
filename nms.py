# import the necessary packages
import numpy as np

#@profile
def non_max_suppression_tf(session, boxes, scores, max_boxes, iou_threshold):
	print("<Tensorflow for NMS>")
	import tensorflow as tf
	from keras import backend as K

	max_boxes_tensor = K.variable(max_boxes, dtype='int32')
	session.run(tf.variables_initializer([max_boxes_tensor]))
	# print(boxes)
	#修改
	scores = np.array(scores, dtype=np.float32)
	nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
	print(boxes.tolist())
	print(scores)
	print(max_boxes_tensor)
	print(iou_threshold)
	# import tensorflow as tf
	# import numpy as np
	# from keras import backend as K
	# boxlist =boxes
	# boxes = np.array(boxlist, dtype=np.float32)
	# scorelist =scores
	# scores = np.array(scorelist, dtype=np.float32)
	# max_boxes_tensor = K.variable(500, dtype='int32')
	#
	# with tf.Session() as sess:
	# 	selected_indices = sess.run(
	# 		tf.image.non_max_suppression(boxes=boxes, scores=scores, iou_threshold=0.5, max_output_size=500))
	# 	print(selected_indices)
	# 	selected_boxes = sess.run(K.gather(boxes, selected_indices))
	# 	print(selected_boxes)

	boxes = K.gather(boxes, nms_index)
	scores = K.gather(scores, nms_index)
	# print(boxes)
	# print(scores)



	with session.as_default():
		boxes_out = boxes.eval()
		scores_out = scores.eval()

	return boxes_out, scores_out


# Malisiewicz et al.
# see https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



# Alternatively look at:
# https://github.com/bharatsingh430/soft-nms/tree/master/lib/nms
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):
	"""Pure Python NMS baseline."""
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)

		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep