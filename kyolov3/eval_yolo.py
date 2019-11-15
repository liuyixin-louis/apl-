#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from kyolov3.ky.yolo3.model import yolo_eval, yolo_head

from timeit import default_timer as timer


def _main(args, input_paths, ground_truths, output_paths, num_frames, num_crops_per_frames, save_annotated_images=False, verbose=1, person_only=True):
    model_path = os.path.expanduser(args["model_path"])
    print(model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args["anchors_path"])
    classes_path = os.path.expanduser(args["classes_path"])
    #test_path = os.path.expanduser(args.test_path)
    #output_path = os.path.expanduser(args.output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    if verbose > 0:
        print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))


    ####### EVALUATION

    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args["score_threshold"],
        iou_threshold=args["iou_threshold"],
        max_boxes=50)

    evaluation_times = []
    additional_times = []
    bboxes = []

    frame_number = 0
    print("Frame", frame_number, " with ", num_crops_per_frames[frame_number], " crops.")
    images_processed = 0
    for image_i in range(0,len(input_paths)):
        if images_processed >= num_crops_per_frames[frame_number]:
            images_processed = 0
            frame_number += 1
            print("Frame", frame_number, " with ", num_crops_per_frames[frame_number], " crops.")

        images_processed += 1


        start_loop = timer()

        image_file = input_paths[image_i]
        output_file = output_paths[image_i]

        try:
            image_type = imghdr.what(image_file)
            if not image_type:
                continue
        except IsADirectoryError:
            continue

        image = Image.open(image_file)

        """
        image_data = np.array(image, dtype='float32')
        """
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)


        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        if images_processed < 2:
            print("# image size: ",image_data.shape, image.size)

        ################# START #################
        start_eval = timer()
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        end_eval = timer()
        ################# END #################
        evaluation_time = (end_eval - start_eval)
        evaluation_times.append(evaluation_time)

        people = 0
        bboxes_image = []
        #print(num_frames, num_crops)
        for i, c in reversed(list(enumerate(out_classes))):

            predicted_class = class_names[c]

            if predicted_class == 'person':
                people += 1
            if person_only and (predicted_class != 'person'):
                continue

            box = out_boxes[i]
            score = out_scores[i]

            #print(predicted_class, box, score)

            bboxes_image.append([predicted_class, box, score, c])

            if save_annotated_images:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                # print (dir_path + '/' + 'font/FiraMono-Medium.otf')

                font = ImageFont.truetype(
                    font=(dir_path + '/' + 'font/FiraMono-Medium.otf'),
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300

                label = '{} {:.2f}'.format(predicted_class, score)

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                if verbose > 1:
                    print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[c])
                draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        if verbose > 0:
            num = len(out_boxes)
            if person_only:
                num = people
            print('Found {} boxes for {} in {}s'.format(num, image_file[-13:], evaluation_time))


        bboxes.append(bboxes_image)
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        if save_annotated_images:
            image.save(output_file, quality=90)

        end_loop = timer()
        loop_time = (end_loop - start_loop)
        additional_times.append(loop_time - evaluation_time)

    #sess.close()

    return evaluation_times, additional_times, bboxes

#if __name__ == '__main__':
#    print(parser.parse_args())
#    _main(parser.parse_args())
