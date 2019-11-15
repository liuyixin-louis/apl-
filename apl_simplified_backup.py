# 调用包
import tensorflow as tf
import os, fnmatch
from shutil import copyfile
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from crop_functions import crop_from_one_frame, mask_from_one_frame, crop_from_one_frame_WITH_MASK_in_mem, \
    get_number_of_crops_from_frame
from yolo_handler import run_yolo
from mark_frame_with_bbox import annotate_image_with_bounding_boxes, bboxes_to_mask, \
    annotate_prepare
from nms import non_max_suppression_tf
from bbox_postprocessing import postprocess_bboxes_by_splitlines
from data_handler import is_non_zero_file
from config import *

frame_files = []




def main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS):
    # 目录路径
    video_file_root_folder = str(Path(INPUT_FRAMES).parents[0])
    output_frames_folder = video_file_root_folder + "/output/" + RUN_NAME + "/frames/"

    # 形成路径数组
    folderlist = [output_frames_folder]

    # 分别建目录
    for folder in folderlist:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 是否开启注意力评估阶段
    attention_model = SETTINGS["attention"]
    # 该参数与相邻帧的处理好像有关
    attention_spread_frames = SETTINGS["att_frame_spread"]

    # 读每个ｆｒａｍｅ
    files = sorted(os.listdir(INPUT_FRAMES))
    # print("files",len(files), files[0:10])
    files = [path for path in files if is_non_zero_file(INPUT_FRAMES + path)]
    # print("files", len(files), files[0:10])
    frame_files = fnmatch.filter(files, '*.jpg')
    print("jpgs:", frame_files[0:2], "...")

    # 本轮测试所用ｆｒａｍｅ范围
    start_frame = SETTINGS["startframe"]
    end_frame = SETTINGS["endframe"]

    if end_frame is not -1:
        frame_files = frame_files[start_frame:end_frame]
    else:
        frame_files = frame_files[start_frame:]

    # 一帧中允许的检测出的最多的ｂｏｘ数目
    allowed_number_of_boxes = SETTINGS["allowed_number_of_boxes"]

    # 注意力评估阶段：放缩粗检测，为最终评估提供Ｍａｓｋ，用于激活相应的ｃｒｏｐｓ。

    print("################## Mask generation ##############")
    crop_per_frames = []
    crop_number_per_frames = []
    if attention_model:
        print("##", len(frame_files), "of frames")

        # 生成粗检测所用的分割ｃｒｏｐｓ
        # 根据 split、输入图像分辨率运算，切割成若干块，之后把每块缩放成608*608再yolo识别
        # 1 generate crops from full images
        mask_crops_per_frames = []
        scales_per_frames = []
        mask_crops_number_per_frames = []

        for frame_i in range(0, len(frame_files)):
            frame_path = INPUT_FRAMES + frame_files[frame_i]
            # working with many large files - relatively slow
            mask_crops, scale_full_img, attention_crop_TMP_SIZE_FOR_MODEL = mask_from_one_frame(frame_path,
                                                                                                SETTINGS
                                                                                                )  ### <<< mask_crops
            mask_crops_per_frames.append(mask_crops)
            mask_crops_number_per_frames.append(len(mask_crops))
            scales_per_frames.append(scale_full_img)

        # 2 eval these calculate
        # 粗检测
        bboxes_per_frames = run_yolo(mask_crops_number_per_frames,
                                     mask_crops_per_frames,
                                     attention_crop_TMP_SIZE_FOR_MODEL,
                                     INPUT_FRAMES, frame_files,
                                     resize_frames=scales_per_frames,
                                     allowed_number_of_boxes=allowed_number_of_boxes,
                                     VERBOSE=0
                                     )
        # 根据粗检测结果激活下一轮切割对应ｃｒｏｐｓ
    print("################## Cropping frames : extracting crops from images ##################")
    print("##", len(frame_files), "of frames")
    summed_croping_time = []
    save_one_crop_vis = True
    crop_per_frames = []
    crop_number_per_frames = []
    for frame_i in range(0, len(frame_files)):
        start = timer()

        frame_path = INPUT_FRAMES + frame_files[frame_i]

        if attention_model:

            if attention_spread_frames == 0:
                bboxes = bboxes_per_frames[frame_i]
                # print(len(bboxes), bboxes)

            else:
                from_frame = max([frame_i - attention_spread_frames, 0])
                to_frame = min([frame_i + attention_spread_frames, len(frame_files)]) + 1

                bboxes = [item for sublist in bboxes_per_frames[from_frame:to_frame] for item in sublist]
                # print(from_frame,"to",to_frame-1,len(bboxes), bboxes)

            scale = scales_per_frames[frame_i]

            img = Image.open(frame_path)
            mask = bboxes_to_mask(bboxes, img.size, scale, SETTINGS["extend_mask_by"])

            mask_over = 0.1  # SETTINGS["over"]
            horizontal_splits = SETTINGS["horizontal_splits"]
            overlap_px = SETTINGS["overlap_px"]
            crops, crop_TMP = crop_from_one_frame_WITH_MASK_in_mem(img, mask, frame_path,
                                                                   horizontal_splits, overlap_px, mask_over,
                                                                   show=False, save_crops=False,
                                                                   save_visualization=save_one_crop_vis,
                                                                   )

        else:
            horizontal_splits = SETTINGS["horizontal_splits"]
            overlap_px = SETTINGS["overlap_px"]

            crops, crop_TMP = crop_from_one_frame(frame_path, horizontal_splits, overlap_px,
                                                  show=False, save_visualization=save_one_crop_vis,
                                                  save_crops=False)

        crop_per_frames.append(crops)
        crop_number_per_frames.append(len(crops))
        save_one_crop_vis = False

        end = timer()
        time = (end - start)
        summed_croping_time.append(time)

    crop_TMP_SIZE_FOR_MODEL = crop_TMP

    # Run YOLO on crops
    print("")
    print("################## Running Model ##################")
    bboxes_per_frames = run_yolo(crop_number_per_frames, crop_per_frames,
                                 crop_TMP_SIZE_FOR_MODEL, INPUT_FRAMES,
                                 frame_files,
                                 anchors_txt=SETTINGS["anchorfile"],
                                 allowed_number_of_boxes=allowed_number_of_boxes)

    iou_threshold = 0.5  # towards 0.01 its more drastic and deletes more bboxes which are overlapped
    limit_prob_lowest = 0  # 0.70 # inside we limited for 0.3

    sess = tf.Session()
    colors = annotate_prepare()

    #
    for frame_i in range(0, len(frame_files)):
        test_bboxes = bboxes_per_frames[frame_i]
        from_number = len(test_bboxes)

        arrays = []
        scores = []
        for j in range(0, len(test_bboxes)):
            if test_bboxes[j][0] == 'person':
                score = test_bboxes[j][2]
                if score > limit_prob_lowest:
                    arrays.append(list(test_bboxes[j][1]))
                    scores.append(score)
        print(arrays)
        arrays = np.array(arrays)

        if len(arrays) == 0:
            # no bboxes found in there, still we should copy the frame img
            copyfile(INPUT_FRAMES + frame_files[frame_i], output_frames_folder + frame_files[frame_i])
            continue

        person_id = 0
        # test
        DEBUG_TURN_OFF_NMS = False
        if not DEBUG_TURN_OFF_NMS:

            """
            nms_arrays = py_cpu_nms(arrays, iou_threshold)
            reduced_bboxes_1 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],0.0,person_id]
                reduced_bboxes_1.append(a)
            """

            nms_arrays, scores = non_max_suppression_tf(sess, arrays, scores, allowed_number_of_boxes,
                                                        iou_threshold)
            reduced_bboxes_2 = []

            for j in range(0, len(nms_arrays)):
                a = ['person', nms_arrays[j], scores[j], person_id]
                reduced_bboxes_2.append(a)

            test_bboxes = reduced_bboxes_2

        print("in frame", frame_i, "reduced from", from_number, "to", len(test_bboxes), "bounding boxes with NMS.")

        if SETTINGS["postprocess_merge_splitline_bboxes"]:
            replace_test_bboxes = postprocess_bboxes_by_splitlines(crop_per_frames[frame_i], test_bboxes,
                                                                   overlap_px_h=SETTINGS["overlap_px"],
                                                                   DEBUG_POSTPROCESS_COLOR=SETTINGS[
                                                                       "debug_color_postprocessed_bboxes"])
            # test_bboxes += replace_test_bboxes
            test_bboxes = replace_test_bboxes

        annotate_image_with_bounding_boxes(INPUT_FRAMES + frame_files[frame_i],
                                           output_frames_folder + frame_files[frame_i], test_bboxes, colors,
                                           draw_text=False, save=True, show=False,
                                           thickness=SETTINGS["thickness"])
    sess.close()


if __name__ == '__main__':
    # start = timer()
    main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)
    # end = timer()
    # time = (end - start)
    # print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")
