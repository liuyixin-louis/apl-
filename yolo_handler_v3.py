# call yolo code on our own data
# my data will come in list of images
# i want to get measurements of both time and accuracy while using yolo v2
from data_handler import get_data_from_list


#@profile
def run_yolo(num_crops_per_frames, crop_per_frames, fixbb_crop, INPUT_FRAMES, frame_files, resize_frames=None,
             model_h5='v3.h5', anchors_txt='yolo_anchors.txt', allowed_number_of_boxes=100, VERBOSE=1):

    """
    print("num_crops_per_frames", num_crops_per_frames)
    print("crop_per_frames", crop_per_frames)
    print("fixbb_crop", fixbb_crop)
    print("INPUT_FRAMES", INPUT_FRAMES)
    print("frame_files", frame_files)
    """

    path_to_yolo = "/home/dl/lyx/myapl/kyolov3/ky/"

    # path_to_yolo = use_path_which_exists(yolo_paths)
    #
    # print (path_to_yolo)

    import site
    site.addsitedir(path_to_yolo)
    #print (sys.path)  # Just verify it is there

    from kyolov3 import eval_yolo_direct_images

    ################################################################
    num_frames = len(num_crops_per_frames)
    image_names, ground_truths, frame_ids, crop_ids = get_data_from_list(crop_per_frames)


    args = {}

    args["anchors_path"]=path_to_yolo+'model_data/' + anchors_txt
    args["classes_path"]=path_to_yolo+'model_data/coco_classes.txt'
    args["model_path"]=path_to_yolo+'model_data/' + model_h5
    args["score_threshold"]=0.3
    args["iou_threshold"]=0.5
    args["output_path"]=''
    args["test_path"]=''

    full_path_frame_files = [INPUT_FRAMES + s for s in frame_files]
    bboxes = eval_yolo_direct_images._main(args, frames_paths=full_path_frame_files, crops_bboxes=crop_per_frames, crop_value=fixbb_crop, resize_frames=resize_frames, verbose=VERBOSE, person_only=True, allowed_number_of_boxes=allowed_number_of_boxes)




    bboxes_per_frames = []
    for i in range(0,num_frames):
        bboxes_per_frames.append([])

    for index in range(0,len(image_names)):
        frame_index = frame_ids[index] - frame_ids[0]
        crop_index = crop_ids[index]

        #if len(bboxes_per_frames) < frame_index+1:
        #    bboxes_per_frames.append([])
        if bboxes_per_frames[frame_index] is None:
            bboxes_per_frames[frame_index] = []

        crops_in_frame = crop_per_frames[frame_index]
        current_crop = crops_in_frame[crop_index]
        a_left = current_crop[1][0]
        a_top = current_crop[1][1]

        if len(bboxes[index]) > 0: #not empty
            fixed_bboxes = []
            for bbox in bboxes[index]:
                bbox_array = bbox[1]
                fix_array = bbox_array + [a_top, a_left, a_top, a_left]
                bboxes_per_frames[frame_index].append([bbox[0],fix_array,bbox[2],bbox[3]])

            bboxes_per_frames[frame_index] += fixed_bboxes

    return bboxes_per_frames
