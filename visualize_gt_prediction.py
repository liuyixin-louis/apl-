from voc_eval import voc_eval, parse_rec, voc_eval_twoArrs
from mark_frame_with_bbox import annotate_image_with_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np

#gt_path_folder = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/RuzickaDataset/samples/_S1000040_5fps/"
#output_model_predictions_folder = '/home/ekmek/intership_project/video_parser_v1/_videos_to_test/RuzickaDataset/output/Annot_S1000040_5fps_Splits2to4/'

# EXAMPLE
#gt_path_folder = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/input/auto_annot/"
#output_model_predictions_folder = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/output_annotation_results/"

gt_path_folder = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PEViD_UHD_annot/Exchanging_bags_day_indoor_2/"
# v1 code
output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/output/Exchanging_bags_day_indoor_2_1to2/"
# v2 code
### NMS definitely helps .. output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/Exchanging_bags_day_indoor_2__1to2_beforeNMS/"
output_model_predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_full/Exchanging_bags_day_outdoor_6_1to3/"

#2to4_20px_subset 0.81
output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/cleanSet_Exchanging_bags_day_indoor_2_2to4/"
#1to3_50px_subset 0.95
output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/cleanSet_Exchanging_bags_day_indoor_2_1to3_over50/"
#1to3_50px_90perc subset 0.95
output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/cleanSet_Exchanging_bags_day_indoor_2_1to3_over50_90percovereat/"


output_model_predictions_folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/cleanSet_Exchanging_bags_day_indoor_2_1to3/"


file = "2to4_over20/My4KOver20__S1000040_5fps_2to4" # is worse!
gt = "_S1000040_5fps"
output_model_predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/"+file+"/"
gt_path_folder = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/annotations/samples/"+gt+"/"
globally_saved_annotations = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/annotations/samples/"


show_figures = True
draw_text = False # text with confidences

imagesetfile = output_model_predictions_folder+"annotnames.txt"
predictions_file  = output_model_predictions_folder+"annotbboxes.txt"
rec, prec, ap = voc_eval(predictions_file,gt_path_folder,imagesetfile,'person')

print("ap", ap)

with open(predictions_file, 'r') as f:
    lines = f.readlines()
predictions = [x.strip().split(" ") for x in lines]

predictions_dict = {}

for pred in predictions:
    score = float(pred[1])
    # <image identifier> <confidence> <left> <top> <right> <bottom>
    left   = int(pred[2])
    top    = int(pred[3])
    right  = int(pred[4])
    bottom = int(pred[5])
    arr = [score, left, top, right, bottom]
    if not pred[0] in predictions_dict:
        predictions_dict[pred[0]] = []
    predictions_dict[pred[0]].append(arr)

print("predictions",len(predictions_dict), predictions_dict)

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]

aps = []

for imagename in imagenames:
    img = gt_path_folder + imagename + ".jpg"
    gt_file = gt_path_folder + imagename + ".xml"

    gt = parse_rec(gt_file)
    if imagename not in predictions_dict:
        continue
    predictions = predictions_dict[imagename]

    print(imagename)
    print("ground truth:", len(gt), gt)
    print("predictions:", len(predictions), predictions)

    colors = [(0,128,0,125), (255, 165,0,125)] # green=GT orange=PRED

    bboxes_gt = []
    bboxes_pred = []
    c_gt = 0
    for i in gt:
        bb = i["bbox"]
        print(bb)
        # left, top, right, bottom => top, left, bottom, right
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_gt.append(['person_gt', bb, 1.0, c_gt])

    print("-")


    c_pred = 1
    for p in predictions:
        print(p)
        bb = p[1:]
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_pred.append(['person', bb, p[0], c_pred])

    bboxes = bboxes_gt + bboxes_pred
    #bboxes = bboxes_gt

    rec, prec, ap = voc_eval_twoArrs(bboxes_gt,bboxes_pred,ovthresh=0.5)
    aps.append(ap)

    # show only not working ones:
    #show_figures = (ap < 1.0)

    if show_figures:
        img = annotate_image_with_bounding_boxes(img, "", bboxes, colors, ignore_crops_drawing=True, draw_text=draw_text,
                                       show=False, save=False, thickness=[4.0, 1.0], resize_output = 1.0)


    if show_figures:
        #fig = plt.figure()
        plt.imshow(img)
        plt.title("Frame " + imagename + ", ap: " + str(ap) + " (green=GT orange=PRED)")
        plt.show()
        plt.clf()

    print("")

print(aps)
print("[AP] min, max, avg:",np.min(aps), np.max(aps), np.mean(aps))
fig = plt.figure()
plt.title("AP over frames, avg: "+str(np.mean(aps)))
plt.plot(aps)
plt.show()
