import argparse


parser = argparse.ArgumentParser(description='Project: Find BBoxes in video.')
parser.add_argument('-horizontal_splits', help='number or horizontal splits in image', default='2')
parser.add_argument('-overlap_px', help='overlap in pixels', default='50')
parser.add_argument('-atthorizontal_splits', help='number or horizontal splits in image for attention model',
                    default='1')
parser.add_argument('-input', help='path to folder full of frame images',
                    default="/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PL_Pizza sample/input/frames/")
# parser.add_argument('-name', help='run name - will output in this dir', default='_Test-' + day + month)
parser.add_argument('-attention', help='use guidance of automatic attention model', default='True')
parser.add_argument('-thickness', help='thickness', default='10,2')
parser.add_argument('-extendmask', help='extend mask by', default='300')
parser.add_argument('-startframe', help='start from frame index', default='0')
parser.add_argument('-endframe', help='end with frame index', default='-1')
parser.add_argument('-attframespread', help='look at attention map of this many nearby frames - minus and plus',
                    default='0')
parser.add_argument('-annotategt', help='annotate frames with ground truth', default='False')
parser.add_argument('-reuse_last_experiment',
                    help='Reuse last experiments bounding boxes? Skips the whole evaluation part.', default='False')
parser.add_argument('-postprocess_merge_splitline_bboxes',
                    help='PostProcessing merging closeby bounding boxes found on the edges of crops.', default='True')

parser.add_argument('-debug_save_masks',
                    help='DEBUG save masks? BW outlines of attention model. accepts "one" or "all"', default='one')
parser.add_argument('-debug_save_crops', help='DEBUG save crops? Attention models crops. accepts "one" or "all"',
                    default='False')
parser.add_argument('-debug_color_postprocessed_bboxes', help='DEBUG color postprocessed bounding boxes?',
                    default='False')
parser.add_argument('-debug_just_count_hist',
                    help='DEBUG just count histograms of numbers of used crops from each video do not evaluate the outside of attention model.',
                    default='False')
parser.add_argument('-anchorf', help='anchor file', default='yolo_anchors.txt')
args = parser.parse_args()
INPUT_FRAMES = args.input
# RUN_NAME = args.name
SETTINGS = {}
SETTINGS["attention_horizontal_splits"] = int(args.atthorizontal_splits)
SETTINGS["overlap_px"] = int(args.overlap_px)
SETTINGS["horizontal_splits"] = int(args.horizontal_splits)
SETTINGS["anchorfile"] = args.anchorf
SETTINGS["startframe"] = int(args.startframe)
SETTINGS["endframe"] = int(args.endframe)
SETTINGS["attention"] = (args.attention == 'True')
SETTINGS["annotate_frames_with_gt"] = (args.annotategt == 'True')
SETTINGS["extend_mask_by"] = int(args.extendmask)
SETTINGS["att_frame_spread"] = int(args.attframespread)
thickness = str(args.thickness).split(",")
SETTINGS["thickness"] = [float(thickness[0]), float(thickness[1])]
SETTINGS["allowed_number_of_boxes"] = 500
SETTINGS["reuse_last_experiment"] = (args.reuse_last_experiment == 'True')
SETTINGS["postprocess_merge_splitline_bboxes"] = (args.postprocess_merge_splitline_bboxes == 'True')

SETTINGS["debug_save_masks"] = args.debug_save_masks
SETTINGS["debug_save_crops"] = (args.debug_save_crops == 'True')
SETTINGS["debug_color_postprocessed_bboxes"] = (args.debug_color_postprocessed_bboxes == 'True')
SETTINGS["debug_just_count_hist"] = (args.debug_just_count_hist == 'True')

INPUT_FRAMES = "/home/dl/lyx/myapl/frames/"
SETTINGS["annotate_frames_with_gt"] = True
SETTINGS["endframe"] = 10
RUN_NAME = "debugruns"
SETTINGS["attention_horizontal_splits"] = 2
SETTINGS["horizontal_splits"] = 4



