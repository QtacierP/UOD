from mmdet.apis import init_detector, inference_detector
import mmcv
import argparse
from models.uncertanty import UncertaintyRoIHead, UncertaintyBoxHead, UncertaintyLoss, UncertaintyShared2FCBBoxHead
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('config', help='train config file path')
parser.add_argument('--load_from', help='load model from')
parser.add_argument('--input_image', default='', help='load image from')
parser.add_argument('--input_video', default='', help='load video from')
parser.add_argument('--output', help='output to somewhere')


args = parser.parse_args()


# Specify the path to model config and checkpoint file
config_file = args.config
checkpoint_file = args.load_from
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

if args.input_image != '':
    # test a single image and show the results
    img = args.input_image  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
    # visualize the results in a new window
    #model.show_result(img, result, show=True)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=args.output)

elif args.input_video != '':
    # test a video and show the results
    video = mmcv.VideoReader(args.input_video)
    frames = []
    for frame in tqdm(video):
        result = inference_detector(model, frame)
        frames.append(model.show_result(frame, result, wait_time=1))
    vidFilename = args.output
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(vidFilename, fourcc, video.fps, video.resolution)
    for frame in tqdm(frames):
        writer.write(frame.astype('u1'))
    writer.release()
    
