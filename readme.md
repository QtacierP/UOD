## Uncertainty-aware Object Detection for traffic light

------------

PyTorch + MMDetection backbone

![demo](https://github.com/QtacierP/UOD/blob/main/docs/s101.gif)

## Train

Firstly, you need to convert original GTSDB dataset into MS-COCO dataset format.

Put your dataset GTSDB into ../data/ and rename it as original_GTSDB.

Then run 

```bash
python gtsdb2coco.py
```

Then run on single GPU

```bash
cd codes
CUDA_VISIBLE_DEVICES=X python main.py configs/xxx.json 
```

Or you can choose launch on multiple GPUs. You can use the template from current sh dist_train_faster_rcnn_r50_2x.sh

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=20001 ./train.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_gtsdb.py --resume-from /data2/chenpj/UOD/codes/work_dirs/faster_rcnn_r50_fpn_2x_gtsdb/epoch_36.pth  --launcher pytorch
```

## Inference

For inference, just use demo.py !

```
python demo.py --input_image ${IMAGE} --input_video ${VIDEO} --output {OUTPUT_FILE} --load_from ${MODEL_PATH}
```

Now, enjoy  your traffic light detecting playground!



## Model Zoo

We only test Faster-RCNN-FPN-ResNet50-FPN as baseline. Here, we provide the .json config and pre-trained model.

You can use any model in MMDetection to implement the traffic light detecting easily! 



| backbone                         | config                                                       | pre-trained model                                            | mAP    |
| :------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------ |
| Faster-RCNN-ResNet50-FPN         | [config](https://github.com/QtacierP/UOD/blob/main/codes/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_gtsdb.py) | [pre-trained weights](https://github.com/QtacierP/UOD/releases/download/Model/faster_rcnn_r50_fpn-fb4e1380.pth) | 0.5210 |
| Faster-RCNN-ResNeSt50-FPN-SyncBN | TODO                                                         | TODO                                                         | 0.5660 |



## Next  Plan

We should add custom uncertainty-head in model and uncertainty-aware loss function.

## Acknowledgement

Thanks for [open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark (github.com)](https://github.com/open-mmlab/mmdetection) to share such a wonderful tool for universal object detection pipeline. If you are beneficial from this repo, please star it :)

