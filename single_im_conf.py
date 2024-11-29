
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.torch_utils import select_device, smart_inference_mode



def single_im_conf(
    data='./data/DE.yaml',#attacktarLLVIP.yaml
    weights='./runs/train/tarLLVIP50/weights/best.pt',  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.5, #0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="test",  # train, val, test, speed or study
    device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    training = model is not None

    device = select_device(device, batch_size=batch_size)

    # # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        totalpred = []
        for si, pred in enumerate(preds):
            # labels = targets[targets[:, 0] == si, 1:]
            # nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            # path, shape = Path(paths[si]), shapes[si][0]
            # correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            # seen += 1
            #
            # if npr == 0:
            #     if nl:
            #         stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            #         if plots:
            #             confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            #     continue
            #
            # # Predictions
            # if single_cls:
            #     pred[:, 5] = 0
            # predn = pred.clone()
            # scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            #
            # # Evaluate
            # if nl:
            #     tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            #     scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            #     labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            #     #correct = process_batch(predn, labelsn, iouv)
            #     if plots:
            #         confusion_matrix.process_batch(predn, labelsn)
            # stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            #print("pred：",pred)
            totalpred = pred[:, 4].detach().cpu().numpy()
            print("所有置信度：",totalpred)
            prednum = len(totalpred)
            totalpred = totalpred.sum() / prednum
            if totalpred > 0:
                totalpred = totalpred
            else:
                totalpred = 0
            print("平均置信度：", totalpred)

        #print("置信度函数输出")
        #print(preds)
        return totalpred