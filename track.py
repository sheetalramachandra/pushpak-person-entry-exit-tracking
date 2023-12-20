# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, '.')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.downloads import attempt_download
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

''' Additional imports for entry-exit logic '''
import numpy as np
from get_line_r import roi2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def person_entry_exit(bboxes, id, threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, \
    x2_line, x1_line2, x2_line2, person_seq, person_exit_count, person_entry_count, person_in, person_out):
    ''' This function determines the logic of getting the count of person in and out in a particular ROI '''
    #if id not in person_count and bboxes[3] >= (threshold_1 - 5):
    if id not in person_seq.keys() and bboxes[3] >= (threshold_1 - 5):
        #person_count.append(id)
        person_seq[id] = []
        # person_seq_exit[id] = []

    ''' For getting count of person in and person out '''
    try:
        if ((threshold_1) <= bboxes[3] <= (threshold_2)) and  bboxes[0] <= x2_line:
            if 'R1' not in person_seq[id]:
                person_seq[id].append('R1')
                # person_seq[id].append(str(frame))

        if ((threshold_3) <= bboxes[3] <= (threshold_4)) : #and  bboxes[0] <= x2_line
            if 'R2' not in person_seq[id]:
                person_seq[id].append('R2')
        
        if ((threshold_5) <= int(bboxes[3]) <= (threshold_6)) and (x1_line2 <= bboxes[0] <= x2_line2):# or ( x1_line2 <= bboxes[2] <= x2_line2)):
            if 'R3' not in person_seq[id]:
                person_seq[id].append('R3')
                
        if id in person_seq.keys() and (len(person_seq[id]) >= 2):# and  frame_diff > 6 :
            
            # if bboxes[3] <= threshold_1 and (person_seq[id][-2] > person_seq[id][-1]) :
            if bboxes[3] <= max(threshold_1, threshold_5) and ((person_seq[id][0] =='R2' and (person_seq[id][-1] =='R1' or person_seq[id][1] =='R1' )) or (person_seq[id][0] =='R3' and person_seq[id][-1] =='R1')):
            # if (person_seq[id][0] == 'R2' and person_seq[id][1] == 'R1') :
                #if 1:
                if id not in person_exit_count:
                    # print("EEEEEEEEEEEXXXXXXXXXXXIIIIIIIIIIITTTTTTTTT", id)
                    person_exit_count.append(id)
                    person_out = person_out + 1 
                    # print("deleting...........", id) 
                    del person_seq[id]

            # elif  bboxes[3] >= threshold_2  and (person_seq[id][-2] < person_seq[id][-1]): #or bboxes[3] >= threshold_1
            elif (bboxes[3] >= threshold_4 and (person_seq[id][0] =='R1' and (person_seq[id][-1] =='R2' or person_seq[id][1] == 'R2'))) or (person_seq[id][0] =='R1' and person_seq[id][-1] =='R3'): 
            # elif (person_seq[id][0] =='R1' and person_seq[id][1] =='R2'):
                # if 1:
                if id not in person_entry_count:
                    # print("EEEEEEEEEENNNNNNNNNNNTTTTTTTTTTRRRRRRRRRYYYYYYYYYY", id)
                    person_entry_count.append(id)
                    person_in = person_in + 1
                    # print("deleting...........", id) 
                    del person_seq[id]

            
    except Exception as e:
        pass

    return person_seq,person_exit_count, person_entry_count, person_in, person_out


def detect():
    ''' Argument Initializations'''
    yolo_model = 'yolov5s.pt' # type=str, help='model.pt path(s)'
    deep_sort_model = 'osnet_x0_25'
    source = "rtsp://admin:123456@123.176.45.172:554/cam/realmonitor?channel=1&subtype=0"  #type=str, help='source file path'

    imgsz = [640] # type=int, help='inference size h,w'
    conf_thres=0.4 # type=float, help='object confidence threshold'
    iou_thres=0.45 # type=float, help='IOU threshold for NMS'

    device='cpu' # default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    show_vid = True # action='store_true', help='display tracking video results'
    save_vid = True # action='store_true', help='save video tracking results' 

    classes = [0] # type=int, help='filter by class: --class 0, or --class 16 17'
    config_deepsort = "deep_sort/configs/deep_sort.yaml" # type=str, default="deep_sort/configs/deep_sort.yaml"
    project = ROOT / 'runs/track' # help='save results to project/name'
    name = 'exp' # help='save results to project/name'

    
    # source = str(source)
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initialize
    start_time = time.time()
    fps_ =  0 # to calculate average fps of the inference
    ctr = 0 # keep track of frame number
    person_entry_count= [] # in every frame, to store a unique id # In future optimization, see if we can remove this variable, as it seems redundant
    person_exit_count = [] # in every frame, to store a unique id # In future optimization, see if we can remove this variable, as it seems redundant
    person_in = 0 # the final stats of person entry count being displaye on the screen
    person_out = 0 # the final stats of person entry count being displaye on the screen
    person_seq = {}

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=False)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    up_time = time.time()
    dele_seq = time.time()

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        if webcam: 
            frame1 = im0s[0].copy()
        else:
            frame1 = im0s.copy()

        try:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            total = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        except:
            # print("Video is live no information of fps and total number frames avilabale")
            fps = 25

        if ctr == 0:
            ''' At frame number 0, mark the coordinates so as to make entry box, exit box & side box.'''
            x1_line, x2_line, y1_line, y2_line, x1_line1, x2_line1, y1_line1, y2_line1, x1_line2, x2_line2, y1_line2, y2_line2= roi2(frame1)
            # x1_line,  y1_line, x2_line, y2_line, x1_line1, y1_line1,  x2_line1, y2_line1, x1_line2,  y1_line2, x2_line2, y2_line2 = 521, 329, 755, 415, 500, 435, 747, 540, 761, 330, 876, 435
         
            threshold_1 = y1_line 
            threshold_2 = y2_line

            threshold_3 = y1_line1 
            threshold_4 = y2_line1 

            threshold_5 = y1_line2 
            threshold_6 = y2_line2

        ''' Performing inference on every second frame '''
        if ctr % 2 == 0:

            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(img, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=100) #agnostic_nms = False
            dt[2] += time_sync() - t3

            # Process detections
            temp_class = {}
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                person_count_curr_frame = 0 # if we don't define this, sometimes unreferenced local variable local variable error occurs

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # loop through each class and update the temp list to get the number of detections per class in a frame.
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        key1 = f"{names[int(c)]}"
                        val1 = int(f"{n.item()}")
                        temp_class.update({key1: val1})
                   
                    # since we are only interested in 'person' class:
                    if 'person' not in temp_class.keys():
                        temp_class.update({'person': 0})

                    person_count_curr_frame = temp_class['person'] # number of detections for 'person' class in this frame

                    ''' Deep SORT code starts here for tracking '''

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    '''track only inside area. Pass detections to deepsort'''

                    outputs = []
                    once = 0

                    ''' If all the object of interest (detected people) are above entry-box (y-coordinate-wise) by a sufficient distance, 
                        no need to track that object.
                    '''

                    for val in  det[:, 3:4]:
                        if val >=  (threshold_1 - 5) and once == 0:
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            once = once + 1
                    
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            # label = f'{id} {names[c]} {conf:.2f}'
                            label = f'{id}'
                            # print("--------------------label values : ",label)
                            # annotator.box_label(bboxes, label, color=colors(c, True))
                            annotator.box_label(bboxes, label, color=(255, 0, 0))

                            ''' Logic for monitoring the person entered or exited at from a ROI '''
                            person_seq, person_exit_count, person_entry_count, person_in, person_out,= person_entry_exit(bboxes, id, \
                                threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, x2_line, x1_line2, x2_line2, \
                                    person_seq, person_exit_count, person_entry_count, person_in, person_out)                            

                else:
                    deepsort.increment_ages()
                    LOGGER.info('No detections')

                '''Need to delete unnecessary data'''
                print("the sequence got ", person_seq)
                print("person_entry_list", person_entry_count)
                print("person_exit_list", person_exit_count)
                # print("the sequence exit got ", person_seq_exit)

                '''update list'''
                elapse = time.time() - up_time
                
                if elapse >= 3:  # after every three seconds, delete the lists as we have already updated the values
                    #person_seq.clear()
                    person_entry_count.clear()
                    person_exit_count.clear()
                    up_time = time.time()

                # Print time (inference-only)                
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')

                                # In future optimizations, see if this section of code can be shifted to outer loop. 
                                # Will that have any impact in optimization

                ''' To draw the translucent rectangular regions (entry-box, exit-box, side-box) '''
                sub_img = im0[y1_line:y2_line,x1_line:x2_line]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8)*255
                res = cv2.addWeighted(sub_img,0.5,white_rect, 0.5,1.0)
                im0[y1_line:y2_line,x1_line:x2_line] = res

                sub_img = im0[y1_line1:y2_line1,x1_line1:x2_line1]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8)*255
                res = cv2.addWeighted(sub_img,0.5,white_rect, 0.5,1.0)
                im0[y1_line1:y2_line1,x1_line1:x2_line1] = res

                sub_img = im0[y1_line2:y2_line2,x1_line2:x2_line2]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8)*255
                res = cv2.addWeighted(sub_img,0.5,white_rect, 0.5,1.0)
                im0[y1_line2:y2_line2,x1_line2:x2_line2] = res

                cv2.putText(im0, 'person_cnt/frame: ' + str(person_count_curr_frame), (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(im0, 'person_exit: ' + str(person_out), (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(im0, 'person_enter: ' + str(person_in), (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                fps_  = ( fps_ + (1./(time.time()-t1)) ) / 2  
                print("fps_ is:", fps_)

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        ctr = ctr +1
    
    print("total processing time", time.time()- start_time)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    with torch.no_grad():
        detect()
