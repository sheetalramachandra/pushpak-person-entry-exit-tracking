# Pushpak-Person-Entry-Exit-Tracking

## High Level Description
The project aims to monitor the presence of individuals on the premises based on the following three descriptors:
- Counting the number of individuals entering the premises. 
- Counting the number of individuals exiting the premises.
- Counting the total number of people present on the premises.

## Low Level Description
- YOLOv5 model is used for object (person) detection in every frame.
- Deepsort model is used for object (person) tracking across a series of frames.

## About this Repo
- This Repo is built on top of the baseline [Real-time multi-object tracker using YOLO v5 and deep sort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).
- At the time of writing this repo, we used the Deepsort baseline code with commit ID [e5d79a1](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/e5d79a17d4c951fe1265470b1fc51f24a46e096e) for and YOLOv5 baseline code with commit ID [aff0281](https://github.com/ultralytics/yolov5/tree/aff0281969f12b2805833bebb885bcf1377676dc).

## Using this Repo on your Local Machine
- Install this Repo using the following shell command:
```
git clone https://gitlab.com/DeeptiRawat/pushpak-person-entry-exit-tracking.git
```
- Go to the directiory using the shell command 
```
cd <this repo's name in your local machine>
```
- Make sure to install the required dependencies (preferalbly in a virtual environment) using the `requirements.txt`:
```
pip install -r requirements.txt
```
- Track the number of person entries and exits, along with the present people count using the following:
```
python3 track.py --source <source file> --yolo_model <yolo weight file> --device <device name> --classes 0 --show-vid --save-vid
```
    - `track.py` is the main python file containing the tracking logic for computing the three descriptors.

## About requirements.txt (*pip install -r requirements.txt*)

`torchreid`
> Torch ReID is a person ReID (re-identification) library written in torch. Person Re-Identification (ReID) aims to recognize a person-of-interest across different places and times.
> 
> Torch ReID is essentially useful to track an individual across multiple cameras.

Here we are not doing `pip install torchreid` as this module is already present in this Repo - *./deep_sort/deep/reid/torchreid*

`Cython`
> In our project, we are using Cython library for torch ReID modules. Using Cython makes the end-to-end evaluation of ReID faster. 
> 
> In case, it is not available or not properly installed, Python evaluation is utilised and you get a runtime warning as follows:
- `UserWarning: Cython evaluation (very fast so highly recommended) is unavailable, now use python evaluation`

To Fix this warning, use Cython evaluation:

## Using Cython Evaluation:
While running the root python file `track.py`, one may get the following warning:
- `UserWarning: Cython evaluation (very fast so highly recommended) is unavailable, now use python evaluation`.

The aforementioned warning is thrown because in *rank.py* file, we are trying to call a Cython function (line 11) without performing C library linking.

To perform C library linking:
1) Download the latest Cython release from [commonmark.js](https://cython.org/). Unpack the tarball or zip file, enter the directory, and then run:
*python setup.py install*   
   
1) Run `python setup.py build_ext --inplace` (./deep_sort/deep/reid/torchreid/metrics/rank_cylib/) 
2) Once the build is successful, run `track. py` again. The warning must disappear. 

For more info on Cython implementation visit: [commonmark.js](http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html)
- For this project, we are using *Cython-3.0.0a10*
 
Note: Cython evaluation is faster than Python evaluation.

## Comparing the old Deepsort (ckpt.t7 weights) approach with the new (osnet weights)

We are using a ReID model *osnet* model as compared to *ckpt.7* model used previously ([Github link](https://github.com/pushpakai/pushpak-thub-deployment) - branch person-entry-exit-v2). 
The advantages are:
- *osnet* model size - 4 MB, whereas *ckpt.7* model size - 46 MB.
- Person tracking accuracy of *osnet* and *ckpt.7* model checked accross different videos are comparable.

## Benchmarking New Tracking Weights against the Old Tracking weights
To benchmark new tracking weights (osnet_x0_25_imagenet.pth) against the old tracking weights (ckpt.t7), we compare their performance on three basis:
- FPS
- Memory footprint
- Accuracy of tracking algorithm

Note: While using the old code ([a0c89a0](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/a0c89a0cc9df013fd4f16ed0a57f55cfc14e4f7e)), modify the config file as follows:
```
 DEEPSORT:
  REID_CKPT: "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
  MAX_DIST: 0.2  # previously 0.2   #will keep 0.125 #0.15
  MIN_CONFIDENCE: 0.4 #0.5 # previously 0.6 
 # NMS_MAX_OVERLAP: 0.55
  MAX_IOU_DISTANCE: 0.7
  MAX_AGE: 30 # tried 40 # on 10 it gives unique id to each vehicle. #20
  N_INIT: 3  #tried 2
  NN_BUDGET: 100 # tried 100
```
