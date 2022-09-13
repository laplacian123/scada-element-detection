## Requirements
Before you use this package, please make sure you have installed the requirements and completely followed the instructions of [the original repository of yolov5](https://github.com/ultralytics/yolov5).

## Usage
### Image Augmentation
To train the object detection model for the single-line diagram, we must first prepare the training data for each element. Since there is only one image for each type of element, we have to apply some image augmentation such as blur, zoom or adding noise. Here we can complete this task with only one line of command, as long as the training images are saved in the `src_img` folder with their own label/class name as their file name: 
```
python augment.py "tr" "cb" "mccb"
```
To improving the training performance, you can also add some background images with file names `bg1`, `bg2`, `bg3`... in the `src_img` folder. By the time you use the previous command, the background images are also augmented. The augmentation pipeline is defined in the `augment.py` file, you can change it if needed.


### Training
Here we use yolov5 to make our object detection model. Before any training process, we must re-arrange our data such that the yolov5 can make use of it. We can complete this task by only a line of command (again):
```
python data_prep.py
```
`data_prep.py` can take some arguments. For example, you can set the randomseed by `-r {randomseed}`, where the default randomseed is `87`, etc.

Then, we have the change the working folder to `yolov5`:
```
cd yolov5
```

As an example, we train and finetune the yolov5n model (the smallest yolov5 model) by the following command:
```
python train.py --img 1024 --batch 16 --epochs 500 --data ../dataset.yaml
-weights yolov5n.pt --device 0 --freeze 12
```
For a detailed information of what these arguments mean, please refer to [the original repository of yolov5](https://github.com/ultralytics/yolov5).

## Inference
To make inferences, please run `detection.ipynb`.