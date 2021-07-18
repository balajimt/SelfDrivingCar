# Enhanced End-to-End System for Autonomous Driving Using Deep Convolutional Networks

Scholar article: https://link.springer.com/chapter/10.1007/978-3-030-65661-4_4

Abstract: The emergence of autonomous cars in today’s world makes it imperative to develop superlative steering algorithms. Deep convolutional neural networks are widely adopted in vision problems for their adept nature to classify images. End-to-end models have acted as an excellent substitute for handcrafted feature extraction. This chapter’s proposed system, which comprises of steering angle prediction, road detection, road centering, and object detection, is a facilitated version of an autonomous steering system over just considering a single-blind end-to-end architecture. The benefits of proposing such an algorithm for the makeover of existing cars include reduced costs, increased safety, and increased mobility.


## Training datasets

* [Rancho Palos Verdes dataset](https://drive.google.com/open?id=1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B) - Approximately 63,000 images, 3.1GB. Data was recorded around Rancho Palos Verdes and San Pedro California.
* [San Pedro Dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing) - Approximately 45,500 images, 2.2GB. One of the original datasets I made in 2017. Data was recorded around Rancho Palos Verdes and San Pedro California.

These images can be downloaded to `DrivingDatasetOutput` directory and the included filenames can be included in `DrivingDatasetOutput/data.txt`. This can be configured in `Configurations.py`

## Model training
```
python3 Trainer.py
```

## Real time steering wheel predictor
```
python3 SteeringAnglePredictor.py
```
### Output

![image](https://user-images.githubusercontent.com/7134314/126082995-25e6be03-aa4d-4725-a34d-0b9cef5ca6d6.png)

