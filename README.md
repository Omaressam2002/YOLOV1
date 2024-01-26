# YOLOV1
**Architecture and training was inspired by the original YOLOV1 paper**: https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf

**Used the model to create a Swift APP for Object Detection, repo link**: https://github.com/Omaressam2002/BasketBallDetectionAPP 

**Training** :
- Pretrainig :
  - For pretraining we use the first 20 convolutional layers , Dakrnet , followed by an average-pooling layer and a fully connected layer 
  - trained this network on the ImageNet data set of 1000 class : https://www.image-net.org 
  - trained for 3 weeks and acheived 65% top 5 accuracy
- Training :
  - Trained the full network on a custom gathered dataset with three classes of objects : Basketball, Players, and Hoops
  - model after training : https://drive.google.com/file/d/1ZxVegVUlBxFkqVk7Vt6si7QHMMFHbKGc/view?usp=sharing

**Test Video** : https://youtu.be/VRyyayQanQM

**For more test cases look ath the end of the notebook in the repo or try the model yourself** : https://drive.google.com/file/d/1ZxVegVUlBxFkqVk7Vt6si7QHMMFHbKGc/view?usp=sharing 

