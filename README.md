# Improved Hierarchical Transformer-Based Multi-Similarity Hashing via Label-guided Learning for Remote Sensing Image Retrieval
This paper is accepted for publication with Journal of Machine Learning and Cybernetics.


## Dependencies
We use python to build our code, you need to install those package to run
- Python 3.9.7
- Pytorch 1.12.1
- torchvision 13.1
- CUDA 11.3


## Training
All parameters are defined in the train.py file, and the test methods are integrated into the train.py file. Therefore, it is only necessary to run
python train.py


### Processing dataset
Before training, you need to download the UCMerced dataset http://weegee.vision.ucmerced.edu/datasets/landuse.html,
AID dataset from https://captain-whu.github.io/AID ,WHURS dataset from https://captain-whu.github.io/BED4RS.


### Download Hierarchical Transformer pretrained model
Pretrained model is jx_nest_base-8bc41011.pth.

