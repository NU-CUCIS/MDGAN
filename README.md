# Microstructural materials design via deep adversarial learning methodology
This software is an deep learning application for generating materials microstructure images using generative adversarial networks. The proposed approach is trained on synthetic 2D microstructure images.

To use this software, what the algorithm requires as input are a numpy array. The shape of data point is (x, 128, 128) where x is the number of microstructure images and the dimension of microstructure should be two-dimensional (i.e. 128x128). The software will take the microstructure images as input, and train the generative adversarial networks. After training, the generator can be used to produce microstructure images. (The detail about data preprocessing and model is in related sections of published paper). 

## Requirements ##
Python 2.7.12;
Numpy 1.14.0;
Keras 2.1.2;
Pickle 2.0;
TensorFlow 1.5.0-rc1;
H5PY 2.7.1;

## Files ##
1. gan_training.py: The script to train the GAN that can generate two-phase microstructure image.
2. example_data.pkl: Example data of synthetic microstructure images, including 10 2D microstructures.

## How to run it
1. Weights of VGG16 is required to train the proposed model. Please download its weights at https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5, and put it in the same repository.
2. To run gan_training.py: use commend 'python gan_training.py'. The script will train the GAN and save your GAN.


## Acknowledgement
The Rigorous Couple Wave Analysis simulation is supported by Prof. Cheng Sun's lab at Northwestern University. This work is primarily supported by the Center of Hierarchical Materials Design (NIST CHiMaD 70NANB14H012) and Predictive Science and Engineering Design Cluster (PS&ED, Northwestern University). Partial support from NSF awards DMREF-1818574, DMREF-1729743, DIBBS-1640840, CCF-1409601; DOE awards DE-SC0007456, DE-SC0014330; AFOSR award FA9550-12-1-0458; and Northwestern Data Science Initiative is also acknowledged. 

## Related Publications ##
Z. Yang, X. Li, L. C. Brinson, A. Choudhary, W. Chen, and A. Agrawal, “Microstructural Materials Design via Deep Adversarial Learning Methodology,” Journal of Mechanical Design, vol. 140, no. 11, p. 10, 2018.

## Contact
Zijiang Yang <zyz293@ece.northwestern.edu>; Ankit Agrawal <ankitag@ece.northwestern.edu>



