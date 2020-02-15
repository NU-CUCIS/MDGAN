# Microstructural Materials Design via Deep Adversarial Learning Methodology
This software is an deep learning application for generating materials microstructure images using generative adversarial networks. The networks are trained on synthetic 2D microstructure images.

This software requires the input in the form of a numpy array of shapre (x, 128, 128) where x is the number of microstructure images and the dimension of microstructure is 128 by 128.
The software takes the microstructure images as input to train the generative adversarial networks.
The trained generator can then be used to produce microstructure images.
The detailed description about data preprocessing and model can be found in the published paper given below. 

## Requirements
* Python 2.7.12
* Numpy 1.14.0
* Keras 2.1.2
* Pickle 2.0
* TensorFlow 1.5.0-rc1
* H5PY 2.7.1

## Source Files
1. `gan_training.py`: The script that trains the GAN that can generate two-phase microstructure image.
2. `example_data.pkl`: Example data of synthetic microstructure images, including 10 2D microstructures.

## How to run it
1. Download the file containing the weights of VGG16,
   [vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5), before the training.
2. Run commend below, which trains the GAN and save the ooutput.
   ```
   python gan_training.py
   ```

## Acknowledgement
The Rigorous Couple Wave Analysis simulation is supported by Prof. Cheng Sun's lab at Northwestern University. This work is primarily supported by the Center of Hierarchical Materials Design (NIST CHiMaD 70NANB14H012) and Predictive Science and Engineering Design Cluster (PS&ED, Northwestern University). Partial support from NSF awards DMREF-1818574, DMREF-1729743, DIBBS-1640840, CCF-1409601; DOE awards DE-SC0007456, DE-SC0014330; AFOSR award FA9550-12-1-0458; and Northwestern Data Science Initiative is also acknowledged. 

## Publications
Z. Yang, X. Li, L. C. Brinson, A. Choudhary, W. Chen, and A. Agrawal, “Microstructural Materials Design via Deep Adversarial Learning Methodology,” Journal of Mechanical Design, vol. 140, no. 11, p. 10, 2018.

## Contact
* Zijiang Yang <zyz293@ece.northwestern.edu>
* Ankit Agrawal <ankitag@ece.northwestern.edu>



