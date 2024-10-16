# Cell_Free_Unsupervised_Learning_Message_Allocation
Code from the paper "Semi-decentralized Message Allocation for Cell-Free Networks using Unsupervised Learning"

Directories:
Matlab_scripts - Contains scripts used to run centralized methods from paper: Interior-Point and Sequential Quadratic Programming
Matlab_scripts_Creat_Datasets - Largely based on "Making Cell-Free Massive MIMO Competitive With MMSE Processing and Centralized Implementation" paper cited below. Used to generate channel data for simulation.
Python_scripts - Contains scripts used to train/test Neural networks for message/power allocation. Borrows inspiration from eras DDPG demo cited below. 
Test_Sets - Contains test data for the cell-free networks used in the paper
Training_Sets - Contains link to Dropbox to download training data for the cell-free networks used in the paper


Citations:

Installed Libraries:
Numpy - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020).

Matplotlib -  J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.

Tensorflow/keras - Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.


Cell-Free Generation: 
Making Cell-Free Massive MIMO Competitive With MMSE Processing and Centralized Implementation - Emil Björnson and Luca Sanguinetti, “Making Cell-Free Massive MIMO Competitive With MMSE Processing and Centralized Implementation,” IEEE Transactions on Wireless Communications, vol. 19, no. 1, pp. 77-90, January 2020.
Github for Making Cell-Free code: https://github.com/emilbjornson/competitive-cell-free

Keras tutorial: https://keras.io/examples/rl/ddpg_pendulum/


