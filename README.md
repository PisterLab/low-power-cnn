# low-power-cnn

This directory is very much _in development_ and will include a series of tutorials for undergrads supporting data collection and analysis for training an end-to-end convolutional nueral network for microrobot control. Expect description of tutorials to be added along with our new implementations. Contact: [Nathan Lambert](mailto:nol@berkeley.edu) & [Lydia Lee](mailto:lydia.lee@berkeley.edu).

Research Direction
------------------
We want to equip our microrobots with control policies to explore their environment and eventually to accomplish tasks. An initial goal is to ant like cognition, by getting a small rc car or walking robot to wander the lab and avoid objects. The initial exploration is to train [SqueezeDet](https://arxiv.org/abs/1612.01051) on our own data. From there, we will need to decrease computational operations and maintain performance. 


Tutorials
---------
This section is intended for our undergrads supporting the project. The initial tutorial will walk you through the overall structure of using [Tensorflow for convolutional nueral networks](https://www.tensorflow.org/tutorials/layers). Future tutorials will be added below, so turn on notifications for updates.

Also in this directory is some Pytorch Tutorials and source code. We pulled the [raw code for SqueezeNet](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py) to show how custome nets are implemented. Also, we changed the [Pytorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) to use SqueezeNet rather than the built in ResNet18.

For the undergrads, we will be challenging you to identify the decade of the yearbook images found [here](http://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html). We have prepared the data in the common structure of '~\dir\class' where the classes are decades [here](https://berkeley.box.com/s/boleml1o1ltu5rbfhbt3lbwn51hbce2p).
