# FlappyBirdAI
Using the NEAT Genetic Neural Network Architecture to train a set of birds to play the popular game Flappy Bird. Also playable by user.

#### Table of Contents 
- [Description](#desc)
- [Usage Guide](#inst)
  * [Installation](#inst1)
- [Architecture](#arch)
  * [Neural Network](#nnar)
  * [Activation Function](#acfn)
- [Contributors](#cont)
- [License](#lics)

<a name="desc"></a>
## Description

This project was made using only the Python programming language. 

Using the open-source [NeuroEvolution of Augmenting Topologies (NEAT)](https://neat-python.readthedocs.io/en/latest/) library in python, we can train an Artificial Intelligence through a reward/punishement system to learn to play the popular game Flappy Bird. 

The user can choose the population number of birds, to try different architectures of neural networks at the same time, and also the generation number, which determines how many times the AI plays the game, training itself after each generation to play better. (Generation and population option is saved after one game is played using these options)

The user can also play the game. The high-score of the user is also saved, even after the game is closed and reopened.


<a name="inst"></a>
## Usage Guide

<a name="inst1"></a>
### Installation
1. Requirements: Python 3.5+ (64-bit)
2. Run below lines in your terminal

| Terminal | Demo |
| :----: | :--: |
| <code>$ cd ../FlappyBirdAI-master<br><br>$ pip3 install -r requirements.txt<br><br>$ python3 flappy_bird.py</code> |![][installation]|


<a name="arch"></a>
## Architecture

<a name="nnar"></a>
### Neural Network
- The Neural Network only starts with connected input and output layers as shown below.
- Hidden layers, biases and weights are then added. substracted, and modified as needed, (hence the Augmenting Topologies in NEAT)

![][neuralnet]

<a name="acfn"></a>
### Activation Function
The used [activation function](https://en.wikipedia.org/wiki/Activation_function) in this neural network is the TanH, where if an output of more than 0.5 is calculated, the bird jumps

![][activation]

<a name="cont"></a>
## Contributors
- Jad Haddad : jadhaddad01@protonmail.com

<a name="lics"></a>
## License
This project is licensed under the GPL-3.0 License. [License Details](../master/LICENSE)

[installation]: ./imgs/installation.gif

[neuralnet]: ./imgs/nnarch.png
[activation]: ./imgs/activation-tanh.png