# FlappyBirdAI
Using the NEAT Genetic Neural Network Architecture to train a set of birds to play the popular game Flappy Bird. Also playable by user.

#### Table of Contents 
- [Description](#desc)
- [Usage Guide](#inst)
  * [Tutorial](#tuto)
  * [Dependencies](#depd)
  * [Installation](#inst1)
  * [Running Program](#runp)
- [Architecture](#arch)
  * [Neural Network](#nnar)
  * [Activation Function](#acfn)
- [Contributors](#cont)
- [License](#lics)

<a name="desc"></a>
## Description

This project was made using only the Python programming language. 

Using the open-source NeuroEvolution of Augmenting Topologies (NEAT) library in python, we can train an Artificial Intelligence through a reward/punishement system to learn to play the popular game Flappy Bird. 

The user can choose the population number of birds, to try different architectures of neural networks at the same time, and also the generation number, which determines how many times the AI plays the game, training itself after each generation to play better. (Generation and population option is saved after one game is played using these options)

The user can also play the game. The high-score of the user is also saved, even after the game is closed and reopened.


<a name="inst"></a>
## Usage Guide

<!--<a name="tuto"></a>
### Tutorial
**Click below for the tutorial**
[![Watch the video](https://img.youtube.com/vi/8hMP5crzj6c/maxresdefault.jpg)](https://youtu.be/8hMP5crzj6c)-->

<a name="depd"></a>
### Dependencies
- Can run on Windows, Mac, or Linux.
- Install pip (python) and make sure it can run in your terminal.
- Run each line through your terminal to satisfy each library needed:
```
pip install pygame
pip install neat-python
pip install pygame-menu
pip install Pillow
pip install graphviz
pip install matplotlib
pip install numpy
```
<a name="inst1"></a>
### Installation Guide
1. Clone repository to your system OR press code -> Download ZIP.
2. If download as ZIP, extract FlappyBirdAI-master.zip and access the extracted folder.

<a name="runp"></a>
### Running the Program
1. Make sure you have followed the [Installation Guide](#inst1).
2. Open terminal and cd into the extracted folder's path.
3. Run flappy-bird.py

Terminal:
```
cd ../FlappyBirdAI-master
python "flappy_bird.py"
```

<a name="arch"></a>
## Architecture

<a name="nnar"></a>
### Neural Network
- The Neural Network only starts with connected input and output layers as shown below.
- Hidden layers, biases and weights are then added. substracted, and modified as needed, (hence the Augmenting Topologies in NEAT)

![alt text](https://github.com/jadhaddad01/FlappyBirdAI/blob/master/imgs/nnarch.png)

<a name="acfn"></a>
### Activation Function
The used [activation function](https://en.wikipedia.org/wiki/Activation_function) in this neural network is the TanH, where if an output of more than 0.5 is calculated, the bird jumps

![alt text](https://github.com/jadhaddad01/FlappyBirdAI/blob/master/imgs/activation-tanh.png)

<a name="cont"></a>
## Contributors
- Jad Haddad : jadhaddad01@protonmail.com

<a name="lics"></a>
## License
This project is licensed under the GPL-3.0 License. [License Details](../master/LICENSE)
