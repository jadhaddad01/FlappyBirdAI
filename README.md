# FlappyBirdAI
Using the NEAT Genetic Neural Network Architecture to train a set of birds to play the popular game Flappy Bird. Also playable by user.

#### Table of Contents 
- [Description](#desc)
- [Usage Guide](#inst)
  * [Installation](#inst1)
  * [Playing Game](#plgm)
  * [In-Game Information](#gmin)
  * [AI Output Visualization](#aiov)
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

| Terminal                              | Demo            |
| :---------:                           | :---------:     |
| <code>$ cd ../FlappyBirdAI-master</code><br><br><code>$ pip3 install -r requirements.txt</code><br><br><code>$ python3 flappy_bird.py</code> |![][installation]|

<a name="plgm"></a>
### Playing Game
|             | User        | AI          |
| :---------: | :---------: | :---------: |
|             | ![][user]   |![][ai]      |
| **Options**     | None        | <ol><li><strong>Population:</strong> How many birds to train each generation at the same time</li><li><strong>Generations:</strong> How many times the AI will try the game again after all birds die</li></ol> |

<a name="gmin"></a>
### In-Game Information
| User        | AI          |
| :---------: | :---------: |
| <ul><li><strong>Score: </strong>How many times the user passed a pipe</li><li><strong>High Score: </strong>Highest score the user got since playing the game</li></ul>        | <ul><li><strong>Score: </strong>How many times the AI passed a pipe</li><li><strong>Gen: </strong>Which generation the AI is currently playing in</li><li><strong>Alive: </strong>How many birds are still playing the game</li><li><strong>Best NN: </strong>Visualized neural network of one of the birds that are still alive </li></ul> |

<a name="aiov"></a>
### AI Output Visualization
Output will be in the same folder **../FlappyBirdAI-master**.
For explanation on used terms, refer to [Neural Network Architecture](#nnar)
| File Name        | Demo          | Explanation          |
| :---------:      | :---------:   | :---------:          | 
| best_neural_net.png | ![][bnn]   | <ul><li>The best neural network throughout all generations</li><li><strong>If</strong> the user quits before all generations end, it shows the best neural network for the last generation played<ul><li style="color:green">Green Connection: positive weight</li><li style="color:red">Red Connection: negative weight</li><li>Dotted Connected: disabled connection</li></ul></li></ul> |
| avg_fitness.svg | ![][avft] | Model of the best bird in terms of fitness in each generation, and the average of all birds |
| speciation.svg | ![][spct] | Shows the population per species throughout the generations |



<a name="arch"></a>
## Architecture
<a name="nnar"></a>
### Neural Network
- The Neural Network only starts with connected input and output layers as shown below.
- Hidden layers, bias neurons, and weights are then added. substracted, and modified **as needed**, (hence the Augmenting Topologies in NEAT).
- Each bird is different from the other. Bird **species** is defined when birds possess different base architectures of nodes connected to one another. Birds of the same species only have different values for their weight connections.
- Bird **fitness** is measured by score and distance. By trying to maximize fitness, the neural network changes it's architecture each generation to fulfill that goal.

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

[user]: ./imgs/user.gif 
[ai]: ./imgs/ai.gif 
[installation]: ./imgs/installation.gif 

[bnn]: ./imgs/best_neural_net.png
[avft]: ./imgs/avg_fitness.svg
[spct]: ./imgs/speciation.svg

[neuralnet]: ./imgs/nnarch.png
[activation]: ./imgs/activation-tanh.png