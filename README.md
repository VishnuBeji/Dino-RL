A 2013 publication by DeepMind titled ‘Playing Atari with Deep Reinforcement Learning’ introduced a new deep learning model for reinforcement learning, and demonstrated its ability to master difficult control policies for Atari 2600 computer games, using only raw pixels as input. In this tutorial, I will implement this paper using Keras. 


AI playing the game
I started with this project in early Sept 2023 and got some good results. However, the CPU-only system was the bottleneck for learning more features. 
There are many steps and concepts that we need to understand before we have a running model.

Steps:
Build a two-way interface between Browser (JavaScript) and Model (Python)
Capture and pre-process images
Train model
Evaluate

Getting Started
Reinforcement Learning Dino Run.ipynb
Make sure you run init_cache() first time to initialize the file system structure.

Reinforcement Learning
A child learning to walk

This might be a new word for many but each and every one of us has learned to walk using the concept of Reinforcement Learning (RL) and this is how our brain still works. A reward system is the basis for any RL algorithm. If we go back to the analogy of a child’s walk, a positive reward would be a clap from parents or the ability to reach a candy and a negative reward would be no candy. The child then first learns to stand up before starting to walk. In terms of Artificial Intelligence, the main aim for an agent, in our case the Dino, is to maximize a certain numeric reward by performing a particular sequence of actions in the environment. The biggest challenge in RL is the absence of supervision (labeled data) to guide the agent. It must explore and learn on its own. The agent starts by randomly performing actions and observing the rewards each action brings and learns to predict the best possible action when faced with a similar state of the environment

rl-framework
A vanilla Reinforcement Learning framework
Q-learning
We use Q-learning, a technique of RL, where we try to approximate a special function which drives the action-selection policy for any sequence of environment states. Q-learning is a model-less implementation of Reinforcement Learning where a table of Q values is maintained against each state, action taken and the resulting reward. A sample Q-table should give us the idea how the data is structured. In our case, the states are game screenshots and actions, do nothing and jump
0
,
1
q-table
A sample Q-table
We take advantage of the Deep Neural Networks to solve this problem through regression and choose an action with highest predicted Q-value. For detailed understanding of Q-learning please refer this amazing blog post by Tambet Matiisen. You can also refer my previous post to get around all the hyper-parameters specific to Q-learning

Setup
Let's setup our environment to start the training process.

1. Select the VM: We need a complete desktop environment where we can capture and utilize the screenshots for training. I chose a Paperspace ML-in-a-box (MLIAB) Ubuntu image. The advantage of MLIAB is that it comes pre-installed with Anaconda and many other ML-libraries.

Machine Learning in a Box
2. Configure and install Keras to use GPU:
We need to install Keras and tensorflow's GPU version
paperspace's VMs have these pre-installed but if not install them
pip install keras
pip install tensorflow

Also, make sure the GPU is recognized by the setup. Execute the python code below and you should see available GPU devices
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

3. Installing Dependencies

Selenium pip install selenium
OpenCV pip install opencv-python
Download Chromedrive from http://chromedriver.chromium.org
Game Framework
You can launch the game by pointing your browser to chrome://dino or just by pulling the network plug. An alternate approach is to extract the game from the open source repository of chromium if we intend to modify the game code.

Our model is written in python and game is built in JavaScript, we need some interfacing tools for them to communicate with each other.

Selenium, a popular browser automation tool, is used to send actions to the browser and get different game parameters like current score.

Now that we have an interface to send actions to the game, we need a mechanism to capture the game screen

The Selenium and OpenCV gave best performance for screen capture and pre-processing of the images respectively, achieving a descent frame-rate of 6-7 fps.

We require just 4 frames per time frame, enough to learn the speed as a feature

Game Module
We implement the interfacing between Python and JavaScript using this module. The snippet below should give you a gist of what's happening in the module.


Image Processing
Model Architecture
So we got the input and a way to utilize the output of the model to play the game so lets look at the model architecture.

We use a series of three Convolution layers before flattening them to dense layers and output layer. The CPU only model did not include pooling layers because I had removed many features and adding pooling layers would've led to significant loss of already sparse features. But with power of a GPU, we can accommodate more features without any drop in frame rate.

Max Pooling layers significantly improves the processing of dense feature set.

Original Game
Model Architecture
Our output layers consists of two neurons, each representing the maximum predicted reward for each action. We then choose the action with maximum reward (Q-value)
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


Training
These are the things happening in the training phase

Start with no action and get initial state (s_t)
Observe game-play for OBSERVATION number of steps
Predict and perform an action
Store experience in Replay Memory
Choose a batch randomly from Replay Memory and train model on it
Restart if game over

Average scores
Average scores per 10 games
The highest score recorded was 4000+ which is way beyond the previous model of 250 (and way beyond what most humans can do!). The plot shows the progress of highest scores for the training period games (scale = 10).

high scores
Max scores per 10 games
The speed of the Dino is proportional to the score, making it harder to detect and decide an action at higher speed. The entire game was hence trained on constant speed. The code snippets in this blog are just for reference. Please refer the GitHub repo for functional code with additional settings. Feel free to play around the code and contribute.
