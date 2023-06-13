# Genetic Neural Network
Access the game through this link: https://alexaoliveira2000.github.io/

This project was developed as a way for me to learn how to implement feedforward neural networks and how to train them to become expert at a given game. In this document, I show you the game I created and the logic of the NN evolution.

# Terminologies Definitions
Before you start testing the game, it is important that you understand what you're playing with because I use some terminology that is specific for Genetic Neural Networks, and so here's a list of these terms in case you don't follow:

### Environment
All the information that the game itself provides; it is the context where the agent operates.

### Agent
Autonomous entity that interacts with the environment (game). Processes inputs from the environment and makes decisions based on learned patterns. It has a brain  (neural network) which is shown on the game. 

### Fitness
It is a measure (percentage) of how well the agent performed.

### Generation
A group of agents that are going to be tested in a row and then selected to a new generation based on each agent's fitness (hopefully a more refined one).

### Mutation Rate
When a new agent is created from another one (child), the agent's brain is "mutated" at a rate. The bigger the rate, the more the brain changes.

### Crossover
When two agents brains are put together to generate a new brain (child), the new brain is a mix of both parents, combining traits/patterns. This is called crossover.

### Crossover Rate
Percentage of the first parent brain that is going to be on the child brain. Usually when the first parent has higher fitness than the second one, we want a bigger percentage of this parent's brain.

## Game Instructions
Once you enter the provided link, you'll se a screen with a bunch of information. All the inputs are part of the *Train* button, and each one of them is explained below.
- Play: this starts the game for a human to play - you. This is just a way to test the envrionment and look for bugs. The avaliable actions you can make are the up and down arrows - to jump and crouch, respectively. There's no score limit, so you can keep going as far as you want to. Once you hit an obstacle, the game stops and you can reset it:
- Reset: resets the current game the human is/was playing;
- Train: this starts the train based on all the inputs provided above.
