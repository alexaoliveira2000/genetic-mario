# Genetic Neural Network
This project was developed as a way for me to learn how to implement feedforward neural networks and how to train it to become expert at a given game. In this document, I show you the game I created and the logic of the NN evolution.

Before we start, in this document I use some terminology that is specific for Genetic Neural Networks, and so here's a list of these terms in case you don't follow:
- Environment: all the information that the game itself provides; it is the context where the agent operates;
- Agent: autonomous entity that interacts with the environment (game); processes inputs from the environment and makes decisions based on learned patterns;

## The Game
I wanted to make a simple game (like an arcade game), where the actions we can make are very limited, aswell as the information we analyse as players. I ended up with thinking about Super Mario, but with the difference that the agent cannot walk sideways, which means that the only available options are to *stand*, *jump* and *crouch*.
