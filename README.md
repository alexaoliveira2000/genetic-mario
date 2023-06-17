# Genetic Neural Network
Access the game through this link: https://alexaoliveira2000.github.io/

This project was developed as a way for me to learn how to implement feedforward neural networks and how to train them through reinforcement learning. In this document, I show you the game I created and the logic of the evolution I implemented.

# Terminologies Definition
For the training in-game, there are some parameters you can change. Some important definitions are below:
| Terminology | Definition |
|---|---|
| Environment | All the information that the game itself provides. It is the context where the agent operates. |
| Agent | Autonomous entity that interacts with the environment (game). Processes inputs from the environment and makes decisions based on learned patterns. It has a brain  (neural network) which is shown on the game.  |
| Fitness | It is a measure (percentage) of how well the agent performed. |
| Generation | A group of agents that are going to be tested in a row and then selected to a new generation based on each agent's fitness (hopefully a more refined one). |
| Mutation Rate | When a new agent is created from another one (child), the agent's brain is "mutated" at a rate. The bigger the rate, the more the brain changes.
| Crossover | When two agents brains are put together to generate two new brains (children), the new brains is a mix of both parents, combining traits/patterns. This is called crossover. The children have the exact opposite genes from one another. |
| Crossover Rate | Percentage of the first parent brain that is going to be on the first child brain.

## Neural Network
The brain of each agent is, in reality, a neural network. You can see this network on the game, when an agent is training.

In practice, a neural network is just an object that stores a bunch of values for each connection - called weights (each black line you see at the agent's brain on the game) - and can act as a function that predicts outputs based on given inputs. The objective of any neural network is to predict the correct output based on any input, like a function. The difference from a normal function is that these outputs are calculated based just on these weights, not a specific algorithm inside it.

Each node on the neural network (each blue circle) can receive one ore more inputs and produce an output, just like a function. This is why the first layer of nodes is called **input layer** - the input each node receives is one information about the environment. On all other layers, the input is based on the weights that come from the previous layer.

Alone, these weights don't mean anything (because they are just numbers), but together with all of the network, they can behave exactly the way we want it to. When the value of a weight changes on a network, all of the other weights (and, therefore, the predictions) can become completely different.
Initially these weights are completely random, but with time and train they can have values that all together predict correctly. So, if you notice, training a neural network is just the action of changing these weights little by little and seeing how the predictions behave.

To make this clearer, let's look at an example.

1. Create a neural network:
```` js
// Create a neural network with 4 input nodes, 6 hidden nodes and 3 output nodes
let neuralNetwork = new NeuralNetwork(4, 6, 3)
````
2. Get the 4 weights from the first input node to each of the hidden layer nodes:
```` js
// all weights from the neural netowrk
let weights = neuralNetwork.model.getWeights()
// all weights from each input node to each hidden layer node
let weightsInputToHidden = weights[0].arraySync()
// all weights from the first input node to each hidden layer node
let weightsFirstInputToHidden = weightsInputToHidden[0]
// show the weights on console
console.log(weightsFirstInputToHidden)
````
3. Output:
```` js
[0.43523460626602173, -0.4457567632198334, -0.11613214761018753, 0.3477533161640167, -0.7608528733253479, -0.6688799858093262]
````

4. See the model:

![nn](https://github.com/alexaoliveira2000/genetic-mario/assets/77057098/404f9abe-1b01-4ceb-9420-410bec4ae984)

If you focus your attentions on the weights from the first input node to each of the hidden layer nodes on the image abore, you'll notice that the first connection is the strongest (more thick), which corresponds to the value of the output array, because the first value is the highest.

As you can see, a neural network is just a bunch of values in a bunch of arrays, working together to make a prediction.

## Training / Evolution
In this game, evolution happens in two ways: through mutations and and through crossover (which is optional, as you may see). These events also happen on the real world, and so you may find some similarities with biology.

### Mutations
Mutations are little changes in the agent's brain that happen every time we create an agent from another one (parent). The percentage of the brain that is changed is defined by a rate.

In practice, if you have a mutation rate of 0.01, this just means that for each weight of the neural network, there's a 1% chance that that weight value is changed to a random value.

The **NeuralNetwork** class has a method that does just this to change it's model:
```` js
mutate(rate) {
        tf.tidy(() => {
            const weights = this.model.getWeights();
            const mutatedWeights = [];
            for (let i = 0; i < weights.length; i++) {
                let tensor = weights[i];
                let shape = weights[i].shape;
                let values = tensor.dataSync().slice();
                for (let j = 0; j < values.length; j++) {
                    if (Math.random() < rate) {
                        let w = values[j];
                        values[j] = w + this.randomGaussian();
                    }
                }
                let newTensor = tf.tensor(values, shape);
                mutatedWeights[i] = newTensor;
            }
            this.model.setWeights(mutatedWeights);
        });
    }
````

When we create an agent from another one (if we pass an already existing brain as an argument to the constructor of the new agent), this mutation always happen:
```` js
class Agent extends Item {
    constructor(context, x, y, width, height, color, brain, generation, mutationRate, hiddenNodes) {
        super(context, x, y, width, height, color);
        this.velocity = 0;
        this.action = Action.STAND;
        this.score = 0;
        this.fitness = 0;
        this.generation = generation;
        if (brain instanceof NeuralNetwork) {
            this.brain = brain.copy();
            this.brain.mutate(mutationRate);
        } else {
            this.brain = new NeuralNetwork(4, hiddenNodes || 6, 3);
        }
    }
    ...
}
````

In game, you have the power to change this rate.

### Crossover
For crossover to happen, you must have two brains (two parents), because as I explained on the first section, crossover is the combination of two neural networks.

In practice, it's just the creation of two neural networks that have some percentage of the first parent weights and some percentage of the second parent weights. The two generated neural networks have the exact opposites genes from one another (i.e., if child1 has the weight from parent2, then child2 has the weight from parent1, for each weight of the network).

### Pool Selection
Pool selection (also called Tournament) is a way of selecting a member from the current population, based on its fitness. The "pool" is just an array with all agents multiplied by they're fitness. This means that an agent with highest fitness will appear more times in the pool, having therefore more chances to be picked. An agent is then randomly picked from the pool. 

### Evolution Process
The workflow of the evolution process with crossover is as follows:
1. Create generation;
2. Test each agent and save score and fitness;
3. Create "selection pool";
4. While new population hasn't been fully populated:
5. Pick 2 agents from the pool;
7.          Perform crossover with them - 2 children are generated;
9.          Perform mutations for each new child;
10.         Put children in the new generation;
11. Back to step 1.
There are many more aspects of training neural networks a little more in-depth, such has the number of layers (and number of nodes on each one of them), activation functions, weights normalization and much more that will not be covered here.
