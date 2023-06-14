# Genetic Neural Network
Access the game through this link: https://alexaoliveira2000.github.io/

This project was developed as a way for me to learn how to implement feedforward neural networks and how to train them to become expert at a given game. In this document, I show you the game I created and the logic of the NN evolution. Anyone that has some basic programming skills must be able to understand what is discussed in this document.

# Terminologies Definition
Before you start testing the game, it is important that you understand what you're playing with because I use some terminology that is specific for Genetic Neural Networks, and so here's their definition:
| Terminology | Definition |
|---|---|
| Environment | All the information that the game itself provides. It is the context where the agent operates. |
| Agent | Autonomous entity that interacts with the environment (game). Processes inputs from the environment and makes decisions based on learned patterns. It has a brain  (neural network) which is shown on the game.  |
| Fitness | It is a measure (percentage) of how well the agent performed. |
| Generation | A group of agents that are going to be tested in a row and then selected to a new generation based on each agent's fitness (hopefully a more refined one). |
| Mutation Rate | When a new agent is created from another one (child), the agent's brain is "mutated" at a rate. The bigger the rate, the more the brain changes.
| Crossover | When two agents brains are put together to generate a new brain (child), the new brain is a mix of both parents, combining traits/patterns. This is called crossover. |
| Crossover Rate | Percentage of the first parent brain that is going to be on the child brain. Usually when the first parent has higher fitness than the second one, we want a bigger percentage of this parent's brain. |

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

In practice, it's just the creation of a neural network that has some percentage of the first parent weights and some percentage of the second parent weights. We can define how much percentage of the first agent brain is going to be on the final brain in a static way (say, for example, always 50%), but on this game I defined this percentage based on fitness.

By instance, if the first parent has 0.25 of fitness and the second one has 0.5, we'll want to have a bigger percentage of the second parent weights - more specifically 75% of the second one because it has twice has higher fitness - and so we can create a formula to calculate the percentage of the second parents brain that is going to be on the created one:

$rate$ = $parent2.fitness$ $/$ $(parent1.fitness$ $+$ $parent2.fitness$ $+$ $0.01$)

The "+ 0.01" on the formula is to prevent the complete crossover of the second parent to the first one in case the second parent fitness is 0 (100% crossover). Also, this formula just doesn't make sense if both fitnesses are 0, and so in this case the rate is just 0.5 (50%):
```` js
performCrossover() {
        let bestAgent = this.members[0];
        for (const agent of this.members) {
            if (agent.fitness > bestAgent.fitness)
                bestAgent = agent;
        }
        let newMembers = []
        for (const agent of this.members) {
            let rate = bestAgent.fitness ? bestAgent.fitness / (bestAgent.fitness + agent.fitness + 0.01) : 0.5;
            agent.brain.crossover(bestAgent.brain.model, rate)
            let newAgent = new Agent(agent.context, agent.x, agent.y, agent.width, agent.height, agent.color, agent.brain, agent.generation + 1, this.mutationRate, this.hiddenNodes);
            newMembers.push(newAgent);
        }
        this.members = newMembers;
}
````

```` js
crossover(model, crossoverProbability) {
        let tensorsModel1 = this.model.getWeights();
        let tensorsModel2 = model.getWeights();
        // for each tensor
        for (let i = 0; i < tensorsModel1.length; i++) {
            let nodesWeightsModel1 = tensorsModel1[i].arraySync();
            let nodesWeightsModel2 = tensorsModel2[i].arraySync();
            // if it is a bias tensor
            if (tensorsModel1[i].name.includes("bias")) {
                // for each bias
                for (let k = 0; k < nodesWeightsModel1.length; k++)
                    if (Math.round(Math.random()))
                    nodesWeightsModel1[k] = nodesWeightsModel1[k];
                tensorsModel1[i] = tf.tensor(nodesWeightsModel1)
            } else {
                // for each node connections
                for (let j = 0; j < nodesWeightsModel1.length; j++) {
                    let nodeWeightsModel1 = nodesWeightsModel1[j];
                    let nodeWeightsModel2 = nodesWeightsModel2[j];
                    // for each weight
                    for (let k = 0; k < nodeWeightsModel1.length; k++)
                        if (Math.random() < crossoverProbability)
                            nodeWeightsModel1[k] = nodeWeightsModel2[k];
                }
                tensorsModel1[i] = tf.tensor2d(nodesWeightsModel1)
            }
        }
        this.model.setWeights(tensorsModel1);
        return this.model;
    }
````

### Selection
The workflow of the evolution process with crossover is as follows:
1. Create generation;
2. Test each agent and save score and fitness;
3. Pick best agent (agent with highest fitness);
4. Perform crossover with best agent and every other agent;
5. Perform mutations for each new child;
6. Back to step 1.

The act of picking the best agent and performing crossover with it and the rest of the population is called elitist crossover, because we're slowly decreasing the diversity on each new generation, but becoming closer to the best neural network.

## A quick demonstration of training a GNN
Let's look at an example with oversimplified neural networks (array of 5 weights). Assume the array that gives us the perfect fitness is **[4, 1, 3, 3, 1]**, and so for each agent, the corresponding fitness is the amount of numbers that were in the correct spot. The objective of training any neural networks is to find these perfect values, the problem is a real neural network can have millions of weights.
1. Create generation with 4 agents (initially with random weights on the neural network);

| Agent | Neural Network |
|-|-|
| 1 | [1, 4, 3, 0, 5] |
| 2 | [3, 2, 0, 0, 3] |
| 3 | [4, 5, 1, 2, 4] |
| 4 | [3, 1, 3, 0, 2] |

2. Test the agents and calculate score and fitness:

| Agent | Neural Network | Score | Fitness
|-|-|-|-|
| 1 | [1, 4, 3, 0, 5] | 1 | 0.20
| 2 | [3, 2, 0, 0, 3] | 0 | 0.00
| 3 | [4, 5, 1, 2, 4] | 1 | 0.20
| 4 | [3, 1, 3, 0, 2] | 2 | 0.40

3. Pick best agent:
   
| Agent | Neural Network | Score | Fitness
|-|-|-|-|
| 4 | [3, 1, 3, 0, 2] | 2 | 0.40

4. Perform crossover with best agent and every other agent:

| Agents | Crossover Rate | New Neural Network
|-|-|-|
| 1 & 4 | 0.4 / (0.4 + 0.2 + 0.01) = 0.66 | [1, 4, 3, 0, 2]
| 2 & 4 | 0.4 / (0.4 + 0.0 + 0.01) = 0.97 | [3, 1, 3, 0, 5]
| 3 & 4 | 0.4 / (0.4 + 0.2 + 0.01) = 0.66 | [4, 1, 3, 0, 4]
| 4 & 4 | 0.4 / (0.4 + 0.4 + 0.01) = 0.49 | [3, 1, 3, 0, 2]

5. Perform mutations for each new child (20%);

| Agent | Neural Network | Mutated Neural Network
|-|-|-|
| 1 | [1, 4, 3, 0, 2] | [1, 4, 1, 0, 2]
| 2 | [3, 1, 3, 0, 5] | [5, 1, 3, 1, 5]
| 3 | [4, 1, 3, 0, 4] | [4, 1, 3, 2, 4]
| 4 | [3, 1, 3, 0, 2] | [3, 5, 3, 0, 2]

6. Back to step 1.

On this example, the solution would be found in few iterations because of the neural networks size, but in a real neural network there would be hundreds, thousands or even millions of weights that could be changed.

This example illustrates some important aspects of training neural networks.
- Bigger populations (or generations) are good because we can test several solutions and have a bigger diversity; the downside is it takes time to test them and many agents can have similar traits which we do not want; it can also take time more to reach the solution comparing to smaller populations;
- The perfect mutation rate can be hard to find and dependes on the problem. We want the mutation to have a significant impact on a neural network, but not too much to the point it forgets learned patterns;
- Crossover is a way to mimic real life reproduction, mixing traits of both parents; in real life the average crossover rate is about 0.5, but here we have the power to adjust the way we want it to, trying to keep alive good traits that are found.

There are many more aspects of training neural networks a little more in-depth, such has the number of layers (and number of nodes on each one of them), activation functions, weights normalization and much more that will not be covered here.
