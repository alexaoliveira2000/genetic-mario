# Genetic Neural Network
This project was developed as a way for me to learn how to implement feedforward neural networks and how to train it to become expert at a given game. In this document, I show you the game I created and the logic of the NN evolution.

Before we start, in this document I use some terminology that is specific for Genetic Neural Networks, and so here's a list of these terms in case you don't follow:
- Environment: all the information that the game itself provides; it is the context where the agent operates;
- Agent: autonomous entity that interacts with the environment (game); processes inputs from the environment and makes decisions based on learned patterns;

## The Game
I wanted to make a simple game (like an arcade game), where the actions we can make are very limited, aswell as the information we analyse as players. I ended up with thinking Super Mario, but with the difference that the agent cannot walk sideways, which means that the only available options are to *stand*, *jump* and *crouch*.

I do not worry about the aspect of this game, just the mechanics, and so all you're going to see in the game are rectangles:
![image](https://github.com/alexaoliveira2000/genetic-mario/assets/77057098/f3e941af-93ac-4f23-acfa-7cc71a3ad169)

You probably understand it at first glance, but the red rectangle represents the agent, the green one is the floor and the black one is the obstacle. All obstacles are generated randomly, so that the agent doesn't learn patterns that are specific of the path - just useful patterns that can adapt to any map generation. As for the score, each time the agent gets through an obstacle, the score goes up by 2 points, and each time he "almost" goes through the block (doesn't hit it on the left side, but above or below), 1 point is incremented to the score - as a way of indicating that that was almost the correct action. The game ends when the agent reaches a score of 100 (gets through 50 blocks), or when the agent collides with an object.

All rectangles in this game share some properties, such as the coordinates (x and y), sizes (width and height), and movement on screen. This is why I created the class Item:
```` js
class Item {
    constructor(context, x, y, width, height, color) {
        this.context = context;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
    }

    show() {
        this.context.fillStyle = this.color;
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }

    move(dx, dy) {
        this.context.fillStyle = this.color;
        this.x += dx;
        this.y += dy;
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }
}
````

The agent, however, is a bit different. It must have additional methods, such as the brain (neural network), all the actions (stand, jump and crouch), its score and collisions validation (some of the things related to NNs are explained further ahead):
```` js
class Agent extends Item {
    constructor(context, x, y, width, height, color, brain) {
        super(context, x, y, width, height, color);
        this.defaultHeight = height;
        this.velocity = 0;
        this.action = Action.STAND;
        this.score = 0;
        this.fitness = 0;
        if (brain instanceof NeuralNetwork) {
            this.brain = brain.copy();
            this.brain.mutate(0.1);
        } else {
            this.brain = new NeuralNetwork(4, 6, 3);
        }
    }

    calculateFitness() {
        this.fitness = this.score / 100;
    }

    collided(item) {
        if ((this.x >= item.x && this.x <= item.x + item.width) || (this.x + this.width >= item.x && this.x + this.width <= item.x + item.width)) {
            if ((this.y >= item.y && this.y <= item.y + item.height) || (this.y + this.height >= item.y && this.y + this.height <= item.y + item.height)) {
                return true;
            }
        }
        return false;
    }

    move() {
        this.context.fillStyle = this.color;
        if (this.action == Action.JUMP) {
            this.y += this.velocity;
            this.velocity = this.velocity + GRAVITY * TIME
            if (this.y + this.height >= 400) {
                this.action == Action.STAND;
                this.y = 300;
            }
        } else if (this.action == Action.STAND) {
            this.height = this.defaultHeight;
        }
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }

    jump() {
        this.context.fillStyle = this.color;
        if (this.y + this.height < 400 || this.action == Action.CROUCH) {
            return;
        }
        this.action = Action.JUMP;
        this.velocity = -JUMP_FORCE;
        this.y += this.velocity;
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }

    crouch() {
        this.context.fillStyle = this.color;
        if (this.y + this.height != 400)
            return
        this.action = Action.CROUCH;
        this.height = this.defaultHeight / 2;
        this.y = 400 - this.height;
        this.context.fillRect(this.x, this.y, this.width, this.height);
    }

    stand() {
        if (this.y + this.height != 400)
            return
        this.action = Action.STAND;
        this.height = this.defaultHeight;
        this.y = 400 - this.height;
    }
}
````

As for the obstacles (black rectangles), they are put on an array - you can call it a queue. This queue always has 3 obstacles. Each time a block reaches the end of the screen (left side), it is removed from the queue (and also the screen) and a new one is generated and put on the end of the queue. Here are the types of objects that can be generated:
```` js
let obstacleTypes = [
        { y: 300, width: 50, height: 100 },
        { y: 350, width: 50, height: 50 },
        { y: 0, width: 50, height: 280 },
        { y: 0, width: 50, height: 330 }
    ]
````
We can, at any time, add more types of obstacles as so to make the game a little bit harder. Here's the logic of obstacles generation:
```` js
// initially, put 3 random obstacles on the queue
let obstacles = [];
for (let i = 0; i < 3; i++) {
    let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
    let obstacle = new Item(context, x, obstacleType.y, obstacleType.width, obstacleType.height, "black");
    obstacles.push(obstacle);
    obstacle.show();
    x += 500
}

// while the agent hasn't hit an obstacle
while (obstacles.every(obstacle => !agent.collided(obstacle))) {
    // if the most left object is out of screen, remove it (shift) and add a new one at the end of the queue (push)
    if (obstacles[0].x + obstacles[0].width < 0) {
        obstacles.shift()
        let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
        let obstacle = new Item(context, 1400, obstacleType.y, obstacleType.width, obstacleType.height, "black");
        obstacles.push(obstacle)
        points++;
        score.textContent = "Score: " + points;
        agent.score++;
     }
     // move all objects at a given speed (-2 in this case)
     for (const obstacle of obstacles) {
        obstacle.move(-2, 0);
     }
}
````

This is the main code of the game. The remainder is irrelevant to talk about it here (buttons and refresh information on screen).

## The Neural Network
The brain of the agent is this neural network. It consists of a class that implements a dense neural network and performs some operations related to it. The input layer can be seen as each information that the agent receives, and the output layer all the available actions. The hidden layer can have any number of nodes as we'd like (tipically, if the NN has one hidden layer, x inputs and y outputs, it is recommended that we start with (x+y)/2 nodes in the hidden layer).

Briefly analysing the game and all the information it provides, I think there's about 4 relevant informations we must give the agent - the visual properties of the next obstacle: x, y, width and height. If the game sped up through time, we probably would have to feed the speed of the obstacle aswell.


