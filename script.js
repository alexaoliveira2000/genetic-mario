const GRAVITY = 0.05
const JUMP_FORCE = 5
const TIME = 1
var START = false

const Action = {
    STAND: 0,
    JUMP: 1,
    CROUCH: 2
}

class NeuralNetwork {
    constructor(inputs, hiddenUnits, outputs, model = {}) {
        this.input_nodes = inputs;
        this.hidden_nodes = hiddenUnits;
        this.output_nodes = outputs;

        if (model instanceof tf.Sequential) {
            this.model = model;

        } else {
            this.model = this.createModel();
        }
    }

    // Copy a model
    copy() {
        return tf.tidy(() => {
            const modelCopy = this.createModel();
            const weights = this.model.getWeights();
            const weightCopies = [];
            for (let i = 0; i < weights.length; i++) {
                weightCopies[i] = weights[i].clone();
            }
            modelCopy.setWeights(weightCopies);
            return new NeuralNetwork(this.input_nodes, this.hidden_nodes, this.output_nodes, modelCopy);
        });
    }

    randomGaussian() {
        let mean = 0;
        let stdDev = 1;

        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return num * stdDev + mean;
    }

    mutate(rate) {
        tf.tidy(() => {
            const weights = this.model.getWeights();
            const mutatedWeights = [];
            for (let i = 0; i < weights.length; i++) {
                let tensor = weights[i];
                let shape = weights[i].shape;
                let values = tensor.dataSync().slice();
                for (let j = 0; j < values.length; j++) {
                    if (Math.random(1) < rate) {
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

    dispose() {
        this.model.dispose();
    }

    predict(inputs) {
        return tf.tidy(() => {
            const xs = tf.tensor2d([inputs]);
            const ys = this.model.predict(xs);
            const output = ys.dataSync();
            return output;
        });
    }

    createModel() {
        const model = tf.sequential();
        const hiddenLayer = tf.layers.dense({
            units: this.hidden_nodes,
            inputShape: [this.input_nodes],
            activation: "relu"
        });
        model.add(hiddenLayer);
        const outputLayer = tf.layers.dense({
            units: this.output_nodes,
            activation: "softmax"
        });
        model.add(outputLayer);
        return model;
    }
}

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

class Population {
    constructor(members) {
        this.size = members.length;
        this.members = members;
    }

    nextGeneration() {
        this.normalizeFitness();
        this.members = this.generate();
    }

    // generate a new population (a more refined one)
    generate() {
        let newMembers = [];
        for (let i = 0; i < this.members.length; i++) {
            let member = this.poolSelection(this.members);
            newMembers[i] = member;
        }
        this.members = newMembers;
        return newMembers;
    }

    // After all population attempts, normalize score
    normalizeFitness() {
        for (let i = 0; i < this.members.length; i++) {
            this.members[i].score = Math.pow(this.members[i].score, 2);
        }

        let sum = 0;
        for (let i = 0; i < this.members.length; i++) {
            sum += this.members[i].score;
        }

        for (let i = 0; i < this.members.length; i++) {
            this.members[i].fitness = this.members[i].score / sum;
        }
    }

    // Select one member of the population (based on the score)
    poolSelection() {
        let pool = [];
        this.members.forEach((member) => {
            let fitness = Math.floor(member.fitness * 100) || 1;
            for (let i = 0; i < fitness; i++) {
                pool.push(member);
            }
        });
        let selectedMember = pool[Math.floor(Math.random() * pool.length)];
        return new Agent(selectedMember.context, selectedMember.x, selectedMember.y, selectedMember.width, selectedMember.height, selectedMember.color, selectedMember.brain);
    }

}

let updateModel = function(canvas, context, model, prediction) {

    context.clearRect(0, 0, canvas.width, canvas.height);

    let layerWeights = model.getWeights().filter(weights => weights.name.includes("kernel"));
    let inputWeights = layerWeights[0].arraySync();
    let outputWeights = layerWeights[1].arraySync();

    let layer1y = 170
    let layer2y = 125
    let layer3y = 200

    let outputLabels = ["stand", "jump", "crouch"];

    // input nodes
    for (let i = 1; i <= 4; i++) {
        context.beginPath();
        context.arc(100, layer1y + 50*i, 10, 0, 2 * Math.PI);
        context.fillStyle = "blue";
        context.fill();
        context.closePath();
        for (let j = 1; j <= 6; j++) {
            context.beginPath();
            context.moveTo(100, layer1y + 50*i);
            context.lineTo(350, layer2y + 50*j);
            context.strokeStyle = "black";
            context.lineWidth = inputWeights[i-1][j-1] + 1 + prediction[j-1];
            context.stroke();
            context.closePath();
        }
    }
    // hidden nodes
    for (let i = 1; i <= 6; i++) {
        context.beginPath();
        context.arc(350, layer2y + 50*i, 10, 0, 2 * Math.PI);
        context.fillStyle = "blue";
        context.fill();
        context.closePath();
        for (let j = 1; j <= 3; j++) {
            context.beginPath();
            context.moveTo(350, layer2y + 50*i);
            context.lineTo(600, layer3y + 50*j);
            context.strokeStyle = "black";
            context.lineWidth = outputWeights[i-1][j-1] + 1 + prediction[j-1];
            context.stroke();
            context.closePath();
        }
    }
    // output nodes
    for (let i = 1; i <= 3; i++) {
        context.beginPath();
        context.arc(600, layer3y + 50*i, 10, 0, 2 * Math.PI);
        context.fillStyle = "blue";
        context.fill();
        context.closePath();
        context.font = `18px sans-serif`;
        context.fillStyle = "black";
        context.fillText(prediction[i-1].toFixed(2), 630, layer3y + 5 + 50*i);
        context.fillText(outputLabels[i-1], 680, layer3y + 5 + 50*i);
    }
}

let createEnvironment = async function () {
    let canvas = document.getElementById("canvas");
    let context = canvas.getContext("2d");
    let score = document.getElementById("score");
    let startButton = document.getElementById("start");
    let resetButton = document.getElementById("reset");
    let trainButton = document.getElementById("train");
    let trainTable = document.getElementById("train_table");
    let modelCanvas = document.getElementById("model");
    let modelContext = modelCanvas.getContext("2d");
    let generationText = document.getElementById("generation");
    let agentText = document.getElementById("agent");
    let floor, agent, obstacles, points;
    let obstacleTypes = [
        { y: 300, width: 50, height: 100 },
        { y: 350, width: 50, height: 50 },
        { y: 0, width: 50, height: 280 },
        { y: 0, width: 50, height: 330 }
    ]

    let reset = function () {
        START = false;
        startButton.disabled = false;
        context.clearRect(0, 0, canvas.width, canvas.height);
        obstacles = [];
        points = 0;
        score.textContent = "Score: " + points;
        floor = new Item(context, 0, 400, 1000, 100, "green");
        agent = new Agent(context, 100, 300, 50, 100, "red");
        floor.show()
        agent.show()
        let x = 900
        for (let i = 0; i < 3; i++) {
            let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
            let obstacle = new Item(context, x, obstacleType.y, obstacleType.width, obstacleType.height, "black");
            obstacles.push(obstacle);
            obstacle.show();
            x += 500
        }
    }

    let start = async function () {
        START = true;
        startButton.disabled = true;
        while (START && obstacles.every(obstacle => !agent.collided(obstacle))) {
            context.clearRect(0, 0, 1000, 400);
            if (obstacles[0].x + obstacles[0].width < 0) {
                obstacles.shift()
                let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
                let obstacle = new Item(context, 1400, obstacleType.y, obstacleType.width, obstacleType.height, "black");
                obstacles.push(obstacle)
                points++;
                score.textContent = "Score: " + points;
                agent.score++;
            }
            for (const obstacle of obstacles) {
                obstacle.move(-1, 0);
            }
            agent.move();
            await new Promise(r => setTimeout(r, TIME));
        }
    }

    document.addEventListener("keydown", function (event) {
        if (!START)
            return
        if (event.key === "ArrowUp") {
            console.log("Jump");
            agent.jump();
        } else if (event.key === "ArrowDown") {
            console.log("Crouch");
            agent.crouch();
        }
    });

    document.addEventListener("keyup", function (event) {
        if (!START)
            return
        if (event.key === "ArrowDown") {
            console.log("Stand");
            agent.stand();
        }
    });

    startButton.addEventListener("click", function () {
        if (!START)
            start();
    })

    resetButton.addEventListener("click", function () {
        if (START)
            reset();
    })

    trainButton.addEventListener("click", async function () {

        let agentsThreshold = 10;
        let agentNumber = 0;

        let members = [];
        for (let i = 0; i < agentsThreshold; i++) {
            members.push(new Agent(context, 100, 300, 50, 100, "red"));
        }
        let population = new Population(members);

        let generation = 0;
        while (generation < 50) {
            generation++;
            let tableDiv = document.createElement("div");
            let generationTable = document.createElement("h4");
            generationTable.innerHTML = "Generation " + generation
            let table = document.createElement("table")
            let tr = document.createElement("tr")
            let header1 = document.createElement("td")
            let header2 = document.createElement("td")
            let header3 = document.createElement("td")
            header1.textContent = "Agent"
            header2.textContent = "Score"
            header3.textContent = "Fitness"
            tr.appendChild(header1);
            tr.appendChild(header2);
            tr.appendChild(header3);
            tableDiv.appendChild(generationTable);
            tableDiv.appendChild(table);
            table.appendChild(tr);
            trainTable.appendChild(tableDiv);
            generationText.textContent = "Generation: " + generation;
            for (const member of population.members) {
                agentNumber++;
                agentText.textContent = "Agent: " + agentNumber;
                agent = member;
                agent.show();
                if (agent.x + agent.height != 400) {
                    reset();
                    agent.show();
                }
                let closestObject;
                while (obstacles.every(obstacle => !agent.collided(obstacle))) {
                    if (agent.score == 100)
                        break

                    context.clearRect(0, 0, 1000, 400);

                    closestObject = obstacles[0];
                    let inputs = [closestObject.x, closestObject.y, closestObject.width, closestObject.height];
                    const min = Math.min(...inputs);
                    const max = Math.max(...inputs);
                    const normalizedInputs = inputs.map((value) => (value - min) / (max - min));
                    let prediction = agent.brain.predict(normalizedInputs);

                    updateModel(modelCanvas, modelContext, agent.brain.model, prediction)

                    if (prediction[1] > prediction[0] & prediction[0] > prediction[2]) {
                        if (agent.action == Action.CROUCH) {
                            agent.stand();
                        } else {
                            agent.jump();
                        }
                    } else if (prediction[2] > prediction[1] & prediction[2] > prediction[0]) {
                        agent.crouch()
                    } else {
                        agent.stand()
                    }

                    // if (prediction[0] == bestGuess) {
                    //     agent.stand()
                    // } else if (prediction[0] == bestGuess) {
                    //     if (agent.action == Action.CROUCH) {
                    //         agent.stand();
                    //     } else {
                    //         agent.jump();
                    //     }
                    // } else if (prediction[2] == bestGuess) {
                    //     agent.crouch()
                    // }

                    if (closestObject.x + closestObject.width < 0) {
                        obstacles.shift()
                        let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
                        let obstacle = new Item(context, 1400, obstacleType.y, obstacleType.width, obstacleType.height, "black");
                        obstacles.push(obstacle);
                        obstacle.show();
                        agent.score += 2;
                        score.textContent = "Score: " + agent.score;
                    }

                    for (const obstacle of obstacles) {
                        obstacle.move(-2, 0);
                    }

                    agent.move();
                    await new Promise(r => setTimeout(r, TIME));
                }

                if (agent.x >= closestObject.x + 5 && agent.x <= closestObject.x + closestObject.width - 5) {
                    agent.score++;
                }
                if (agent.x + agent.width >= closestObject.x + 5 && agent.x + agent.width <= closestObject.x + closestObject.width - 5) {
                    agent.score++;
                }

                agent.calculateFitness();
                let tr = document.createElement("tr");
                let td1 = document.createElement("td");
                let td2 = document.createElement("td");
                let td3 = document.createElement("td");
                td1.textContent = agentNumber;
                td2.textContent = agent.score;
                td3.textContent = agent.fitness;
                tr.appendChild(td1);
                tr.appendChild(td2);
                tr.appendChild(td3);
                table.appendChild(tr);

                // if (agent.fitness == 1)
                //     return

                reset();

            }
            agentNumber = 0;
            population.nextGeneration();
        }
    })
    reset();
}

window.onload = createEnvironment;