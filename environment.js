// environment constants
const GRAVITY = 0.05;
const JUMP_FORCE = 5;
const TIME = 1;
const SPEED = 2;

// all types of obstacles that can be generated
const obstacleTypes = [
    { id: 1, y: 300, width: 50, height: 100 },
    { id: 2, y: 350, width: 50, height: 50 },
    //{ id: 3, y: 0, width: 50, height: 280 },
    { id: 4, y: 0, width: 50, height: 330 }
]

// html elements
let canvas, context, score, generationText, agentText, playButton, resetButton, trainButton, trainTable, modelCanvas, modelContext, trainGenerations, trainAgents, trainNodes, trainRate, trainCrossover, trainCrossoverYes, trainCrossoverNo;

// boolean values indicating if game is playing or training
let playing = false;
let training = false;

// function to check if an agent has collided with any of the given obstacles
let hasCollided = function (agent, obstacles) {
    return obstacles.some(obstacle => agent.collided(obstacle));
}

// function to create a number of random obstacles
let createObstacles = function (number) {
    let result = []
    for (let x = 900; x <= 900 + 500 * (number - 1); x += 500) {
        let obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
        let obstacle = new Item(context, x, obstacleType.y, obstacleType.width, obstacleType.height, "black");
        result.push(obstacle);
    }
    return result;
}

// function to update the given obstacles array
let updateObstacles = function (agent, obstacles) {
    if (obstacles[0].x + obstacles[0].width < 0) {
        obstacles.shift()
        let obstacleType;
        do {
            obstacleType = obstacleTypes[Math.floor(Math.random() * obstacleTypes.length)];
        } while (obstacleType.id == obstacles[0].id && obstacleType.id == obstacles[0].id);
        let obstacle = new Item(context, 1400, obstacleType.y, obstacleType.width, obstacleType.height, "black");
        obstacles.push(obstacle)
        agent.score += 2;
    }
    return obstacles;
}

// function to update all game environment
let updateGame = function (agent, obstacles, number) {
    // update queue of obstacles
    obstacles = updateObstacles(agent, obstacles);

    // update labels
    score.textContent = "Score: " + agent.score;
    generationText.textContent = "Generation: " + agent.generation;
    agentText.textContent = "Agent: " + number;

    // update game canvas
    context.clearRect(0, 0, 1000, 400);
    agent.move()
    for (const obstacle of obstacles)
        obstacle.move(-SPEED, 0)
}

// function to update the shown neural network
let updateModel = function (context, model, prediction, inputs) {
    // function to draw a node at a given position
    let drawNode = function (x, y) {
        context.beginPath();
        context.arc(x, y, 10, 0, 2 * Math.PI);
        context.fillStyle = "blue";
        context.fill();
        context.closePath();
    }

    // function to draw a weight at a given position
    let drawWeight = function (x1, y1, x2, y2, weight) {
        context.beginPath();
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.strokeStyle = "black";
        context.lineWidth = weight;
        context.stroke();
        context.closePath();
    }

    // function to draw an output prediction and label
    let drawOutputLabel = function (x, y, probability, text) {
        context.font = `18px sans-serif`;
        context.fillStyle = "black";
        context.fillText(probability, x, y);
        context.fillText(text, x + 50, y);
    }

    let outputLabels = ["stand", "jump", "crouch"];
    let weights = model.getWeights().filter(weights => weights.name.includes("kernel"));
    let inputWeights = weights[0].arraySync();
    let outputWeights = weights[1].arraySync();

    let layer1Position = 120;
    let layer2Position = 225 - 25 * outputWeights.length;
    let layer3Position = 150;

    context.clearRect(0, 0, 800, 700);

    // input nodes
    for (let i = 1; i <= 4; i++) {
        drawNode(100, layer1Position + 50 * i)
        // weights from input layer to hidden layer
        for (let j = 1; j <= outputWeights.length; j++) {
            let weight = inputWeights[i - 1][j - 1] + 1 + inputs[i - 1];
            drawWeight(100, layer1Position + 50 * i, 350, layer2Position + 50 * j, weight)
        }
    }
    // hidden nodes
    for (let i = 1; i <= outputWeights.length; i++) {
        drawNode(350, layer2Position + 50 * i)
        // weights from hidden layer to output layer
        for (let j = 1; j <= 3; j++) {
            let weight = outputWeights[i - 1][j - 1] + 1 + prediction[j - 1];
            drawWeight(350, layer2Position + 50 * i, 600, layer3Position + 50 * j, weight)
        }
    }
    // output nodes
    for (let i = 1; i <= 3; i++) {
        drawNode(600, layer3Position + 50 * i)
        drawOutputLabel(630, layer3Position + 5 + 50 * i, prediction[i - 1].toFixed(2), outputLabels[i - 1])
    }
}

// function to create a new table for a given generation
let createGenerationTable = function (generation) {
    let generationDiv = document.createElement("div");
    let generationTitle = document.createElement("h4");
    let generationTable = document.createElement("table");
    let header = document.createElement("tr");
    let headerAgent = document.createElement("th");
    let headerScore = document.createElement("th");
    let headerFitness = document.createElement("th");
    generationDiv.style.marginRight = "32px";
    generationDiv.style.marginBottom = "10px";
    generationTable.style.border = "1px solid";
    generationTitle.textContent = "Generation " + generation;
    headerAgent.textContent = "Agent";
    headerScore.textContent = "Score";
    headerFitness.textContent = "Fitness";
    header.appendChild(headerAgent);
    header.appendChild(headerScore);
    header.appendChild(headerFitness);
    generationTable.appendChild(header);
    generationDiv.appendChild(generationTitle);
    generationDiv.appendChild(generationTable);
    trainTable.appendChild(generationDiv);
    return generationTable;
}

// add information to the table about the given agent
let addGenerationTable = function (table, agent, number) {
    let row = document.createElement("tr");
    let tdAgent = document.createElement("td");
    let tdScore = document.createElement("td");
    let tdFitness = document.createElement("td");
    tdAgent.textContent = number;
    tdScore.textContent = agent.score;
    tdFitness.textContent = agent.fitness;
    row.appendChild(tdAgent);
    row.appendChild(tdScore);
    row.appendChild(tdFitness);
    table.appendChild(row);
}

let predictedAction = function (agent, prediction) {
    let actionIndex = 0;
    for (let i = 1; i < prediction.length; i++) {
        if (prediction[i] > prediction[actionIndex]) {
            actionIndex = i;
        }
    }
    switch (actionIndex) {
        case 0: agent.stand();
            break
        case 1: agent.jump();
            break
        case 2: agent.crouch();
            break
    }
}

let checkScore = function (agent, obstacles) {
    let closestObject = obstacles[0];
    if (agent.x >= closestObject.x + 5 && agent.x <= closestObject.x + closestObject.width - 5) {
        agent.score++;
    }
    if (agent.x + agent.width >= closestObject.x + 5 && agent.x + agent.width <= closestObject.x + closestObject.width - 5) {
        agent.score++;
    }
}

let normalizeInputs = function (inputs) {
    const min = Math.min(...inputs);
    const max = Math.max(...inputs);
    const normalizedInputs = inputs.map((value) => (value - min) / (max - min));
    return normalizedInputs;
}

var reset = function () {
    // initialize variables
    playing = false;
    training = false;
    playButton.disabled = false;
    trainButton.disabled = false;

    // clear screen and reset labels
    context.clearRect(0, 0, canvas.width, canvas.height);
    score.textContent = "Score: 0";
    generationText.textContent = "Generation: 0";
    agentText.textContent = "Agent: 0";

    // draw start screen
    context.fillStyle = "black";
    context.font = `50px sans-serif`;
    context.fillText("Genetic Mario", 300, 220);
    context.font = `20px sans-serif`;
    context.fillText("Select an option to start", 350, 270);
}

var play = async function () {
    // change environment variables
    playing = true;
    training = false;
    playButton.disabled = true;
    trainButton.disabled = true;

    // generate objects
    let floor = new Item(context, 0, 400, 1000, 100, "green");
    let agent = new Agent(context, 100, 300, 50, 100, "red");
    let obstacles = createObstacles(3);
    floor.show();

    // player keyboard actions
    document.addEventListener("keydown", function (event) {
        if (!playing)
            return
        switch (event.key) {
            case "ArrowUp": agent.jump()
            case "ArrowDown": agent.crouch()
        }
    });
    document.addEventListener("keyup", function (event) {
        if (!playing)
            return
        switch (event.key) {
            case "ArrowDown": agent.stand();
        }
    });

    while (playing && !hasCollided(agent, obstacles)) {
        updateGame(agent, obstacles);
        await new Promise(r => setTimeout(r, TIME));
    }
}

var train = async function () {

    // check inputs
    trainGenerations = document.getElementById("train_generations");
    trainAgents = document.getElementById("train_agents");
    trainNodes = document.getElementById("train_nodes");
    trainRate = document.getElementById("train_rate");
    trainCrossoverYes = document.getElementById("train_crossover_yes");
    trainCrossoverNo = document.getElementById("train_crossover_no");
    if (trainGenerations.value < 1 || trainGenerations.value > 100)
        return
    if (trainAgents.value < 1 || trainAgents.value > 50)
        return
    if (trainNodes.value < 4 || trainNodes.value > 10)
        return
    if (trainRate.value < 0.00 || trainRate.value > 0.99)
        return
    let doCrossover = trainCrossoverYes.checked;

    // disable inputs
    trainGenerations.disabled = true;
    trainAgents.disabled = true;
    trainNodes.disabled = true;
    trainRate.disabled = true;
    trainCrossoverYes.disabled = true;
    trainCrossoverNo.disabled = true;

    // change environment variables
    playing = false;
    training = true;
    playButton.disabled = true;
    resetButton.disabled = true;
    trainButton.disabled = true;

    let population = new Population(trainAgents.value, context, trainRate.value, parseInt(trainNodes.value));
    for (let generation = 1; generation <= trainGenerations.value; generation++) {
        let generationTable = createGenerationTable(generation);
        for (let agentNumber = 1; agentNumber <= trainAgents.value; agentNumber++) {
            let floor = new Item(context, 0, 400, 1000, 100, "green");
            let agent = population.members[agentNumber - 1];
            let obstacles = createObstacles(3);
            floor.show();
            while (training && !hasCollided(agent, obstacles) && agent.score < 30) {
                // make prediction
                let closestObject = obstacles[0];
                let inputs = [closestObject.x, closestObject.y, closestObject.width, closestObject.height];
                inputs = normalizeInputs(inputs);
                let prediction = agent.brain.predict(inputs);
                predictedAction(agent, prediction)

                // update environment
                updateModel(modelContext, agent.brain.model, prediction, inputs)
                updateGame(agent, obstacles, agentNumber);

                // wait between frames
                await new Promise(r => setTimeout(r, TIME));
            }
            checkScore(agent, obstacles);
            agent.calculateFitness();
            addGenerationTable(generationTable, agent, agentNumber);
        }
        if (doCrossover)
            population.performCrossover();
        else
            population.nextGeneration();
    }
}

window.onload = function () {
    canvas = document.getElementById("canvas");
    context = canvas.getContext("2d");
    score = document.getElementById("score");
    generationText = document.getElementById("generation");
    agentText = document.getElementById("agent");
    playButton = document.getElementById("play");
    resetButton = document.getElementById("reset");
    trainButton = document.getElementById("train");
    trainTable = document.getElementById("train_table");
    modelCanvas = document.getElementById("model");
    modelContext = modelCanvas.getContext("2d");

    playButton.addEventListener("click", () => play());
    resetButton.addEventListener("click", () => reset());
    trainButton.addEventListener("click", () => train());

    reset()
}