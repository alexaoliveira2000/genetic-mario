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
                    if (Math.random() < crossoverProbability) {
                        let temp = nodesWeightsModel1[k];
                        nodesWeightsModel1[k] = nodesWeightsModel2[k];
                        nodesWeightsModel2[k] = temp;
                    }
                tensorsModel1[i] = tf.tensor(nodesWeightsModel1);
                tensorsModel2[i] = tf.tensor(nodesWeightsModel2);
            } else {
                // for each node connections
                for (let j = 0; j < nodesWeightsModel1.length; j++) {
                    let nodeWeightsModel1 = nodesWeightsModel1[j];
                    let nodeWeightsModel2 = nodesWeightsModel2[j];
                    // for each weight
                    for (let k = 0; k < nodeWeightsModel1.length; k++)
                        if (Math.random() < crossoverProbability) {
                            let temp = nodeWeightsModel1[k];
                            nodeWeightsModel1[k] = nodeWeightsModel2[k];
                            nodeWeightsModel2[k] = temp;
                        }
                }
                tensorsModel1[i] = tf.tensor2d(nodesWeightsModel1);
                tensorsModel2[i] = tf.tensor2d(nodesWeightsModel2);
            }
        }
        this.model.setWeights(tensorsModel1);
        model.setWeights(tensorsModel2);
        return model;
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