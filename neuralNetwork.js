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