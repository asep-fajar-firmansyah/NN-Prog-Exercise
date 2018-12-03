package com.asep;

import java.util.Arrays;

public class Network {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] error_signal;
    private double[][] output_derivate;

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;
    /*
        This function used for create network dimension
     */
    public Network(int... NETWORK_LAYER_SIZES){
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivate = new double[NETWORK_SIZE][];

        for(int i=0; i < NETWORK_SIZE; i++){
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivate[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], 0.3, 0.7);

            if (i>0){
                weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i-1], 0.3, 0.5);
            }


        }
        for(int x=0; x<output.length; x++){
            //System.out.print(Arrays.toString(output[x]));
            //System.out.println();
        }
    }

    public void train(TrainSet set, int loops, int batch_size){
        for (int i=0; i < loops; i++){
            TrainSet batch = set.extractBatch(batch_size);
            for (int b=0; b < batch_size; b++){
                this.train(batch.getInput(b), batch.getOutput(b), 0.3);
            }
        }
    }

    /*
        Start - Backpropgation purpose
     */
    public void train(double[] input, double[] target, double eta){
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE)
            return;
        calculate(input);
        backpropError(target);
        updateWeight(eta);
    }


    public void backpropError(double[] target){
        for (int neuron=0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron++){
            error_signal[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron]) * output_derivate[NETWORK_SIZE-1][neuron];
        }
        for (int layer = NETWORK_SIZE-2; layer > 0; layer --){
            for (int neuron=0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double sum = 0;
                for (int nextNeuron =0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++){
                    sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer+1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivate[layer][neuron];
            }
        }
    }

    public void updateWeight(double eta){
        for (int layer = 1; layer < NETWORK_SIZE; layer++){
            for (int neuron=0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double delta = - eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++ ){
                    weights[layer][neuron][prevNeuron] += delta * output[layer-1][prevNeuron];
                }

            }
        }
    }
    /*
        End - Backpropagation purpose
     */

    /*
        This function for calculate output based on input to neurons on each network layer.
     */
    public double[] calculate(double... input){
        if(input.length != this.INPUT_SIZE)
            return null;
        this.output[0] = input;
        for (int layer=1; layer < NETWORK_SIZE; layer++){
            for (int neuron=0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double sum = bias[layer][neuron];
                for (int prevNeuron=0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++){
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];

                }
                output[layer][neuron] = sigmoid(sum);
                output_derivate[layer][neuron] = output[layer][neuron] * (1-output[layer][neuron]);
            }
        }
        return output[NETWORK_SIZE-1];
    }
    /*
        This is activation function as sigmoid.
     */
    private double sigmoid(double x){
        return 1d / (1 + Math.exp(-x));
    }
    /*
    // Create dimension and backpropagation demo
    public static void main(String[] args){
        Network net = new Network(4,3,3,2);

        double[] input = new double[]{0.1, 0.2, 0.3, 0.4};
        double[] target = new double[] {0.9, 0.1};

        double[] input2 = new double[]{0.6, 0.1, 0.4, 0.8};
        double[] target2 = new double[] {0.1, 0.9};

        for (int i=0; i<10000; i++){
            net.train(input, target, 0.3);
            net.train(input2, target2, 0.3);
        }

        System.out.println(Arrays.toString(net.calculate(input)));
        System.out.println(Arrays.toString(net.calculate(input2)));
    }
    */

    // Advanced Learning demo
    public static void main(String[] args) {
        Network net = new Network(4, 3, 3, 2);

        TrainSet set = new TrainSet(4, 2);
        set.addData(new double[]{0.1, 0.2, 0.3, 0.4}, new double[]{0.9, 0.1});
        set.addData(new double[]{0.9, 0.8, 0.7, 0.6}, new double[]{0.1, 0.9});
        set.addData(new double[]{0.3, 0.8, 0.1, 0.4}, new double[]{0.3, 0.7});
        set.addData(new double[]{0.9, 0.8, 0.1, 0.2}, new double[]{0.7, 0.3});

        net.train(set, 100000, 1);

        for (int i=0; i<1; i++){
            System.out.println(Arrays.toString(net.calculate(set.getInput(i))));
        }
    }
}
