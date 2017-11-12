#include "FFNN.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
double sigmoid(double a) { if (a < -45.0) return 0; if (a > 45.0) return 1; return 1.0 / (1 + exp(-a));}
FFNN *FFNN_init(int num_inputs, int num_hidden, int hidden_size, int num_outputs, double learning_rate){
    const int num_weights = (num_inputs+1)*hidden_size + (hidden_size+1)*(hidden_size)*(num_hidden-1) + (hidden_size+1)*num_outputs;
    const int num_neurons = (num_inputs + hidden_size * num_hidden + num_outputs);
    FFNN *net = malloc(sizeof(FFNN)+sizeof(double)*(num_weights + num_neurons + (num_neurons-num_inputs)));
    srand((unsigned) time(NULL));
    net->num_inputs = num_inputs;
    net->num_hidden = num_hidden;
    net->hidden_size = hidden_size;
    net->num_outputs = num_outputs;
    net->num_weights = num_weights;
    net->num_neurons = num_neurons;
    net->weights = (double*)((char*)net + sizeof(FFNN));
    net->outputs = net->weights + net->num_weights;
    net->dlayer = net->outputs + net->num_neurons;
    net->learning_rate = learning_rate;
    for (int i = 0; i < net->num_weights; ++i) {
        net->weights[i] = (((double)rand())/RAND_MAX) - 0.50;
    }
    return net;
}
double const *FFNN_run(FFNN const *net, double const *inputs){
    double const *w = net->weights;
    double *o = net->outputs + net->num_inputs;
    double const *i = net->outputs;
    memcpy(net->outputs, inputs, sizeof(double)*net->num_inputs);
    for (int h=0; h <= net->num_hidden; ++h){
        for (int j=0; j < (h < net->num_hidden ? net->hidden_size : net->num_outputs); ++j){
            double sum = *w++ * 1.0;
            for (int k=0; k < (h == 0 ? net->num_inputs : net->hidden_size); ++k){
                sum += *w++ * i[k];
            }
            *o++ = sigmoid(sum);
        }
        i += (h == 0 ? net->num_inputs : net->num_hidden);
    }
    return net->outputs + net->num_inputs + net->num_hidden*net->hidden_size;
}
void FFNN_backprop(FFNN const *net, double const *inputs, double const *target_outputs){
    FFNN_run(net, inputs);
    for (int h = net->num_hidden; h>=0; --h){
        double const *o = net->outputs + net->num_inputs + (h * net->hidden_size);
        double *d = net->dlayer + (h * net->hidden_size);
        if (h == net-> num_hidden){
            double const *t = target_outputs; /* First desired output. */
            for (int j = 0; j < net->num_outputs; ++j) {
                *d++ = (*o - *t) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
        else{
            double const * const dnext = net->dlayer + ((h+1) * net->hidden_size);
            double const * const w_to_next = net->weights + ((net->num_inputs+1) * net->hidden_size) + ((net->hidden_size+1) * net->hidden_size * (h));
            for (int j = 0; j < net->hidden_size; ++j) {
                double delta = 0;
                for (int k = 0; k < (h == net->num_hidden-1 ? net->num_outputs : net->hidden_size); ++k) {
                    delta += dnext[k] * w_to_next[k * (net->hidden_size + 1) + (j + 1)];
                }
                *d = *o * (1.0-*o) * delta;
                ++d; ++o;
            }
        }
    }
    for (int h = net->num_hidden; h >= 0; --h) {
        double const *d = net->dlayer + (h * net->hidden_size);
        double *w = net->weights + (h ? ((net->num_inputs+1) * net->hidden_size + (net->hidden_size+1) * (net->hidden_size) * (h-1)) : 0);
        double const *i = net->outputs + (h ? (net->num_inputs + net->hidden_size * (h-1)) : 0);
        for (int j = 0; j < net->hidden_size; ++j) {
            for (int k = 0; k < (h == 0 ? net->num_inputs : net->hidden_size) + 1; ++k) {
                *w++ -= (k ? *d * net->learning_rate * i[k-1] : *d * net->learning_rate * 1.0);
            }
            ++d;
        }
    }
}
