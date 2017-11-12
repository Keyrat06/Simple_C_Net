#ifndef FFNN_h
#define FFNN_h
#include <stdio.h>
typedef struct FFNN{
    int num_inputs, num_hidden, hidden_size, num_outputs;
    int num_weights;
    int num_neurons;
    double *weights;
    double *outputs;
    double *dlayer;
    double learning_rate;
} FFNN;
FFNN *FFNN_init(int num_inputs, int num_hidden, int hidden_size, int num_outputs, double learning_rate);
double const *FFNN_run(FFNN const *net, double const *inputs);
void FFNN_backprop(FFNN const *net, double const *inputs, double const *target_outputs);
#endif /* FFNN_h */
