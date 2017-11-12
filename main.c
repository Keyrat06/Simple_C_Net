#include <stdio.h>
#include <stdlib.h>
#include "FFNN.h"

int main(int argc, char *argv[])
{
    
    /* Input and expected out data for the XOR function. */
    const double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double targets[4] = {0, 1, 1, 0};
    int i;
    
    FFNN *net = FFNN_init(2, 1, 10, 1, 3.0);
    
    for (i = 0; i < 300; ++i) {
        FFNN_backprop(net, inputs[0], targets + 0);
        FFNN_backprop(net, inputs[1], targets + 1);
        FFNN_backprop(net, inputs[2], targets + 2);
        FFNN_backprop(net, inputs[3], targets + 3);
    }
    
    /* Run the network and see what it predicts. */
    printf("Output for [%1.f, %1.f] is %1.f.\n", inputs[0][0], inputs[0][1], *FFNN_run(net, inputs[0]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", inputs[1][0], inputs[1][1], *FFNN_run(net, inputs[1]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", inputs[2][0], inputs[2][1], *FFNN_run(net, inputs[2]));
    printf("Output for [%1.f, %1.f] is %1.f.\n", inputs[3][0], inputs[3][1], *FFNN_run(net, inputs[3]));
    
    free(net);
    return 0;
}
