#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cNNFW.h>

#define GET_ARRAY_SIZE(x) sizeof(x)/sizeof(x[0])

#define EPOCHS 100000

#define NUM_OF_INPUTS 2
#define NUM_OF_NEURONS_IN_LAYERS 3
#define NUM_OF_OUTPUTS 6

#define NUM_OF_DATA_ROWS 4
#define NUM_OF_DATA_COLS NUM_OF_INPUTS+NUM_OF_OUTPUTS

int main(int argc, char *argv[]) {
    int quit;
    DATA_ROWS row, col;
    size_t i;
    const char *fileName = "parameters.bin";

    N_NET NNetwork = NULL;

    /*  ____|XOR|AND| OR|~XOR|~AND|~OR|
        0 0 | 0 | 0 | 0 |  1 |  1 | 1 |
        0 1 | 1 | 0 | 1 |  0 |  1 | 0 |
        1 0 | 1 | 0 | 1 |  0 |  1 | 0 |
        1 1 | 0 | 1 | 1 |  1 |  0 | 0 |
    Two inputs and six outputs */
    double d[NUM_OF_DATA_ROWS][NUM_OF_DATA_COLS] = {
        {0.0, 0.0,   0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
        {0.0, 1.0,   1.0, 0.0, 1.0, 0.0, 1.0, 0.0},
        {1.0, 0.0,   1.0, 0.0, 1.0, 0.0, 1.0, 0.0},
        {1.0, 1.0,   0.0, 1.0, 1.0, 1.0, 0.0, 0.0}
    };

    /* Configuring. Two inputs, one hidden layer with 3 neurons, six outputs */
    CONFIG config[] = { NUM_OF_INPUTS, NUM_OF_NEURONS_IN_LAYERS, NUM_OF_OUTPUTS };

    /* Reading a neural network from a file. If the file does
        not exist, a new neural network will be created */
    if (0 == CNNFW_LoadFromFile(&NNetwork, fileName)) {
        printf("Loading Neural Network from file\n");
    } else {
        printf("Creating a new Neural Network\n");
        /* Creating a Neural Network */
        if (CNNFW_Create(&NNetwork, config, NUM_OF_DATA_ROWS)) {
            printf("Error of Neural Network creating\n");
            return 1;
        }

        /* By default, the epsilon and learning step are set to 0.01.
        You can set other values for epsilon and learning step */
        /* if (CNNFW_SetEpsilonAndLearningStep(NNetwork, 0.1, 0.001)) {
            printf("error of setting epsilon and learning step values\n");
            return 1;
        } */

        /* The activation function is enabled by default, but you can disable it */
        /* if (CNNFW_SetActivationFunction(NNetwork, DISABLE)) {
            printf("Error of enabling activation function\n");
            return 1;
        } */

        /* Writing data for training to an Neural Network */
        printf("Writting data\n");
        for (row = 0; row < NUM_OF_DATA_ROWS; row++) {
            for (col = 0; col < NUM_OF_DATA_COLS; col++) {
                if (CNNFW_SetValueInData(NNetwork, row, col, d[row][col])) {
                    printf("Error of setting data\n");
                    return 1;
                }
            }
        }

        /* Checking the values of the written data */
        printf("Data:\n");
        for (row = 0; row < NUM_OF_DATA_ROWS; row++) {
            for (col = 0; col < NUM_OF_DATA_COLS; col++) {
                double tmp = -1.0;
                if (0 == CNNFW_GetValueFromData(NNetwork, row, col, &tmp)) {
                    printf("%.0f ", tmp);
                }
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Training */
    for (i = 0; i <= EPOCHS; i++) {
        if (EPOCHS >= 100)
            if (i % (EPOCHS / 100) == 0) printf("\rLerning: %lu%%", (unsigned long)i / (EPOCHS / 100));

        if (CNNFW_Train(NNetwork)) {
            printf("Error of training\n");
            return 1;
        }
    }

    /* Printing of all neural network parameters */
    CNNFW_Print(NNetwork);

    /* Testing */
    quit = 0;
    while (1) {
        double out[NUM_OF_OUTPUTS] = { 0.0 };       /* We will save the values of the outputs here */
        double newInputs[NUM_OF_INPUTS] = { 0.0 };  /* Here we will write the new values of the inputs */

        printf("Your inputs (input < 0 or input > 1 to quit)\n");
        for (i = 0; i < NUM_OF_INPUTS; i++) {
            printf("Input %lu: ", (unsigned long)i + 1);
            scanf("%lf", &newInputs[i]);
            if (newInputs[i] < 0 || newInputs[i] > 1) {
                quit = 1;
                break;
            }
            if (CNNFW_SetInputs(NNetwork, newInputs)) {
                printf("Error writing values to the input");
                quit = 1;
                break;
            }
        }
        if (quit) break;

        /* We give the command to the neural network to recalculate the outputs */
        CNNFW_Calculate(NNetwork);

        /* CNNFW_Print(NNetwork); */

        printf("\n____|XOR|AND| OR|~XOR|~AND|~OR|\n");

        /* We get the values of the outputs */
        for (i = 0; i < NUM_OF_OUTPUTS; i++)
            CNNFW_GetOutput(NNetwork, i, &out[i]);

        printf("%d %d | %d | %d | %d |  %d |  %d | %d |\n\n",
            (int)newInputs[0],
            (int)newInputs[1],
            (out[0] >= 0.5),
            (out[1] >= 0.5),
            (out[2] >= 0.5),
            (out[3] >= 0.5),
            (out[4] >= 0.5),
            (out[5] >= 0.5));
    }

    /* Saving all the parameters of the neural network to a file */
    if (CNNFW_SaveToFile(NNetwork, fileName))
        printf("Unsuccessful saving to file\n");
    else
        printf("Saving Neural Network to file\n");

    /* Freeing up the memory occupied by the neural network */
    CNNFW_Free(&NNetwork);

    printf("\nDone\n");

    return 0;
}