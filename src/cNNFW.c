/* Copyright (c) 2025 Godov Andrey <andygodov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cNNFW.h>

typedef struct {
    size_t rows;
    size_t cols;
    double **data;
} DATA_TRAIN, *p_DATA_TRAIN;

typedef struct {
    size_t inpLen;
    double *inputs;
} INPUT, *p_INPUTS;

typedef struct {
    double value;
    size_t weiLen;
    double *weights;
} NEURON, *p_NEURON;

typedef struct {
    double bias;
    size_t neuLen;
    p_NEURON neurons;
} LAYER, *p_LAYER;

typedef struct {
    int isChanged;
    ACTIVATION_FUNCTION actFunc;
    EPSILON eps;
    LEARNING_STEP step;
    size_t structureSize;
    INPUT Inps;
    size_t layLen;
    p_LAYER Lays;
    DATA_TRAIN Data;
} PRIVATE, *p_PRIVATE;


int create(N_NET *NNetwork, CONFIG *config, size_t configSize, DATA_ROWS rows) {
    size_t i, j, inp, neu, wei, lay;
    size_t bytes = 0;
    size_t inpBytes = 0, layBytes = 0, neuBytes = 0, weiBytes = 0, dataBytes = 0;
    p_PRIVATE prvt = NULL;

    if (NULL == NNetwork) {
        printf("A pointer to a Neural Network object is NULL\n");
        return 1;
    }

    if (NULL != *NNetwork) {
        printf(
            "The Neural Network object is not NULL."
            "Make sure that the memory it occupies is freed"
            "and assign NULL to the object\n"
        );
        return 1;
    }

    if (2 > configSize) {
        printf("The configuration size cannot be less than 2\n");
        return 1;
    }

    if (1 > rows) {
        printf("The training data must contain at least one row\n");
        return 1;
    }

    if (NULL == config) {
        printf("The pointer to the configuration cannot be NULL\n");
        return 1;
    }

    for (i = 0; i < configSize; i++) {
        if (config[i] < 1) {
            printf("One of the config parameter is less than 1\n");
            return 1;
        }
    }

    bytes = sizeof(PRIVATE);

    inpBytes = sizeof(double) * config[0];
    layBytes = sizeof(LAYER) * (configSize - 1);
    for (lay = 1; lay < configSize; lay++) {
        neuBytes += sizeof(NEURON) * config[lay];
        weiBytes += sizeof(double) * config[lay] * config[lay - 1];
    }

    dataBytes = sizeof(double *) * rows + sizeof(double) * rows * (config[0] + config[configSize - 1]);

    bytes += inpBytes + layBytes + neuBytes + weiBytes + dataBytes;

    prvt = (p_PRIVATE)malloc(bytes);
    if (NULL == prvt) {
        printf("Unsuccessful memory allocation\n");
        return 1;
    }
    prvt->isChanged = 0;

    prvt->structureSize = bytes;
    prvt->Inps.inpLen = config[0];
    prvt->Inps.inputs = (double *)(prvt + 1);
    for (inp = 0; inp < prvt->Inps.inpLen; inp++) {
        prvt->Inps.inputs[inp] = 0.0;
    }

    prvt->layLen = configSize - 1;
    prvt->Lays = (p_LAYER)(prvt->Inps.inputs + prvt->Inps.inpLen);
    for (lay = 0; lay < prvt->layLen; lay++) {
        if (lay < prvt->layLen - 1)
            prvt->Lays[lay].bias = 0.0;
        else
            prvt->Lays[lay].bias = 0.0;

        prvt->Lays[lay].neuLen = config[lay + 1];
        if (0 == lay)
            prvt->Lays[0].neurons = (p_NEURON)(prvt->Lays + prvt->layLen);
        else
            prvt->Lays[lay].neurons = prvt->Lays[lay - 1].neurons + prvt->Lays[lay - 1].neuLen;
    }

    for (lay = 0; lay < prvt->layLen; lay++) {
        for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
            prvt->Lays[lay].neurons[neu].value = 0.0;

            prvt->Lays[lay].neurons[neu].weiLen = config[lay];
            if (0 == lay && 0 == neu)
                prvt->Lays[0].neurons[0].weights = (double *)(prvt->Lays[prvt->layLen - 1].neurons + prvt->Lays[prvt->layLen - 1].neuLen);
            else if (0 == neu)
                prvt->Lays[lay].neurons[0].weights = (double *)(prvt->Lays[lay - 1].neurons[prvt->Lays[lay - 1].neuLen - 1].weights + prvt->Lays[lay - 1].neurons[prvt->Lays[lay - 1].neuLen - 1].weiLen);
            else
                prvt->Lays[lay].neurons[neu].weights = (double *)(prvt->Lays[lay].neurons[neu - 1].weights + prvt->Lays[lay].neurons[neu - 1].weiLen);
        }
    }

    for (lay = 0; lay < prvt->layLen; lay++) {
        for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
            for (wei = 0; wei < prvt->Lays[lay].neurons[neu].weiLen; wei++) {
                prvt->Lays[lay].neurons[neu].weights[wei] = /* 0.5 */(1000.0 - (double)(rand() % 2001)) / 1000.0;
            }
        }
    }

    prvt->Data.rows = rows;
    prvt->Data.cols = config[0] + config[configSize - 1];

    prvt->Data.data = (double **)(prvt->Lays[prvt->layLen - 1].neurons[prvt->Lays[prvt->layLen - 1].neuLen - 1].weights + prvt->Lays[prvt->layLen - 1].neurons[prvt->Lays[prvt->layLen - 1].neuLen - 1].weiLen);

    for (i = 0; i < prvt->Data.rows; i++)
        prvt->Data.data[i] = (double *)(prvt->Data.data + prvt->Data.rows) + i * prvt->Data.cols;

    for (i = 0; i < prvt->Data.rows; i++) {
        prvt->Data.data[i] = (double *)(prvt->Data.data + prvt->Data.rows) + i * prvt->Data.cols;
        for (j = 0; j < prvt->Data.cols; j++) {
            prvt->Data.data[i][j] = 0.0;
        }
    }

    prvt->actFunc = ENABLE;
    prvt->eps = 0.01;
    prvt->step = 0.01;

    *NNetwork = (N_NET)prvt;

    return 0;
}

double ActivationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double difference(N_NET NNetwork) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    size_t i, j, out;
    double result = 0.0;
    double diff = 0.0;
    for (i = 0; i < prvt->Data.rows; i++) {
        for (j = 0; j < prvt->Inps.inpLen; j++) {
            prvt->Inps.inputs[j] = prvt->Data.data[i][j];
        }

        CNNFW_Calculate(NNetwork);

        for (out = 0; out < prvt->Lays[prvt->layLen - 1].neuLen; out++) {
            diff = prvt->Lays[prvt->layLen - 1].neurons[out].value - prvt->Data.data[i][prvt->Inps.inpLen + out];
            result += diff * diff;
        }
    }

    return result / prvt->Data.rows;
}

int CNNFW_Train(N_NET NNetwork) {
    size_t neu, lay, wei;
    double curDiff = 0.0;
    double newDiff = 0.0;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;

    if (NULL == prvt) {
        printf("Neural Network is NULL\n");
        return 1;
    }

    if (NULL == prvt->Data.data) {
        printf("Train data is NULL\n");
        return 1;
    }

    curDiff = difference(NNetwork);

    for (lay = 0; lay < prvt->layLen; lay++) {
        if (lay < prvt->layLen - 1) {
            double tmp = prvt->Lays[lay].bias;
            prvt->Lays[lay].bias += prvt->eps;
            newDiff = difference(NNetwork);
            prvt->Lays[lay].bias = tmp;
            prvt->Lays[lay].bias -= prvt->step * ((newDiff - curDiff) / prvt->eps);
        }
        for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
            for (wei = 0; wei < prvt->Lays[lay].neurons[neu].weiLen; wei++) {
                double tmp = prvt->Lays[lay].neurons[neu].weights[wei];
                prvt->Lays[lay].neurons[neu].weights[wei] += prvt->eps;
                newDiff = difference(NNetwork);
                prvt->Lays[lay].neurons[neu].weights[wei] = tmp;
                prvt->Lays[lay].neurons[neu].weights[wei] -= prvt->step * ((newDiff - curDiff) / prvt->eps);
            }
        }
    }

    prvt->isChanged = 1;

    return 0;
}

int CNNFW_Calculate(N_NET NNetwork) {
    size_t lay, neu, wei;
    double tmp;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }

    for (lay = 0; lay < prvt->layLen; lay++) {
        for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
            tmp = 0.0;
            prvt->Lays[lay].neurons[neu].value = 0.0;
            for (wei = 0; wei < prvt->Lays[lay].neurons[neu].weiLen; wei++) {
                if (lay == 0) {
                    tmp += prvt->Inps.inputs[wei] * prvt->Lays[lay].neurons[neu].weights[wei];
                } else {
                    tmp += prvt->Lays[lay - 1].neurons[wei].value * prvt->Lays[lay].neurons[neu].weights[wei];
                }
            }

            if (lay == prvt->layLen - 1) {
                prvt->Lays[prvt->layLen - 1].neurons[neu].value = tmp;
            } else {
                if (prvt->actFunc == ENABLE) {
                    prvt->Lays[lay].neurons[neu].value = ActivationFunction(tmp + prvt->Lays[lay].bias);
                } else if (prvt->actFunc == DISABLE) {
                    prvt->Lays[lay].neurons[neu].value = tmp + prvt->Lays[lay].bias;
                }
            }
        }
    }

    return 0;
}

int CNNFW_SetActivationFunction(N_NET NNetwork, ACTIVATION_FUNCTION state) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("the pointer to the neural network cannot be NULL\n");
        return 1;
    }

    prvt->actFunc = state;

    return 0;
}

void CNNFW_PrintOutputs(N_NET NNetwork) {
    size_t neu;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("the pointer to the neural network cannot be NULL\n");
    } else {
        printf("\n----------------------------------------------------------------------------------------------------\n");
        printf("Outputs:\n");
        for (neu = 0; neu < prvt->Lays[prvt->layLen - 1].neuLen; neu++)
            printf("  output %lu, value %0.3f\n", (unsigned long)neu, prvt->Lays[prvt->layLen - 1].neurons[neu].value);
        printf("----------------------------------------------------------------------------------------------------\n\n");
    }
}

void CNNFW_Print(N_NET NNetwork) {
    size_t lay, neu, inp, wei;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("the pointer to the neural network cannot be NULL\n");
    } else {
        printf("----------------------------------------------------------------------------------------------------\n\n");

        printf("Inputs %lu:\n", (unsigned long)prvt->Inps.inpLen);
        for (inp = 0; inp < prvt->Inps.inpLen; inp++) {
            printf("    value: %0.3f\n", prvt->Inps.inputs[inp]);
        }
        printf("\n");

        for (lay = 0; lay < prvt->layLen; lay++) {
            if (lay + 1 == prvt->layLen)
                printf("output:\n");
            else
                printf("layer %lu:\n", (unsigned long)lay);

            for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
                printf("    neuron %lu, value %0.3f:\n", (unsigned long)neu, prvt->Lays[lay].neurons[neu].value);
                for (wei = 0; wei < prvt->Lays[lay].neurons[neu].weiLen; wei++) {
                    printf("        weight %lu: %0.3f\n", (unsigned long)wei, prvt->Lays[lay].neurons[neu].weights[wei]);
                }
            }

            printf("    bias: %0.3f\n\n", prvt->Lays[lay].bias);
        }

        printf("----------------------------------------------------------------------------------------------------\n\n");
    }
}

int set_inputs(N_NET NNetwork, double *newInputs, size_t newInpLen) {
    size_t i;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }
    if (NULL == newInputs) {
        printf("New inputs set is NULL\n");
        return 1;
    }
    if (prvt->Inps.inpLen != newInpLen) {
        printf("Neural network inputs and new inputs set have different sizes\n");
        return 1;
    }

    for (i = 0; i < newInpLen; i++) {
        prvt->Inps.inputs[i] = newInputs[i];
    }

    return 0;
}

int CNNFW_SetInput(N_NET NNetwork, size_t index, double value) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }

    if (index > prvt->Inps.inpLen) {
        printf("Index is out of range\n");
        return 1;
    }

    prvt->Inps.inputs[index] = value;

    return 0;
}

int CNNFW_GetOutput(N_NET NNetwork, size_t index, double *retValue) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }

    if (index >= prvt->Lays[prvt->layLen - 1].neuLen) {
        printf("Index is out of range\n");
        return 1;
    }

    *retValue = prvt->Lays[prvt->layLen - 1].neurons[index].value;

    return 0;
}

int CNNFW_Mutation(N_NET NNetwork, unsigned int mutationProbability) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    size_t lay, neu, wei, wlen, mutRnd;

    if (NULL == prvt) {
        printf("A neural network object cannot be NULL\n");
        return 1;
    }

    if (0 == rand() % mutationProbability) {
        for (lay = 0; lay < prvt->layLen; lay++) {
            for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
                wlen = prvt->Lays[lay].neurons[neu].weiLen;

                for (wei = 0; wei < wlen; wei++) {
                    mutRnd = rand() % (prvt->Lays[lay].neuLen * 10);
                    if (0 == mutRnd) {
                        prvt->Lays[lay].neurons[neu].weights[wei] = (1000.0 - (double)(rand() % 2001)) / 1000.0;
                    }
                }
            }
        }
        prvt->isChanged = 1;
    }

    return 0;
}

int CNNFW_WeightsCrossingower(N_NET NNdst, N_NET NNsrc) {
    p_PRIVATE prvtDst = (p_PRIVATE)NNdst;
    p_PRIVATE prvtSrc = (p_PRIVATE)NNsrc;
    size_t lay, neu, wei, rnd, wlen/* , mutRnd */;

    if (NULL == prvtDst || NULL == prvtSrc) {
        printf("A neural network object cannot be NULL\n");
        return 1;
    }

    if (prvtDst->layLen != prvtSrc->layLen) {
        printf("The number of layers of neural networks does not match\n");
        return 1;
    }

    for (lay = 0; lay < prvtDst->layLen - 1; lay++) {
        if (prvtDst->Lays[lay].neuLen != prvtSrc->Lays[lay].neuLen) {
            printf("Each neural network has an uneven number of neurons in layer number %lu\n",
                (unsigned long)lay);
            return 1;
        }
        for (neu = 0; neu < prvtDst->Lays[lay].neuLen; neu++) {
            if (prvtDst->Lays[lay].neurons[neu].weiLen != prvtSrc->Lays[lay].neurons[neu].weiLen) {
                printf("Each neural network has an uneven number of weights in layer number %lu, neuron %lu\n",
                    (unsigned long)lay, (unsigned long)neu);
                return 1;
            }
        }
    }

    for (lay = 0; lay < prvtDst->layLen; lay++) {
        for (neu = 0; neu < prvtDst->Lays[lay].neuLen; neu++) {
            wlen = prvtDst->Lays[lay].neurons[neu].weiLen;
            rnd = rand() % 2;
            for (wei = (wlen / 2) * rnd; wei < wlen - (wlen / 2) * (1 - rnd); wei++) {
                prvtDst->Lays[lay].neurons[neu].weights[wei] = prvtSrc->Lays[lay].neurons[neu].weights[wei];
            }
        }
    }
    prvtDst->isChanged = 1;

    return 0;
}

int CNNFW_SetEpsilonAndLearningStep(N_NET NNetwork, EPSILON eps, LEARNING_STEP step) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;

    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }
    if (0 >= eps) {
        printf("The epsilon cannot be equal to or less than zero\n");
        return 1;
    }
    if (0 >= step) {
        printf("The learning step cannot be equal to or less than zero\n");
        return 1;
    }

    prvt->eps = eps;
    prvt->step = step;

    return 0;
}

int CNNFW_SetValueInData(N_NET NNetwork, DATA_ROWS rowIndex, DATA_COLS colIndex, double value) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;

    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }
    if (NULL == prvt->Data.data) {
        printf("Training data is NULL\n");
        return 1;
    }
    if (prvt->Data.rows <= rowIndex) {
        printf("Row index out of range\n");
        return 1;
    }
    if (prvt->Data.cols <= colIndex) {
        printf("Column index out of range\n");
        return 1;
    }

    prvt->Data.data[rowIndex][colIndex] = value;

    return 0;
}

int CNNFW_GetValueFromData(N_NET NNetwork, DATA_ROWS rowIndex, DATA_COLS colIndex, double *retValue) {
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;

    if (NULL == prvt) {
        printf("Neural network is NULL\n");
        return 1;
    }
    if (NULL == prvt->Data.data) {
        printf("Training data is NULL\n");
        return 1;
    }
    if (NULL == retValue) {
        printf("The pointer to the variable where the value should be stored is NULL\n");
        return 1;
    }
    if (prvt->Data.rows <= rowIndex) {
        printf("Row index out of range\n");
        return 1;
    }
    if (prvt->Data.cols <= colIndex) {
        printf("Column index out of range\n");
        return 1;
    }

    *retValue = prvt->Data.data[rowIndex][colIndex];

    return 0;
}

int CNNFW_SaveToFile(N_NET NNetwork, const char *fileName) {
    FILE *fp = NULL;
    p_PRIVATE prvt = (p_PRIVATE)NNetwork;
    if (NULL == prvt) {
        printf("Neural Network is NULL\n");
        return 1;
    }

    if (prvt->isChanged == 0) {
        printf("The Neural Network has not been changed, so it will not be written to the file\n");
    } else {
        fp = fopen(fileName, "wb");
        if (NULL == fp) {
            printf("Unsuccessful file opening\n");
            return 1;
        }

        if (fwrite(prvt, prvt->structureSize, 1, fp) != 1) {
            printf("Unsuccessful file writting\n");
            fclose(fp);
            return 1;
        }

        fclose(fp);

        prvt->isChanged = 0;
    }

    return 0;
}

int CNNFW_LoadFromFile(N_NET *NNetwork, const char *fileName) {
    size_t lay, neu, i;
    FILE *fp = NULL;
    PRIVATE Prvt = { 0 };
    p_PRIVATE prvt = NULL;

    fp = fopen(fileName, "rb");
    if (NULL == fp) {
        printf("Unsuccessful file opening\n");
        return 1;
    }

    if (fread(&Prvt, sizeof(PRIVATE), 1, fp) != 1) {
        printf("Unsuccessful file reading (1)\n");
        fclose(fp);
        return 1;
    }

    fseek(fp, 0, SEEK_SET);

    prvt = (p_PRIVATE)malloc(Prvt.structureSize);
    if (NULL == prvt) {
        fclose(fp);
        printf("unsuccessful memory allocation\n");
        return 1;
    }
    if (fread(prvt, Prvt.structureSize, 1, fp) != 1) {
        printf("Unsuccessful file reading (2)\n");
        free(prvt);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    prvt->Inps.inputs = (double *)(prvt + 1);

    prvt->Lays = (p_LAYER)(prvt->Inps.inputs + prvt->Inps.inpLen);
    for (lay = 0; lay < prvt->layLen; lay++) {
        if (0 == lay)
            prvt->Lays[0].neurons = (p_NEURON)(prvt->Lays + prvt->layLen);
        else
            prvt->Lays[lay].neurons = prvt->Lays[lay - 1].neurons + prvt->Lays[lay - 1].neuLen;
    }

    for (lay = 0; lay < prvt->layLen; lay++) {
        for (neu = 0; neu < prvt->Lays[lay].neuLen; neu++) {
            if (0 == lay && 0 == neu)
                prvt->Lays[0].neurons[0].weights = (double *)(prvt->Lays[prvt->layLen - 1].neurons + prvt->Lays[prvt->layLen - 1].neuLen);
            else if (0 == neu)
                prvt->Lays[lay].neurons[0].weights = (double *)(prvt->Lays[lay - 1].neurons[prvt->Lays[lay - 1].neuLen - 1].weights + prvt->Lays[lay - 1].neurons[prvt->Lays[lay - 1].neuLen - 1].weiLen);
            else
                prvt->Lays[lay].neurons[neu].weights = (double *)(prvt->Lays[lay].neurons[neu - 1].weights + prvt->Lays[lay].neurons[neu - 1].weiLen);
        }
    }

    prvt->Data.data = (double **)(prvt->Lays[prvt->layLen - 1].neurons[prvt->Lays[prvt->layLen - 1].neuLen - 1].weights + prvt->Lays[prvt->layLen - 1].neurons[prvt->Lays[prvt->layLen - 1].neuLen - 1].weiLen);

    for (i = 0; i < prvt->Data.rows; i++)
        prvt->Data.data[i] = (double *)(prvt->Data.data + prvt->Data.rows) + i * prvt->Data.cols;

    prvt->isChanged = 0;

    *NNetwork = (N_NET)prvt;

    return 0;
}

void CNNFW_Free(N_NET *NNetwork) {
    if (NULL != NNetwork) {
        if (NULL != *NNetwork) {
            free(*NNetwork);
            *NNetwork = NULL;
        }
    }
}