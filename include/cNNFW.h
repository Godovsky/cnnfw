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


#ifndef CCNNFW_H
#define CCNNFW_H

/* Values DISABLE or ENABLE for the activation function */
typedef enum {
    DISABLE, ENABLE
} ACTIVATION_FUNCTION;

/* The object of the Neural Network */
typedef void *N_NET;

/* The type of Neural Network configuration */
typedef unsigned int CONFIG;

/* The type of the number of rows in the training data */
typedef size_t DATA_ROWS;

/* The type of the number of columns in the training data */
typedef size_t DATA_COLS;

/* The step for calculating the gradient numerically */
typedef double EPSILON;

/* The Neural Network training step */
typedef double LEARNING_STEP;


/** Creating a neural network using the specified parameters in the config array.
* Use the CNNFW_Create(config) macro to create a neural network to avoid errors
* with configuration size. The epsilon and learning step are set to 0.01 by default for each,
* use CNNFW_SetEpsilonAndLearningStep function to set other values. By default,
* the activation function is enabled, use CNNFW_SetActivationFunction to change
*
* @param   NNetwork Neural Network object
* @param   config   Array of neural network configuration.
*                   The first element is always responsible
*                   for the number of inputs, the last element
*                   is always responsible for the number of outputs,
*                   everything in between is responsible for
*                   the number of hidden layers and the number of
*                   neurons in them.
*                   For example, {2, 4, 5, 3}:
*                   the neural network will have 2 inputs,
*                   there are three outputs,
*                   the first hidden layer has 4 neurons,
*                   the second hidden layer has 5 neurons
* @param   rows     Number of rows in the data. The number of data
*                   columns is taken from the neural network
*                   configuration and it is equal to the number of
*                   inputs plus the number of outputs
* @return           Pointer to neural network. NULL in case of an error
*/
#define CNNFW_Create(NNetwork, config, rows) create((NNetwork), (config), sizeof((config))/sizeof((config)[0]), (rows))
/** Use the CNNFW_Create(config) macro to create a neural network to avoid errors
* with configuration size */
int create(N_NET *NNetwork, CONFIG *config, size_t configSize, DATA_ROWS rows);


/** The function of enabling or disabling the activation function
*
* @param    NNetwork    Neural Network object
* @param    state       The state is ENABLE or DISABLE
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_SetActivationFunction(N_NET NNetwork, ACTIVATION_FUNCTION state);


/** Creating an object that will store data for training
*
* @param   rows    Number of rows of data
* @param   cols    The number of columns in each row of data
*
* @return          A pointer to an object with training data. NULL in case of an error
*/
/* DATASET CNNFW_CreateData(DATASET *Data, NUM_OF_ROWS rows, NUM_OF_COLS cols); */


/** Writes new input values to the neural network object.
* Use the CNNFW_SetInputs(NNetwork, newInputs) macro to avoid errors
* with array size
*
* @param   NNetwork     Neural Network object
* @param   newInputs    A pointer to an array with new data
*
* @return               0 in case of success, 1 in case of error
*/
#define CNNFW_SetInputs(NNetwork, newInputs) set_inputs((NNetwork), (newInputs), sizeof((newInputs))/sizeof((newInputs)[0]))
/** Use the CNNFW_SetInputs(NNetwork, newInputs) macro to avoid errors
* with array size
*/
int set_inputs(N_NET NNetwork, double *newInputs, size_t newInpLen);


/** Writes a new value to a specific position in the training data
*
* @param    NNetwork    Neural Network object
* @param    rowIndex    The index of the row in which you want to set a specific value
* @param    colIndex    The index of the column in which you want to set a specific value
* @param    value       The written value
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_SetValueInData(N_NET NNetwork, DATA_ROWS rowIndex, DATA_COLS colIndex, double value);


/** Reads a value from a specific position in the training data
*
* @param    NNetwork    Neural Network object
* @param    rowIndex    The index of the row in which you want to set a specific value
* @param    colIndex    The index of the column in which you want to set a specific value
* @param    retValue    The pointer by which the value will be saved
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_GetValueFromData(N_NET NNetwork, DATA_ROWS rowIndex, DATA_COLS colIndex, double *retValue);


/** Writes a new input value to one specific input of the Neural Network object
*
* @param    NNetwork    Neural Network object
* @param    index       The index of the input to be written
* @param    value       The written value
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_SetInput(N_NET NNetwork, size_t index, double value);


/** Neural network training. One call to this function is equal to one epoch
*
* @param    NNetwork    Neural Network object
* @param    Data        Data object to training
* @param    eps         Epsilon traditionally denotes a small value for
*                       numerical approximations
* @param    step        The learning step
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_Train(N_NET NNetwork);


/** Takes a specific output value from a Neural Network object
*
* @param    NNetwork    Neural Network object
* @param    index       The index of the output to be taken
* @param    retValue    The pointer by which the value of the selected output will be saved
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_GetOutput(N_NET NNetwork, size_t index, double *retValue);


/** Experimental parameter "mutation" function for the implementation of a genetic algorithm
*
* @param    NNetwork            Neural Network object
* @param    mutationProbability The index of the output to be taken
*
* @return                       0 in case of success, 1 in case of error
*/
int CNNFW_Mutation(N_NET NNetwork, unsigned int mutationProbability);


/** Experimental function for crossing-over to implement a genetic algorithm
*
* @param    NNdst   Destination Neural Network object
* @param    NNsrc   Source Neural Network object
*
* @return           0 in case of success, 1 in case of error
*/
int CNNFW_WeightsCrossingower(N_NET NNdst, N_NET NNsrc);


/** Sets values for epsilon and learning step
*
* @param    NNetwork    Neural Network object
* @param    eps         Epsilon traditionally denotes a small value for
*                       numerical approximations
* @param    step        The learning step
*
* @return               0 in case of success, 1 in case of error
*/
int CNNFW_SetEpsilonAndLearningStep(N_NET NNetwork, EPSILON eps, LEARNING_STEP step);


/** Calculation of the input parameters of the neural network and
* saving all the results in neurons in each hidden and output layers.
* The values of the weights do not change
*
* @param   NNetwork    Neural Network object
*
* @return              0 in case of success, 1 in case of error
*/
int CNNFW_Calculate(N_NET NNetwork);


/** Displaying all the values of the Neural Network object
*
* @param    NeuralNetwork   Neural Network object
*/
void CNNFW_Print(N_NET NeuralNetwork);


/** Displaying the output values of the Neural Network object
*
* @param    NeuralNetwork    Neural Network object
*/
void CNNFW_PrintOutputs(N_NET NNetwork);


/** Saving the entire Neural Network object with all its parameters to a fileName file
*
* @param   NNetwork    Neural Network object
* @param   fileName    The path to the file where the Neural Network will be saved
*
* @return              0 in case of success, 1 in case of error
*/
int CNNFW_SaveToFile(N_NET NNetwork, const char *fileName);


/** Loading the Neural Network object with all its parameters from a fileName file
*
* @param   NNetwork    Neural Network object
* @param   fileName    The path to the file from which the Neural Network will be loaded
*
* @return              0 in case of success, 1 in case of error
*/
int CNNFW_LoadFromFile(N_NET *NNetwork, const char *fileName);


/** Frees up the memory allocated for the Neural Network object
*
* @param    NeuralNetwork   A pointer to Neural Network object
*/
void CNNFW_Free(N_NET *NNetwork);

#endif /* CCNNFW_H */