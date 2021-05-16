#Stock_Prediction

#IMPORT
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#IMPORT THE DATA FILE, REMOVE THE 'DATE' COLUMN FROM DATASET
dataset = pd.read_csv('data_stocks.csv')
dataset = dataset.drop(['DATE'], 1)

#DATASET VARIABLES, 'num_data' = NUMBER OF DATA POINTS, 'num_const' = NUMBER OF STOCKS
num_data = dataset.shape[0]
num_const = dataset.shape[1]

#MAKE DATASET np.array
dataset = dataset.values

#SPLIT DATASET INTO 80% FOR TRAINING DATA AND 20% FOR TESTING DATA
#TRAINING DATA, 80%
traindata = dataset[np.arange(0, int(np.floor(0.8*num_data))), :]
#TESTING DATA, 20%
testdata = dataset[np.arange(int(np.floor(0.8*num_data))+1, num_data), :]

#SCALE DATASET USING MinMaxScaler WITH VALUES BEING IN THE RANGE OF (-1,1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(traindata)

#SCALE BOTH THE TRAINING AND THE TESTING DATASET
traindata = scaler.transform(traindata)
testdata = scaler.transform(testdata)

#SPLIT DATASET INTO X AND y TRAIN AND TEST
# X_train, X_test, etc = are arrays
X_train = traindata[:, 1:]
X_test = testdata[:, 1:]
y_train = traindata[:, 0]
y_test = testdata[:, 0]

#RETREIVING NUMBER OF STOCKS IN SPLIT TRAINING DATASET
num_stocks = X_train.shape[1]

#SETTING UP TensorFlow MODEL, ABSTRACT REPRESENTATION OF NN THROUGH placeholders AND variables
#   placeholders, 'ins' = inputs (stock prices of all S&P 500 stocks), 'outs' = outputs (stock price of the S&P 500)
#   variables, layeri_neurons = number of neurons on layer i, layeri_weight = weight for layer i, layeri_bias = bias for layer i

# layeri_neurons, -----TRY OUT DIFFERENT NUMBER OF NEURONS-----
layer1_neurons = 2000  # double input size
layer2_neurons = 1000  # 50% of previous layer
layer3_neurons = 500   # 50% of previous layer
layer4_neurons = 250   # 50% of previous layer

session = tf.InteractiveSession()

# placeholders
ins = tf.placeholder(dtype=tf.float32, shape=[None, num_stocks])
outs = tf.placeholder(dtype=tf.float32, shape=[None])
### WEIGHT AND BIAS INITIALIZERS using default initialization strategy ###
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
bias_initializer = tf.zeros_initializer()
############################
# layeri_weight, layeri_bias
layer1_weight = tf.Variable(weight_initializer([num_stocks, layer1_neurons]))
layer1_bias = tf.Variable(bias_initializer([layer1_neurons]))
layer2_weight = tf.Variable(weight_initializer([layer1_neurons, layer2_neurons]))
layer2_bias = tf.Variable(bias_initializer([layer2_neurons]))
layer3_weight = tf.Variable(weight_initializer([layer2_neurons, layer3_neurons]))
layer3_bias = tf.Variable(bias_initializer([layer3_neurons]))
layer4_weight = tf.Variable(weight_initializer([layer3_neurons, layer4_neurons]))
layer4_bias = tf.Variable(bias_initializer([layer4_neurons]))
output_weight = tf.Variable(weight_initializer([layer4_neurons, 1]))
output_bias = tf.Variable(bias_initializer([1]))
#NN ARCHITECTURE AND ACTIVATION FUNCTION (ReLU) -----TRY OUT DIFFERENT ACTIVATION FUNCTIONS-----
layer1 = tf.nn.relu(tf.add(tf.matmul(ins, layer1_weight), layer1_bias))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, layer2_weight), layer2_bias))
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, layer3_weight), layer3_bias))
layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, layer4_weight), layer4_bias))
layer_output = tf.transpose(tf.add(tf.matmul(layer4, output_weight), output_bias))

#ERROR ANALYSIS FUNCTION, Measure of deviation of predictions and actual using Mean Squared Error
MSE = tf.reduce_mean(tf.squared_difference(layer_output, outs))
#OPTIMISER RATE TO DECREASE THE MSE, using Adaptive Moment Estimation Optimizer (default for deep learning dev)
MSE_dec = tf.train.AdamOptimizer().minimize(MSE)

#SETTING UP NN SESSION AND PLOT
session.run(tf.initializers.global_variables()) #initialise global variables in plot
plt.ion() #turning on interactive mode
graph = plt.figure() #create new plot
grid_param = graph.add_subplot(111) #subplot grid parameter
real_line, = grid_param.plot(y_test)
pred_line, = grid_param.plot(y_test * 0.5)
plt.show()


#num_in_batch = 256 


#TRAINING WITH DIFFERENT SIZED BATCHES FOR EACH EPOCH
for epoch in range(10):
    #GENERATE SHUFFLED TRAINING DATA
    size = len(y_train)
    batch_range = size //256
    random = np.random.permutation(np.arange(size))
    X_train = X_train[random]
    y_train = y_train[random]
    for x in range(0, batch_range):
        #TRAIN AND RUN THE BATCH AND MINIMIZE MSE
        X_batch = X_train[(256*x):((256*x)+256)]
        Y_batch = y_train[(256*x):((256*x)+256)]
        session.run(MSE_dec, feed_dict={ins:X_batch, outs:Y_batch})

        #DISPLAY PLOT EVERY 50th BATCH
        if(np.mod(x, 50) == 0):
            #RUN A PREDICTION ON THE DATA
            prediction = session.run(layer_output, feed_dict={ins: X_test})
            pred_line.set_ydata(prediction)
            plt.pause(0.01)


#DISPLAY MSE AND TEST SCORE ACCURACY FOR TEST DATA
MSE_test = session.run(MSE, feed_dict={ins:X_test, outs: y_test})
print("---------------------------------------------------")
print("\tMSE for test data: ", MSE_test)
print("\tAccuracy on test data: ", 1-MSE_test)
