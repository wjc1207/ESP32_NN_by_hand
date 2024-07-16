# ESP32_NN_by_hand

This project is about how to run the forward process of a Nerual Network on an embedded system.

## Environment

Programming Environment: ESP-IDF (v5.1.1)

Microprocessor: ESP32-S3 N16R8

## Function

1. conv1d layer
2. flatten layer
3. linear layer
4. ReLU layer

## Example

This example donates a simple neural network, including a ```conv1d``` layer and a ```linear``` layer. 

The size of input, output, weights and bias and the model itself are in ```model.h```. 

The ```hello_world_main.c``` file defines the neural network and uses a dummy input data to test the network.

You can modify these two files to define and run your own network.

## Get Start

**Make sure you have downloaded the ESP-IDF and Git before getting start!**

1. download the code
```
git clone https://github.com/wjc1207/ESP32_NN_by_hand.git
```
2. clean the build
```
idf.py fullclean
```
3. build and flash the code to the ESP32
```
idf.py flash -p <port>
```
4. open the monitor to see the results
```
idf.py monitor -p <port>
```

If everything works well, you should see the following output:
```
0.046326
0.121813
0.021277
-0.025692
Time: 8176
```
