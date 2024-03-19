# CRC/USD FX rate: Multivariate one-step forecasting with LSTM model
This repo houses my final project for a machine learning course at TEC de Costa Rica. Titled 'Time Series Forecasting with LSTM Model' it showcases how to solve a forecasting problem for time series with recurrent neural networks. Although time-constrained, it offers a primer for more sophisticated models and there's ample room for improvement. This is joint work with Cristian Cubero.

Please find below a concise overview of our project's key findings and methodologies. The full project details can be accessed in the Jupyter notebook provided in [this folder](<Code/Proyecto Final - ML.ipynb>).



# Table of Contents

- [1. What is LSTM](<#1-what-is-lstm>): high level explanation and examples.
- [2. Preprocessing](#2-pre-processing): handling missing values and building a rectangular matrix for next steps.
- [3. Time series to supervised learning problem](#3-transformation-of-time-series-for-supervised-learning): transform lags into features.
- [4. Some steps before building the model](#4-some-steps-before-building-the-model): get the data ready and split into train, val and test.
- [5- Model](#5-model): Build the model, train it, see what happens.
- [6- Results and final remarks](#6-results-and-final-remarks): couple lessons from this flash project.


# Data (and some disclaimers)


The variables used were selected according to the following criteria:

1. Short-term variables that motivate the preference for buying or selling dollars by both the public and the Non-Banking Public Sector (e.g., pension funds).
2. Daily frequency to have a larger volume of data and because of having a forecasting horizon of 1 day.

Observations:

* The **data selection can be improved**, for example, by using existing economic literature. As the scope of the project is about the use of ML techniques, no time was invested in literature reviews.
* For the purposes of this work, although variables were selected with certain economic rationale, the **priority is to look for patterns in the very short-term decisions** of the agents. This makes variables related to fundamental factors (e.g., imports, exports, tourism, FDI, etc.) not so relevant for the considered time horizon.
* In addition to the previous point, **fundamental variables were discarded due to their frequency and lag** in publication --> A fundamental challenge in the discipline that has led to the emergence of initiatives such as nowcasting with real-time Google data, satellite data with parking lots of shopping centers, electricity consumption data, etc.


## Variable Dictionary

| Variable              | Description                                  | Source     |
|-----------------------|----------------------------------------------|------------|
| MONEX_VOL             | Volume traded in MONEX                             | BCCR  |
| TC_AVGMONEX           | Average exchange rate MONEX                |  BCCR   |
| COMPRAS_VENTANILLAS   | Dollar purchases at teller windows                       | BCCR  |
| VENTAS_VENTANILLAS    | Dollar sales at teller windows                        | BCCR  |
| PRIME                 | US Prime Rate                                        | BCCR  |
| 6M_TREASURYBILLS     | 6-month Treasury Bills                  | BCCR  |
| 10Y_TREASURYBILLS    | 10-year Treasury Bills                  | BCCR   |
| 2Y_TREASURYBILLS     | 2-year Treasury Bills                   |  BCCR   |
| 5Y_TREASURYBILLS     | 5-year Treasury Bills                   |  BCCR   |
| 30Y_TREASURYBILLS    | 30-year Treasury Bills                  | BCCR  |
| TED                   | Effective Dollar Rate                                          | BCCR |
| TBP                   | Passive Basic Rate                                          | BCCR  |
| EMBI                  | Emerging Markets Bond Index (risk premium vs US)                                         | Bloomberg |


Now, lets get into it.


# 1. What is LSTM


## Neural Network
* Network composed of nodes called neurons that work together to learn patterns.
* Each neuron processes information and passes its result to the following ones, forming layers.
* The network "learns" by adjusting the weights between connections to improve accuracy in its predictions (Goodfellow et al., 2016).


* <img src="https://www.investopedia.com/thmb/5-hnhHpOzLM2GVXPlSstg8tJYLw=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Neural_Network_Apr_2020-01-5f4088dfda4c49d99a4d927c9a3a5ba0.jpg" alt="drawing" width="400" />

* **Example**: Passing a regular NN the letter sequence: 'L I G A' and then 'C A M P E Ó N'.
  * When processing the second sequence 'CAMPEÓN', it already forgot about 'LIGA'.
  * Even when processing the letter 'I' in the first sequence, it already forgot about 'L'.
  * That's a problem because no matter how much it's trained, it will always find it difficult to guess the next most probable sequence: 'CAMPEÓN'.
  * This makes it a rather poor candidate for certain tasks, such as speech recognition, which benefit from the ability to predict what will come next.
  * **Conclusion**: Regular NNs have no memory, and although they serve for certain tasks, if memory is important, they will have poor results.

## Recurrent Neural Network (RNN)

* NN that has feedback connections that allow it to remember previous information. It's useful for processing data sequences, such as text or time series (Cho et al., 2014).
* ![RNN Image](https://miro.medium.com/v2/resize:fit:553/0*xs3Dya3qQBx6IU7C.png)

* **Example**: Passing an RNN the letter sequence: 'S A P R I S S A' and then 'C A M P E Ó N'.

  * The unit, or artificial neuron, of the RNN, upon receiving the sequence 'CAMPEÓN', also takes as input the sequence it received a moment ago 'SAPRISSA'.
  * It adds the immediate past to the present.
  * This gives it the advantage of a limited short-term memory that, along with its training, provides enough context to guess which is more likely to be the next sequence: 'CAMPEÓN'.
  * **Colloquial Conclusion**: Regular RNNs have memory, and that gives them a boost to predict what's coming.



## What kinds of problems benefit from having memory?

Some examples include: 1. **Time Series Prediction**, 2. Natural Language Processing (NLP) 3. Speech Recognition, 4. Long-term Strategy Games, 5. Sentiment Analysis Over Time, 6. Medicine and Health, 7. Autonomous Robotics, 8. Predictive Maintenance



# Long Short-Term Memory (LSTM)

LSTM is a variant of RNN designed to overcome the problem of short-term forgetting.

1. **Memory Cell:**
   - Acts as a long-term memory, remembering or forgetting information based on received signals.

2. **Gates:**
   - **Forget Gate:** Decides which information from the memory cell should be discarded.
   - **Input Gate:** Determines what new information should be added to the memory cell.
   - **Output Gate:** Filters the output information based on the memory cell.

3. **Memory Cell Update:**
   - The memory cell is updated by multiplying the current information by the forget gate and adding the new information multiplied by the input gate.

4. **Output:**
   - The output is filtered using the output gate based on the updated memory cell.

**LSTM Parameters:**
- Weights and biases associated with forget, input, and output gates.
- Activation functions, commonly the sigmoid and hyperbolic tangent functions.

Visual representation:

![LSTM Architecture](https://databasecamp.de/wp-content/uploads/lstm-architecture-1024x709.png)

* Utilizes 2 pathways (long and short-term) instead of just 1 like an RNN - after "a short term" at the beginning of lstm.

* The sigmoid function takes a coordinate on the x-axis and converts it to any coordinate on the y-axis between 0 and 1 (giving us the forget gate, the percentage of long-term memory to be discarded).

* The tanh function takes a coordinate on the x-axis and converts it to any coordinate on the y-axis between (-1, 1). When combined with different weights, it gives us the potential long-term memory and short-term memory.



# 2. Pre-processing

1. The database corresponds to several time series in which there are more data available for some compared to others --> this requires "matching" the series in order to process them as matrices.

    ![Matching](Emparejar.png)

2. Since the exchange rate is not traded during weekends and holidays, there will be NaNs. Also, due to the way the BCCR reports the exchange rate, there may be NaNs or 0's. Here, these NaNs could be either removed or filled with the data from the previous day. In this case, to avoid losing observations, they will be kept with data from the previous day. Moreover, it is reasonable to do this since the exchange rate over the weekend at teller windows remains the last available rate.

    ![alt text](weekends.png)




# 3. Transformation of Time Series for Supervised Learning

The "features" of a time series often include lagged periods of itself and other lagged time series that may influence the main time series. For this reason, in order for a supervised learning model to consider these lags as features, they must be in columns. That is:

We start with a time series:

| Time   | Value |
|--------|-------|
|   1    |  10   |
|   2    |  15   |
|   3    |  20   |
|   4    |  25   |
|   5    |  30   |

Now, let's create a table showing the time series with lags of 1, 2, and 3 periods:

| Time   | Value | Value(t-1) | Value(t-2) | Value(t-3) |
|--------|-------|------------|------------|------------|
|   1    |  10   |      -     |      -     |      -     |
|   2    |  15   |     10     |      -     |      -     |
|   3    |  20   |     15     |     10     |      -     |
|   4    |  25   |     20     |     15     |     10     |
|   5    |  30   |     25     |     20     |     15     |

This transformation converts the time series into a supervised learning problem, where the current value (t) can be predicted based on past values.


## Function to Automatically Perform This

We create a function that receives the parameters:

* data: time series
* n_in: lags to predict t
* n_out: number of predictions or outputs with n_lags

And returns:

* data in supervised learning format

Function taken from:
* https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/



# 4. Some steps before building the model

1. **Load model data and double-check format**

2. **Normalize features**

   MinMaxScaler from scikit-learn is used to scale the features to the range (0, 1). Normalization is important to ensure that all features have the same scale and facilitate model training.

3. **Transform to supervised learning**

   See [Transformation of Time Series for Supervised Learning](#3-transformation-of-time-series-for-supervised-learning). Output is:


   ![alt text](SERIES_TO_SUPERVISED.png)

4. **Drop columns we do not want to predict.**

    Note that we only kept one variable in time t which is our target variable.

5. **Train, validation, and test set split**

   - Train: 70% of the data
   - Validation: 15%
   - Test: 15%


    ![alt text](EXP1_TRAIN_VAL_TEST.png)


6. **Reshape training, validation, and test sets for multivariate problem**

    You can read each parenthesis as -> (n_obs, periods to forecast, no_features) (n_obs, periods to forecast, no_target_var)

   ![alt text](RESHAPE_FEATURES.png)

   Graphical distribution of data in train, validation, and test sets. 
   

   ![alt text](EXP1_TRAIN_VAL_TEST_DIST.png)

   Note that in our target var the data is distributed significantly different in the train, val and test periods which is an important challenge to train the model. As stated at the beginning, this has a lot of room for improvement. Worth to say that we ran more experiments than I am showing here in the jupyter notebook. Feel free to check it out.


# 5. Model

   - **Loss function**: RMSE
   - **Activation function**: linear --> It was found that for forecasting (regression), it is common to use this activation function

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense
   from tensorflow.keras.optimizers import RMSprop, Adam
   import tensorflow as tf

   tf.random.set_seed(123)
   tf.config.experimental.enable_op_determinism()

   N_UNITS = 128 # Size of hidden state (h) and memory cell (c) (128)
   INPUT_SHAPE = (X_train.shape[1], X_train.shape[2]) # (rows/dates) x 12 (features)

   # Hyperparameters

   learn_rate = 0.0005  # 5e-4 before
   EPOCHS = 500 # Hyperparameter
   BATCH_SIZE = 256 # Hyperparameter

   model = Sequential()

   model.add(LSTM(N_UNITS, input_shape=INPUT_SHAPE))

   model.add(Dense(OUTPUT_LENGTH, activation='linear')) # activation = 'linear' because we want to predict (regression)

   # Loss function
   def root_mean_squared_error(y_true, y_pred):
       rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred-y_true)))
       return rmse

   # Compilation
   optimizer = RMSprop(learning_rate= learn_rate)
   model.compile(
       optimizer = optimizer,
       loss = root_mean_squared_error,
   )
   ```

## Training

![Training Animation](https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/a258b2108677535.5fc364926e4a7.gif)

### Evaluation

We use 500 epochs and see that from around 50 the error drops and converges to zero.

   ![Loss Graph](EXP1_LOSS.png)



# 6. Results and Final Remarks

The model scores look quite good. The fit is good, and although it increases in the test period, it does not do so drastically. Although it can certainly be improved, it's not bad.

![Model Scores](EXP1_SCORES.png)


Finally, after rescaling the data (returning to our original magnitude), we can evaluate the forecasts vs the observed time series in reality.

![Forecast vs Observed](EXP1_FORECASTS.png)

Not bad for a first experiment!

## Final Remarks

* Converting the time series forecasting problem to a supervised learning problem involves a different data preparation.
* Despite varying the training, validation, and test dates, the model consistently exhibits low errors.
* The model is complex to understand and build, but especially challenging to explain. This represents a significant obstacle for its application in daily practice, where accountability to third parties may be necessary.
* Something good to do if more time is available: fine-tuning with hyperparameters (e.g., grid search, etc.).
* This model **has limited scope for more everyday problems**; forecasting the exchange rate for the next day may be of little use to most people. Despite this limitation, this practical exercise, which is restricted by time, **serves as a solid foundation** for adding complexity, steps, and generating forecasts over a longer time horizon.














# Resources
## Data

1. MONEX Daily Statistics, Central Bank of Costa Rica.
2. Foreing and Domestic Rates, Central Bank of Costa Rica.
3. CRI Emerging Markets Bond Index, JP Morgan/Bloomberg through Ivenomica.


## Practical Resources

1.   https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
2.   https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

3.  ["Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras" por Jason Brownlee](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
4. ["Understanding LSTM Networks" por Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
5. ["Deep Learning for Time Series Forecasting" por Jason Brownlee](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)
6. ["Sequence to Sequence Learning with Neural Networks" por Ilya Sutskever, Oriol Vinyals, and Quoc V. Le (2014)](https://arxiv.org/abs/1409.3215)


## Bibliographic Resources

1. Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.
2. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
4. Graves, A., Mohamed, A. R., & Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing (pp. 6645-6649). IEEE.
