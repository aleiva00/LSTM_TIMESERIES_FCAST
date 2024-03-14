# Transformation of Time Series for Supervised Learning

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


