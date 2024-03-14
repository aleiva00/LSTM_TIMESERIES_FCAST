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

1. **Time Series Prediction**

2. Natural Language Processing (NLP)

3. Speech Recognition

4. Long-term Strategy Games

5. Sentiment Analysis Over Time

6. Medicine and Health

7. Autonomous Robotics

8. Predictive Maintenance



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

* Utilizes 2 pathways (long and short-term) instead of just 1 like an RNN - after "a corto plazo" at the beginning of lstm.

* The sigmoid function takes a coordinate on the x-axis and converts it to any coordinate on the y-axis between 0 and 1 (giving us the forget gate, the percentage of long-term memory to be discarded).

* The tanh function takes a coordinate on the x-axis and converts it to any coordinate on the y-axis between (-1, 1). When combined with different weights, it gives us the potential long-term memory and short-term memory.

