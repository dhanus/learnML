# How to learn machine learning 
I've had a lot of people ask me how to get started with machine learning and/or deep learning. This is a list of some of the resources that I have either found useful myself or heard people who I trust rave about.  

## General Machine Learning 

**Classes**

* [Coursera's Machine Learning Class](https://www.coursera.org/learn/machine-learning) - This Coursera class, taught by Andrew Ng, seems to have become the canonical introductory machine learning class everyone uses. Most data scientists I know have at some point used this class to study for interviews. 

**Reading**

* [Machine Learning by Chris Bishop](http://www.springer.com/gb/book/9780387310732) - Bishop's book is a common introductory machine learning textbook. While I know some people who have learned machine learning simply by reading this text, I think that it can be a bit thick if it is your first introduction to machine learning, but is a wonderful reference once you have a better idea of how things fit together.  

* [Machine Learning by Kevin Murphy](https://mitpress.mit.edu/books/machine-learning-0) - Murphy's book is another common introductor machine learning textbook. This is also a wonderful reference but is a bit hard to read cover to cover. 

**Programming**

* [SciKitLearn](http://scikit-learn.org/stable/) - SciKitLearn is a Python library for machine learning. Most of the tools provided in it are written with the aim of being usable for those with minimal machine learning background. With this goal in mind, the documentation often contains nice resources for describing heuristics or intuiton to better understand the machine learning behind the library. 

## Deep Learning 

**Classes**

* If you want an intuition for what deep learning is and how it works, [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) has a series of YouTube videos that explain this really well! 

    * [Chapter 1 - What is a Neural Network?](https://www.youtube.com/watch?time_continue=7&v=aircAruvnKk)
    * [Chapter 2 - How Neural Networks Learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
    * [Chapter 3 - Back propagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

* If you want to build something that uses deep learning, [Fast.ai](http://www.fast.ai/) is an online course that will get you using deep learning for practical projects within just a few lessons. 

**Reading**

* [Neural Networks & Deep Learning](http://neuralnetworksanddeeplearning.com/) - This online book explains a lot of hard concepts relatively intuitively. 

* [Deep Learning Book](http://www.deeplearningbook.org/) - This book (availale online and in print) seems to have become one of the canonical books on deep learning. It starts with background knowledge and continues on through modern deep learning research. I thought that part I did a marvelous job reviewing the probability, linear algebra, and other background knowledge that is most useful to get going in deep learning. The explanations of deep learning were a good review if you've seen them before, and the final section on modern research could get a bit confusing at times.

**Programming**

* [Keras](https://keras.io/) - Keras is a high-level neural networks API, written in Python. You can think of it as a wrapper around TensorFlow (and other lower level tools), Theano, etc. If you want to get something that uses deep learning up and running quickly, Keras is a great library to use. However, if you need to do a lot of customization to your architecture, there is a good chance that you will end up needing to use some of the lower level tools (i.e. TensorFlow) too. 

* [Tensorflow](https://www.tensorflow.org/)- Tensfor Flow is a library for implementing deep learning algorithms produced by Google Brain. It 

* [PyTorch](http://pytorch.org/)

**Visualizations**

* [TensorFlow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.63097&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## Reinforcement Learning 

**Reading**

* [Reinforcement Learning by Sutton & Barto](https://mitpress.mit.edu/books/reinforcement-learning) - This is the canonical book on reinforcement learning, and it has been for quite some time.  Consequently, this will get you through the basic ideas of reinforcement learning, but to learn about the most modern advances, you'll beed another resource. During my Master's, I learned reinforcement learning by reading this book and implementing each algorithm discussed in Python, and for me, that was a good balance between theory and practice. 

**Tools**

* [OpenAI Gym](https://github.com/openai/gym)- The OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. This did not yet exist when I learned reinforcement learning, but they have some great visualizations that make the process of training an agent feel more fun and rewarding than the basic text-based maze navigator that I learned with did. 

## Other Resources 

* [Distill.pub](https://distill.pub/) - This online publication aims to publish clear, understandable explanations of machine learning concepts. Currently, their compilation is by no means exhaustive, but if you happen to find an explanation here, there is a good chance that you will understand it better after reading this. 
