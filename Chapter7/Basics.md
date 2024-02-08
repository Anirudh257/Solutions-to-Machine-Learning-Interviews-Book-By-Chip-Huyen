
# 7.1 Basics

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.

Ans. Supervised learning is the task of learning a function that maps an input to the output based on example input-output pairs.

In unsupervised learning, the task is to learn patterns exclusively from unlabeled data.

Weakly supervised learning is the task of learning where a limited, noisy or imprecise sources of data is used to provide supervision signal for labelling large amounts of training data in a supervised learning setting. It is useful in cases when the task of collecting hand-labeled data is costly or impractical. Instead, inexpensive weak labels are employed with the understanding that they are imperfect, but can nonetheless be used to create a strong predictive model.

Semi-supervised learning  is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training.  Semi-supervised learning falls between unsupervised learning and supervised learning, and is a special instance of weak supervision.

Active learning  is a branch of machine learning where a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs. It is useful when a large amount of data needs to be labelled and the model can prioritize samples that make the most impact on training the model. 

Self-supervised learning  is a branch of machine learning that learns from unlabeled data by automatically extracting labels from the sample. For example, we could mask out a word in a sentence, which the algorithm then has to predict.


Reference:
 
* https://www.wikiwand.com/en/Supervised_learning

* https://www.wikiwand.com/en/Unsupervised_learning

* https://www.wikiwand.com/en/Weak_supervision

* https://www.wikiwand.com/en/Semi-supervised_learning

* https://www.wikiwand.com/en/Active_learning_(machine_learning)

* https://www.wikiwand.com/en/Self-supervised_learning


2. Empirical risk minimization.

i.  [E] What’s the risk in empirical risk minimization?

Ans. In empirical risk minimization, we assume that there is a joint probability distribution $P(x, y)$ over an an input space $X$ and a target space $Y$. The goal is to learn a function $h: X \rightarrow Y$ (often called hypothesis) which outputs an object $y \in Y$, given $x \in X$. Moreover, assume that we are given a non-negative real-valued loss function $L(\hat{y}, y)$ which measures how different the prediction $\hat{y}$ of a hypothesis is from the true outcome $y$. \\\\
            The risk associated with the hypothesis $h(x)$ is then defined as the expectation of the loss function:


  $$ 
  R(h) = \mathbb{E}[{L(h(x), y)}] = \int L(h(x), y) dP(x, y) 
  $$

A loss function commonly used in theory is the 0-1 loss function:

$$
            \begin{align*}
                L(\hat{y}, y) = \begin{cases}
                    1 &\text{if } \hat{y} \neq y \\
                    0 &\text{if } \hat{y} = y
                \end{cases}
            \end{align*}
$$

The ultimate goal of a learning algorithm is to find a hypothesis $h^*$ among a fixed class of functions $\mathcal{H}$ for which the risk $R(h)$ is minimal:

$$
            \begin{align*}
                h^* = \text{argmin}_{h \in \mathcal{H}} R(h)
            \end{align*}
$$

ii.  [E] Why is it empirical?

Ans. We do not know the exact value of the distribution $P(x, y)$ and can't compute the risk $R(h)$. We make an estimate of it using our training set.

$$
                R_{\text{emp}} = \frac{1}{n}\sum_{i=1}^n L(h(x_i), y_i)
$$

The empirical risk minimization principle states that the learning algorithm should choose a hypothesis $\hat{h}$ which minimizes the empirical risk:

$$
                \hat{h} = \text{argmin}_{h \in \mathcal{H}} R_{\text{emp}}(h)
$$

  iii.  [E] How do we minimize that risk?

Ans. Empirical risk minimization for a classification problem with a 0-1 loss function is known to be an NP-hard problem even for linear classifiers.  

In practice, machine learning algorithms cope with this problem by employing a convex approximation to the 0-1 (like hinge loss for SVM), which is easier to optimize.

Reference: https://www.wikiwand.com/en/Empirical_risk_minimization#Computational_complexity

3. [E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?

Ans. In ML, we use this principle to select between two different models. Given two models with the same generalization error, the simpler one should be preferred because simplicity is desirable. Given two models with the same training-set error, the simpler one should be preferred because it is likely to have a lower generalization error.

**When choosing between two models, we can only say a simpler model is better if its generalization error is equal to or less than that of the more complex model.**

Reference: https://towardsdatascience.com/what-occams-razor-means-in-machine-learning-53f07effc97c

4. [E] What are the conditions that allowed deep learning to gain popularity in the last decade?

Ans. The three main factors that allowed deep learning to gain popularity are:

• Increased access to large datasets of high quality and fine-grained labels (ImageNet, CityScapes).

• Improved hardware (GPUs, TPUs).

• Algorithmic advances (Residual Connections, Attention mechanism, BatchNorm)

5. [M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

Ans. Having multiple layers helps us learn features at various levels of abstraction. For example, if you train a deep convolutional neural network to classify images, you will find that the first layer will train itself to recognize very basic things like edges, the next layer will train itself to recognize collections of edges such as shapes, the next layer will train itself to recognize collections of shapes like eyes or noses, and the next layer will learn even higher-order features like faces. **Multiple layers are much better at generalizing because they learn all the intermediate features** between the raw data and the high-level classification.

A wider network is also more prone to overfitting.

Reference: https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider

6. [H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?

Ans. The Universal Approximation Theorem states that a neural network with a single hidden layer can approximate any continuous function for inputs within a specific range. While the theorem guarantees that the network can approximate any function, it does not provide specific guidance on the number of neurons or the complexity required to achieve a certain level of accuracy. A low error can be achieved by an exponentially large number of neurons but such a wide model is prone to overfitting and difficult to train.

7. [E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

Ans. Saddle points are critical points where the function attains neither a local maximum or a local minimum value.

![Saddle point - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/1/1e/Saddle_point.svg)

Local minima are points in the function where it reaches the **minimum** value in its neighbourhood. **It may or may not be the minimum** for the whole function.

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/3009fb26-e68d-42eb-a413-ccf9f6fd752e)



Empirically, it has been shown that **saddle points** can cause more problems than **local minimum** for training models. For a point to truly be a local minimum of the loss function, it has to be a local minimum in all directions, where each direction is specified by one of the parameters of the network.Given that we usually train million-  
(or even billion-) parameter models, it is much more likely that at least one direction displays different behavior than the others, as opposed to all of them displaying the same behavior.  Therefore, we can  
conclude that local minima are not as common.

In practice, networks trained by Stochastic Gradient Descent almost always escapes from the local minima. Since we are calculating the loss wrt. the current batch (and not the entire dataset), we are not truly traversing the original loss landscape, but a proxy of it. And if we eventually get stuck in a local minima / saddle point in the loss landscape (or even its current proxy), in the next iteration we are optimizing over a different batch, which is a different proxy of the loss, and therefore will slightly nudge us in a different direction. This regularization effect is a huge reason why we are able to train neural networks that show remarkable capabilities. 

Reference:

* https://davidlibland.github.io/posts/2018-11-10-saddle-points-and-sdg.html

* https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/

* https://www.mathsisfun.com/algebra/functions-maxima-minima.html

8. Hyperparameters.

i. [E] What are the differences between parameters and hyperparameters?

Ans. Parameters are quantities that are optimized by the model during the training process (e.g. weights of a neural network). Hyperparameters are quantities related to the learning procedure, which are not optimized during training, but are set by the user before the training starts (e.g. learning rate of the optimizer).

ii. [E] Why is hyperparameter tuning important?

Ans. Hyperparameter tuning consists of finding a set of optimal hyperparameter values for a learning algorithm while applying this optimized algorithm to any data set. That combination of hyperparameters maximizes the model’s performance, minimizing a predefined loss function to produce better results with fewer errors. Note that the learning algorithm optimizes the loss based on the input data and tries to find an optimal solution within the given setting. However, hyperparameters describe this setting exactly.

Eg. of Hyperparameters for a NN:

a) Number of hidden layers.

b) Number of nodes/neurons per layer

c) Learning Rate

d) Momentum

Most important ones as recommended by Andrew NG are: learning rate, adam's beta parameter, hidden units.  

Some techniques to perform hyperparameter tuning are: random sampling from a grid of hyperparameter space, coarse to fine sampling.  

Reference: https://www.anyscale.com/blog/what-is-hyperparameter-tuning

iii. [M] Explain algorithm for tuning hyperparameters.

Ans. Two fairly naive, but commonly used algorithms for tuning hyperparameters are:  

•  Grid Search – given a set of values for each hyperparameter, the algorithm looks over each possible combination.  

•  Random Search – given an interval of possible hyperparameter values, the algorithm trains the model by sampling randomly from the provided ranges.  

One major drawback of these two approaches is that they are uninformative – the choice of the  next set of parameters is independent of the performance of the previous choice.  This serves as a  motivating factor as to why someone might consider using Bayesian Optimization.

In Bayesian Optimization, we don't have any assumption about the **convexity**, **analytic** form or the **optimization cost** of the function. This is a general technique. We can only compute $f(x)$ at some $x$'s.

Since, we don't have any mathematical expression for $f(x)$, we use a **surrogate model to approximate**  $f(x)$. Usually, Gaussian Processes (GPs) are a good choice for surrogate models. They provide a good model that best fits the data $\mu(x)$ and produce uncertainty estimates $\sigma(x)$ for each x.

We also require a **acquisition model** to decide which $x$ to sample next based on the **surrogate model**. It makes a choice whether to explore the already visited regions to optimize $f$ or *explore*  unknown regions. This is popularly known as the **exploration-exploitation trade-off**. The most common choice for this model is the **upper-confidence-bound** (UCB). It is defined as:

$a_{UCB}(x, \lambda) = \mu(x) + \lambda\sigma(x)$

 where $\mu(x)$ is the mean of the GP posterior at $x$, $\sigma(x)$ is the standard deviation of the GP posterior at $x$, and $\lambda$ is a hyperparameter which trades-off the two (this is user set, and not optimized). As we discussed, the affinity for each hyperparameter $x$ is a weighted sum of the expected performance and the uncertainty surrounding the choice of this parameter.

The overall pseudocode is:

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/e8fe92c9-49e1-4863-a60f-c7d5ec0efa7c)


Reference: 

* https://ekamperi.github.io/machine%20learning/2021/05/08/bayesian-optimization.html

* https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083

* https://github.com/zafstojano/ml-interview-questions-and-answers

9. Classification vs. regression.

i.  [E] What makes a classification problem different from a regression problem?

Ans. In classification, the output is a discrete category label, but in regression, it is a continuous value.

ii.  [E] Can a classification problem be turned into a regression problem and vice versa?

Ans. It is technically possible to convert a regression problem into classification by defining a threshold. For example, if I have a dataset of images with labels of their cloudiness level from 0 to 5 I could define a threshold, i.e. 2.5 and use it to turn the continuous values into discrete one, and use those discrete values as classes (cloudiness level < 2.5 equal image without clouds) but the opposite is definitely not possible.

But converting from classification to regression is not possible as we can't define an order. (Cat < Dog doesn't make sense)


However, there are multiple reasons to avoid it, like:

* Loss of information by binning.

* Continuous targets have an order but classification classes don't.

* Continuous targets usually have some kind of smoothness: Proximity in feature space (for continuous features) means proximity in target space.

*   All this loss of information is accompanied by possibly more parameters in the model, e.g. logistic regression has number of coefficients proportional to number of classes.

* The binning obfuscates whether one is trying to predict the expectation/mean or a quantile.

* One can end up with a badly (conditionally) calibrated regression model, ie biased. (This can also happen for stdandard regression techniques.)

Reference: 

* https://stats.stackexchange.com/questions/565537/is-there-ever-a-reason-to-solve-a-regression-problem-as-a-classification-problem

* https://stackoverflow.com/questions/57268169/could-i-turn-a-classification-problem-into-regression-problem-by-encoding-the-cl

* https://datascience.stackexchange.com/questions/70313/how-to-make-a-classification-problem-into-a-regression-problem

10.  Parametric vs. non-parametric methods.
    
 i.  [E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
 
 Ans.  Parametric model is a method with a fixed size of parameters while a non-parametric model can have potentially infinite size of parameters. In non-parametric methods, the complexity of the model grows with the number of training data samples. For example, linear regression, logistic regression and linear Support-vector machines are parametric models having a fixed size of parameters (weight coefficient). But the KNN, decision trees, or SVM with RBF kernel SVMS are non-parametric as the number of parameters increases with the training data size. **However, as noted in The Handbook of Nonparametric Statistics 1 (1962) on p. 2: “A precise and universally acceptable definition of the term ‘nonparametric’ is not presently available**.
 
Reference: https://sebastianraschka.com/faq/docs/parametric_vs_nonparametric.html
 
ii. [H] When should we use one and when should we use the other?

Ans. It depends on the following aspects:

a) **Dataset size**: Non-parametric methods are more applicable in cases when we have larger datasets that provide sufficient coverage s.t. we can derive the entire model structure based on the data alone. Otherwise, if the datasets are not large enough, we can inject a prior into our training process by fixing the parametric form of the model. This allows the optimization procedure to only focus on inferring the parameter values, and not having to derive the entire model structure.

b) **Inference time requirements**: Since parametric models use a fixed parametrization, they are more applicable in cases when we need consistent inference time guarantees. In contrast, the prediction time of non-parametric methods might depend on the dataset size (e.g. finding k-nearest neighbors, iterating over all support vectors, \ldots)

Reference: https://github.com/starzmustdie/ml-interview-questions-and-answers
