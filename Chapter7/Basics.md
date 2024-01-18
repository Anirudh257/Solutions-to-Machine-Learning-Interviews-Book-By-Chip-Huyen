# 7.1 Basics

1. [E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.

Ans. 

Supervised learning is the task of learning a function that maps an input to the output based on example input-output pairs.

In unsupervised learning, the task is to learn patterns exclusively from unlabeled data.

Weakly supervised learning is the task of learning where a limited, noisy or imprecise sources of data is used to provide supervision signal for labelling large amounts of training data in a supervised learning setting. It is useful in cases when the task of collecting hand-labeled data is costly or impractical. Instead, inexpensive weak labels are employed with the understanding that they are imper-  
fect, but can nonetheless be used to create a strong predictive model.

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

Ans. 

In empirical risk minimization, we assume that there is a joint probability distribution $P(x, y)$ over an an input space $X$ and a target space $Y$. The goal is to learn a function $h: X \rightarrow Y$ (often called hypothesis) which outputs an object $y \in Y$, given $x \in X$. Moreover, assume that we are given a non-negative real-valued loss function $L(\hat{y}, y)$ which measures how different the prediction $\hat{y}$ of a hypothesis is from the true outcome $y$. \\\\
            The risk associated with the hypothesis $h(x)$ is then defined as the expectation of the loss function:


  $$ R(h) = \mathbb{E}[{L(h(x), y)}] = \int L(h(x), y) dP(x, y) $$

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
                h^* = \argmin_{h \in \mathcal{H}} R(h)
            \end{align*}
$$
            
