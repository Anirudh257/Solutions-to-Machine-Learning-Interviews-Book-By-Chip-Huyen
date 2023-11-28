

#### 5.1.4 Calculus and convex optimization

1. Differentiable functions
    1. [E] What does it mean when a function is differentiable?
    2. [E] Give an example of when a function doesn’t have a derivative at a point.
    3. [M] Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?
  
 Ans. 

i. A function is said to be differentiable if its derivative exists at all points in its domain.

Mathematically, a function $f: U \rightarrow \mathbb{R}$ is said to be differentiable if $\forall a \in U$, the following derivative exists:

$$f'(a) = \lim _{h \rightarrow 0} \frac{f(a+h)-f(a)}{h}$$

Also, $f$ is continuous at $a$. Every differentiable function is continuous, but the converse is not true.

Reference: https://calcworkshop.com/derivatives/continuity-and-differentiability/


ii. A function $f(x) = x^{\frac{1}{3}}$ doesn't have a derivative at  $x= 0$.

iii. Some non-differentiable functions commonly used in Machine Learning are:

$$f(x) = |x|$$

$$\text{ReLU(x)} = \begin{cases}
                    x & \text{if } x \ge 0\\
                    0 & \text{otherwise}
                    \end{cases} \\ $$

$$\text{LeakyReLU(x)} = \begin{cases}
                    x & \text{if } x \ge 0\\
                    \alpha x & \text{otherwise}
                    \end{cases} \\ $$

Theoretically, these functions are not differentiable at $x = 0$. For such points, we can use sub-derivatives. Sub-derivatives can be treated as an approximation to derivatives for a non-continuous function.

2.  Convexity
    i.  [E] What does it mean for a function to be convex or concave? Draw it.
    ii.  [E] Why is convexity desirable in an optimization problem?
   iii.  [M] Show that the cross-entropy loss function is convex.

Ans. 

i. A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if its domain (denoted $D(f)$) is a convex set, and if, for all $x, y \in D(f)$ and $\theta \in \mathbb{R}, 0 \leq \theta \leq 1,$

$$f(\theta x + (1 - \theta y)) \leq \theta f(x) + (1 - \theta) f(y)$$

A line drawn between any two points on the graph of the convex function is always above the graph of the function.

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/6ae56b39-5381-4bad-a371-7b4ac39ec779)

Conversely, a concave function is the opposite of a convex function.

Mathematically, we can define it as: 
A real-valued function $f$ on an interval is said to be concave, if for any $x$ and $y$ in the interval and for any $\theta \in [0, 1],$

$$f(\theta y + (1 - \theta x)) \geq \theta f(y) + (1 - \theta) f(x)$$

![ConcaveDef.png](https://upload.wikimedia.org/wikipedia/commons/7/73/ConcaveDef.png)

The line joining any two points on the graph is below the graph of $f$ between the two points.

Reference: https://www.wikiwand.com/en/Concave_function

ii. 

A convex function is desirable as it is guaranteed to have a global minimum.

iii.

Cross-entropy is defined as: 

$$\begin{align}
H(p, q) =  -\sum_{x \in \mathcal{X}} p(x) \cdot \log_b q(x)
\end{align}$$ 
 
 where $X$ is a discrete random variable with possible outcomes $\mathcal{X}$, $P$ and $Q$ are two probability distributions on $X$ with the probability mass functions $p(x)$ and $q(x)$ and $b$ is the base of the logarithm specifying in which unit the cross-entropy is defined.

**Theorem:** We want to prove the convexity of this cross-entropy function, i.e.

$$\begin{align}
\mathrm{H}[p,\lambda q_1 + (1-\lambda) q_2] \leq \lambda \mathrm{H}[p,q_1] + (1-\lambda) \mathrm{H}[p,q_2]
\end{align}$$

where $p$ is fixed and $q_1$ and $q_2$ are any two probability distributions and $0 \leq\lambda\leq1$.

**Proof**: 

We will use the relationship between Kullback-Liebler divergence, entropy, and cross-entropy:

$$\begin{align}
\mathrm{KL}[P||Q] = \mathrm{H}(P,Q) - \mathrm{H}(P) \; .
\end{align}$$

Before delving further into the proof, we will prove that the KL divergence is convex.

The Kullback-Leibler divergence of $P$ and $Q$ for a discrete random variable $X$ is defined as:

$$
\begin{align}
\mathrm{KL}[P||Q] = \sum_{x \in \mathcal{X}} p(x) \cdot \log \frac{p(x)}{q(x)}
\end{align}
$$

The objective is to prove the convexity of KL-divergence, i.e.:

$$
\begin{align}
\mathrm{KL}[\lambda p_1 + (1-\lambda) p_2||\lambda q_1 \\ + (1-\lambda) q_2] \leq \lambda \mathrm{KL}[p_1||q_1] + (1-\lambda) \mathrm{KL}[p_2||q_2]
\end{align}
$$

where $(p_1, q_1)$ and $(p_2, q_2)$ are two pairs of probability distributions and $0 \leq \lambda \leq 1$.

**Sub-proof**:

KL-divergence of $P$ from $Q$ is defined as:

$$
\begin{align}
\mathrm{KL}[P||Q] = \sum_{x \in \mathcal{X}} p(x) \cdot \log \frac{p(x)}{q(x)}
\end{align}
$$

Using the **log-sum inequality**, that states:

$$
\begin{align}
\sum_{i=1}^n a_i \log \frac{a_i}{b_i} \geq \left( \sum_{i=1}^n a_i \right) \log \frac{\sum_{i=1}^n a_i}{\sum_{i=1}^n b_i}
\end{align}
$$

where $a_1, \ldots, a_n$ and $b_1, \ldots, b_n$ are non-negative real numbers.

We can rewrite the KL-divergence of the distribution as:

$$
\begin{align}
\mathrm{KL}[\lambda p_1 + (1-\lambda) p_2||\lambda q_1 + (1-\lambda) q_2] 
\end{align}
$$

From **KL-divergence** definition,
 
$$
\begin{align}
\sum_{x \in \mathcal{X}} \left[ \left[ \lambda p_1(x) + 
 (1-\lambda) p_2(x) \right] \cdot \log \frac{\lambda p_1(x)  (1-\lambda) p_2(x)}{\lambda q_1(x) + (1-\lambda) q_2(x)} \right] 
\end{align}
$$

From **log-sum inequality**,

$$
\begin{align}
\leq \sum_{x \in \mathcal{X}} \left[ \lambda p_1(x) \cdot \log \frac{\lambda p_1(x)}{\lambda q_1(x)} + (1-\lambda) p_2(x) \cdot \log \frac{(1-\lambda) p_2(x)}{(1-\lambda) q_2(x)} \right] \\
\end{align}
$$

$$
\begin{align}
= \lambda \sum_{x \in \mathcal{X}} p_1(x) \cdot \log \frac{p_1(x)}{q_1(x)} + (1-\lambda) \sum_{x \in \mathcal{X}} p_2(x) \cdot \log \frac{p_2(x)}{q_2(x)} \\
\end{align}
$$

$$
\begin{align}
\lambda  \mathrm{KL}[p_1||q_1] + (1-\lambda)  \mathrm{KL}[p_2||q_2]
\end{align}
$$


Source:

1) https://statproofbook.github.io/D/ent-cross

2) https://statproofbook.github.io/P/entcross-conv

3) https://statproofbook.github.io/P/kl-conv

4) https://statproofbook.github.io/D/kl
 
