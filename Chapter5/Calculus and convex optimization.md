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
