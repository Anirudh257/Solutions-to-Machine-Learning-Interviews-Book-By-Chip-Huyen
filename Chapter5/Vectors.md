## 5.1.1 Vectors

1. Dot product
   
   i. [E] What’s the geometric interpretation of the dot product of two vectors?
   
   Ans. It's the length of the projection multiplied by the length of the vector onto which the projection was projected when the 
two vectors are placed so that their tails coincide.

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/c3d2227c-200e-4451-9a27-6bd8aa813e0c)

   ii. [E] Given a vector $u$, find vector $v$ of unit length such that the dot product of $u$ and $v$ is maximum.

   Ans. The dot product of any two vectors is maximum if the two vectors are perfectly aligned with one another or $\theta$ between 
   them is $0 \textdegree$. So, $v$ will be a unit vector parallel to $u$.

2. Outer product

    i. [E] Given two vectors $a = [3, 2, 1]$ and  $b = [-1, 0, 1]$. Calculate the outer product $a^Tb$?

    Ans.
        ![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/53e8ca12-bf64-4739-9dc9-730dff75a350)


    ii. [M] Give an example of how the outer product can be useful in ML.

    Ans. One typical example of the outer product in ML is in Neural Networks and more particularly in backpropagation where the
    outer product is a way of transmitting the local error of a layer backward to the previous layers so that the weights can be
    adjusted properly (namely be corrected to a degree and come closer to the value that they should have).

     ![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/857a854a-f7bc-4f6b-9df9-b3aca99fb95e)
     Source: https://cs224d.stanford.edu/lectures/CS224d-Lecture5.pdf
   
3. [E] What does it mean for two vectors to be linearly independent?

Ans.  A set of two vectors is linearly independent if and only if neither of the vectors is a multiple of the other. 

4. [M] Given two sets of vectors $A = {a_1, a_2, a_3, ..., a_n}$ and $B = {b_1, b_2, b_3, ... , b_m}$. How do you check that
   they share the same basis?

Ans. 

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/96fc1b8f-f78e-40a3-9ad9-12b138614d22)

 
5. [M] Given $n$ vectors, each of $d$ dimensions. What is the dimension of their span?

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/77295c58-e045-4f58-8b7f-d35d63b5f325)

span = min(n, d).

7. Norms and metrics
	
   i. [E] What's a norm? What is $L_0, L_1, L_2, L_{norm}$?

   Ans. In mathematics, a norm is a function from a real or complex vector space to the non-negative real numbers that
   behaves in certain ways like the distance from the origin. It is a measure of the "length" of the vector.

     Formally, a norm is any function $\mathbb{R}^{n} \rightarrow \mathbb{R}$ that satisfies the following 4 properties:
	
	 a. For all $x \in \mathbb{R}^{n}, f(x) \ge 0$ (non-negativity).
	 b. $f(x) = 0$ if and only if x = 0 (definiteness).
	 c. For all $x \in \mathbb{R}^{n}, t \in \mathbb{R}, f(tx) = |t|f(x)$ (homogeneity).
	 d. For all $x, y \in \mathbb{R}^{n}, f(x + y) \le f(x) + f(y)$ (triangle inequality).
	 
   Source: https://www.wikiwand.com/en/Norm_(mathematics), https://cs229.stanford.edu/summer2020/cs229-linalg.pdf

  * $L_0$: Strictly speaking, $L_0$ norm is not a norm. It is a cardinality function that has its definition in the form of $l_p-norm$, though many people call it a norm.  
  
  * $L_1$: The Manhattan distance is the sum of the magnitudes of the vectors in space. It is the most natural way of measure distance between vectors, that is the sum of the absolute difference of the components of the vectors.
                   $||x||_{1} = \sum_{i=1}^n |x_{i}|$

* $L_2$: The Euclidean distance of a vector from the origin is a norm, called the Euclidean norm, or 2-norm, which may also be defined as the square root of the inner product of a vector with itself.
			$||x||_{2} = \sqrt{\sum_{i=1}^n x_{i}^{2} }$

* $L_{norm}$: Also known as $\textit{infinite norm}$. Consider the vector $\boldsymbol{x}$, let’s say if $x_j$ is the highest entry in the vector $\boldsymbol{x}$, by the property of the infinity itself, we can say that is the maximum entries’ magnitude of that vector. 
	  	$||x||_{\infty} = max_{i}|x_{i}|$
	  	
  ii. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

  Ans. A metric measures distances between *pairs of things* while a **norm** measures the size of a single item. Metrics can be defined on pretty much anything, while the notion of a norm applies only to vector spaces: the very definition of a norm requires that the things measured by the norm could be added and scaled. If you have a norm, you can define a metric by saying that the distance between **a** and **b** is the size of **a** - **b**. 

	 $d(a,b) = ∥a−b∥$
On the other hand, if you have a metric you can't usually define a norm.

References: 

- https://good-reward-8b2.notion.site/ML-Interviews-Book-Huyen-Chip-6a22670097ae48c8afc7480daadb6551

- https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/
