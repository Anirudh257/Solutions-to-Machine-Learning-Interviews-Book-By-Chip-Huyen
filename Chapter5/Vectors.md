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

Ans. A set of vectors $\{ v_{1}, v_{2}, ..., v_{n} \}$ is said to form a **basis for a vector space** if:
     
    * The vectors $\{ v_{1}, v_{2}, ..., v_{n} \}$ span the vector space.
    
    * The vectors $\{ v_{1}, v_{2}, ..., v_{n} \}$ are linearly independent.

    Then, show that they span the same space.
 
5. [M] Given $n$ vectors, each of $d$ dimensions. What is the dimension of their span?

Ans. Let matrix  $A$ consists of n vectors side-by-side as columns then this matrix is of $d \times n$ dimensions. Suppose the 
     vectors that constitute this matrix are linearly independent, then:
    
     - The span of the rows of a matrix is called the **row space** of the matrix. The dimension of the row space is the 
       **rank** of the matrix (herein, rank = d)
    
     - The span of the columns of a matrix is called the **range** or the **column space** of the matrix (herein, column 
       space = n). *The row space and the column space always have the same dimension*.
   
7. Norms and metrics
	
   i. [E] What's a norm? What is $L_0, L_1, L_2, L_{norm}$?

   Ans. 
	
  ii. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

References: 

- https://good-reward-8b2.notion.site/ML-Interviews-Book-Huyen-Chip-6a22670097ae48c8afc7480daadb6551

- https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/
