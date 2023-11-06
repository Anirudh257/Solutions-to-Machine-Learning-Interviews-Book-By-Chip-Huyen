## 5.1.2 Matrices

1. Why do we say that matrices are linear transformations?

Ans. The product of a vector by a matrix is a vector.

For example, take the matrix:

$$A = \begin{bmatrix}  
1 & 0 & -1\\  
3 & 1 & 2  
\end{bmatrix}$$

$$\begin{align*}
Ax = \begin{bmatrix} 
1 & 0 & -1 \\
3 & 1 & 2
\end{bmatrix} \begin{bmatrix}x \\ y \\ z\end{bmatrix} = 
\begin{bmatrix}x - z \\ 3x + y + 2z \end{bmatrix} = \\
(x - z, 3x + y + 2z)
\end{align*}
$$

We can define this as a function $\textbf{f(x)} =\textbf{Ax}$ and $f : \mathbb{R}^3 \to \mathbb{R}^2$.

In general, matrix multiplication be defined as $f : \mathbb{R}^n \to \mathbb{R}^m$. But only special functions can be mapped to a matrix, known as **linear transformation**. The function **g(x)** is a linear transformation if each term of each component of **g(x)** is a number times one of the variables.

For example, the functions $\textbf{f}(x, y) = (2x + y, y/2)$ is a linear transformation while $\textbf{f}(x, y) = (x^2, y, x)$ is not a linear transformation.

Source: https://mathinsight.org/matrices_linear_transformations

2. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

Ans. The inverse of a matrix $A$ is another matrix $A^{-1}$ such that the product of $A$ and $A^{-1}$ is the identity matrix.

Some important properties of the inverse are:

$$\begin{align*}AA^{-1} = I \\ 
	A^{-1}A = I
\end{align*}$$

Only square matrices may have a multiplicative inverse. Not all square matrices have an inverse, but if $A$ is invertible, then $A^{−1}$ is unique.

Source: https://courses.lumenlearning.com/waymakercollegealgebra/chapter/find-the-inverse-of-a-matrix/#:~:text=A%20matrix%20that%20has%20a%20multiplicative%20inverse%20is%20called%20an,then%20A%E2%88%921%20is%20unique.

3. What does the determinant of a matrix represent?

Ans. It is an important quantity associated with the matrix that represents the **amount of scaling and reflection** obtained by the linear transformation of the matrix.

As we know that each matrix is associated with a linear transformation, $T(\mathbf{x}) = A\mathbf{x}$
For square matrices, the determinant determines how applying $\mathbf{T}$ expands or compresses objects. 

$|\det(cA)| = c^n |\det(A)|.$

Similarly, multiplying by a negative number changes the direction.

Source: https://mathinsight.org/determinant_geometric_properties#:~:text=The%20determinant%20of%20a%20square,can%20scale%20or%20reflect%20objects.

4. What happens to the determinant of a matrix if we multiply one of its rows by a scalar ?

Ans. If A is an n x n matrix and B is a matrix resulting from multiplying some row of A by a scalar k. Then $det(B) = k \times det(A)$.

If we multiply any 2 rows of A by k, the determinant becomes $k^2 \times det(A)$. If all the n rows are multiplied by a scalar k, the new determinant becomes $k^n \times det(A)$.

Source: https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)/03%3A_Determinants/3.02%3A_Properties_of_Determinants#:~:text=Theorem%203.2.-,2%3A%20Multiplying%20a%20Row%20by%20a%20Scalar,%3Dkdet(A).

5.  A matrix has four eigenvalues $3, 3, 2, -1$. What can we say about the trace and the determinant of this matrix?

Ans. The relationship between trace and determinant of the matrix is given by:

$tr(A) = \lambda_1 + \lambda_2 + \lambda_3 + \lambda_4$ and $det(A) = \lambda_1\lambda_2\lambda_3\lambda_4$ respectively.

Therefore, $tr(A) = 3 + 3+ 2-1 = 7$ and $det(A) = 3.3.2.-1 = -18$.

6.  Given the following matrix:  

$$\begin{bmatrix}1 & 4 & -2 \\
-1 & 3 & 2 \\
3 & 5 & -6
\end{bmatrix}$$

Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?

**Hint**: rely on a property of this matrix to determine its determinant.

Ans. The 1st and 2nd columns of the matrix are multiples of each other, i.e

$$\begin{align*}
\begin{bmatrix}-2 \\ 
2 \\ 
6 \end{bmatrix} 
= -2 \times \begin{bmatrix}
1 \\ 
-1 \\ 
3 
\end{bmatrix} 
\end{align*}$$
   
This makes this matrix linearly dependent and hence the determinant is 0.

Source: https://study.com/academy/lesson/linear-independence-definition-examples.html#:~:text=A%20matrix%20with%20a%20determinant,one%20of%20the%20other%20equations.

7. What’s the difference between the covariance matrix $A^TA$ and the Gram matrix $AA^T$?

Ans. The covariance matrix $A^TA$ represents the pairwise covariances between the columns of the matrix A. It is symmetric and positive definite and used to measure the relationships between the different variables in data.

On the other hand, the Gram matrix $AA^T$ represents the dot product between the rows of the matrix A. It is useful in document classification to compute the similarity between 2 documents. 

8. Given $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^n$

 i. [M] Find $x$ such that $Ax = b$.

Ans.  If $x$ is invertible, it is straightforward to compute $x = A^{-1}b$, else it is not possible to compute the inverse.

We can also use Gaussian elimination, matrix decomposition, or numerical methods to compute $x$.

ii. [E] When does this have a unique solution?

Ans.  It depends on these two scenarios:

a) If **A** is a square matrix (m == n), and matrix **A** has full rank, all the column vectors are independent and span the entire m-dimensional column space. 

b) If **A** is a tall matrix (m > n), and matrix **A** has full rank, the solution is unique only if **b** lies in the **span(A)**.

Reference: https://medium.com/@tseek2021/a-quick-summary-of-all-types-of-solutions-to-system-of-linear-equations-6501cc51673c

iii.  [M] Why is it that when A has more columns than rows,  $Ax=b$  has multiple solutions?

Ans. When there are more columns than rows, we are looking at the **undetermined system of equations**.  If the matrix(A) doesn't have **full rank**, i.e. rank(A) = r < min(m, n). There are only **r** independent column vectors in A and **(m - r)** dependent column vectors. If $b \in span(A)$, there exists infinite solutions to express **b**.

![](https://miro.medium.com/v2/resize:fit:875/1*UGEe5hwXskPL59_DUhh5Nw.png)

iv. [M] Given a matrix A with no inverse. How would you solve the equation  $Ax=b$? What is the pseudoinverse and how to calculate it?

Ans. When A is not invertible, we have 2 cases:

  (a) $Ax = b$ have infinite solutions.

 (b) $Ax = b$ have no solution.

 We can use Gaussian elimination to show that a solution/no solution exists.

If we want a *close-enough/best-fit* solution, we can use a **least squares solution** and solve for  $\min_x \|Ax-b\|^2$ that will give us the **pseudo-inverse**.

9. Derivative is the backbone of gradient descent.

i. [E] What does derivative represent?

Ans. The derivative of a function measures the sensitivity to change in the function output for a change in input.

When it exists, it can be mathematically described as the *slope* of the tangent line to the graph of the function at that point.  Intuitively, the derivative is the **best linear approximation** to the function at the given point.

Reference: https://math.stackexchange.com/questions/3266804/how-to-solve-ax-b-wihout-inverting-a

ii. [M] What’s the difference between derivative, gradient, and Jacobian?

Ans. Derivative of a function $f : \mathbb{R}^n \to \mathbb{R}^m$ at a point $p \in \mathbb{R}^n$, if it exists, is the unique linear transformation $D_f(p) \in L(\mathbb{R}^n, \mathbb{R}^m)$ such that

$$\lim_{h \to 0} \frac{\|f(p+h)-f(p)-D_f(p)h\|}{\|h\|} = 0;$$

The matrix of $D_f(p)$ with respect to the standard orthonormal bases of $\mathbb{R}^n$ and $\mathbb{R}^m$ is called the **Jacobian matrix** of $f$ at $p$ and lies in $M_{m \times n}(\mathbb{R})$.

The gradient of a function $f: \mathbb{R}^n \to \mathbb{R}$ is a vector that points in the direction of the steepest increase of the function at the given point. It is a generalization of derivative to functions of multiple variables. It consists of partial derivatives with respect to each variable. 

$$\begin{align*}
\triangledown_{(x, z)} = 
\begin{bmatrix}
\frac{df(x, z)}{dx} \\
\frac{df(x, z)}{dz}
\end{bmatrix}
\end{align*}$$

Reference: 

https://math.stackexchange.com/questions/336640/gradient-and-jacobian-row-and-column-conventions

https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/derivative_gradient_jacobian/#gradient-jacobian-and-generalized-jacobian

10.  [H] Say we have the weights  $w \in \mathbb{R}^{d \times m}$ and a mini-batch $x$ of $n$ elements, each element is of the shape  $1 \times d$  so that $x \in \mathbb{R}^{n \times d}$. We have the output  $y=f(x;w)= xw$. What’s the dimension of the Jacobian  $\frac{\delta y}{\delta x}$?

Ans. $y = xw \in \mathbb{R}^{n \times m}$.

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/1baf9b76-5394-4444-a8dc-5cf090bab120)

which is of same dimensions as the input $\mathbb{R}^{n \times m}$

Reference: 
https://www.wikiwand.com/en/Jacobian_matrix_and_determinant

https://www.cs.cmu.edu/~10315-s20/recitation/rec2_sol.pdf

11. [H] Given a very large symmetric matrix A that doesn’t fit in memory, say $A \in R^{1M \times 1M}$ and a function $f$ that can quickly compute $f(x) = Ax$ for $x \in R^{1M}$. Find the unit vector $x$ so that $x^TAx$ is minimal.
	
	**Hint**: Can you frame it as an optimization problem and use gradient descent to find an approximate solution?

Ans. By framing it as an optimization problem, we can define the objective function as $J(x) = x^TAx$ and find the corresponding $x$ that minimizes $J(x)$. By using gradient descent, we can initialize with some $x_{init}$ and keep updating the value using the formula: $x_{new} = x_{init} - \alpha\nabla J$ and  $\nabla J = 2Ax$.

For large symmetric matrices, we can randomly sample a subset of data to compute $Ax$ at each iteration. 

Or we can use this approach from https://github.com/starzmustdie/ml-interview-questions-and-answers/blob/main/ML_interview_questions_and_answers.pdf

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/7e78af58-d910-4586-ade9-73d7947844ba)


