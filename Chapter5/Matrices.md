
## 5.1.2 Matrices

1. Why do we say that matrices are linear transformations?

Ans. A vector multiplied by a matrix ends up as a vector.

For example, take a matrix 
$$
\begin{align*}
  A = \left[
    \begin{array}{rrr}
      1 & 0 & -1\\
      3 & 1 & 2
    \end{array}
  \right].
\end{align*}
$$

$$
\begin{align*} A{x} = \left[ \begin{array}{rrr} 1 & 0 & -1\\ 3 & 1 & 2 \end{array} \right] \left[ \begin{array}{c} x\\ y\\ z \end{array} \right] = \left[ \begin{array}{c} x - z\\ 3x + y +2z \end{array} \right] =(x-z,3x+y+2z). \end{align*}
$$

We can define this as a function $\mathbf{f(x)} = \mathbf{Ax}$ and ${f} : \R^3 \to \R^2$.

This can be extended to a general case for ${f} : \R^n \to \R^m$. Each matrix can be extended to a function. But only **special** functions can be mapped to a matrix, known as **linear transformation**. The function **g(x)** is a linear transformation if each term of each component of **g(x)** is a number times one of the variables.

For example, the functions $\mathbf{f}(x, y) = (2x + y, y/2)$ is a linear transformation while $\mathbf{f}(x, y) = (x^{2}, y, x)$ is not a linear transformation.

Source: https://mathinsight.org/matrices_linear_transformations

2. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

Ans. The inverse of a matrix $A$ is another matrix $A^{-1}$ such that the product of $A$ and $A^{-1}$ is the identity matrix.

Some important properties of the inverse are:
$$
\begin{array}{l}A{A}^{-1}=I\\ {A}^{-1}A=I\end{array} 
$$

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
    $$
\begin{align*}
  \left[
    \begin{array}{rrr}
      1 & 4 & -2\\
      -1 & 3 & 2 \\
      3 & 5 & -6 \\
    \end{array}
  \right]
\end{align*}
$$

    Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?
    
    **Hint**: rely on a property of this matrix to determine its determinant.

Ans. The 1st and 2nd columns of the matrix are multiples of each other.
   $$
\begin{gather}
 \begin{bmatrix} 
 -2 \\ 
 2 \\  
 -6\\
 \end{bmatrix} = -2 \times
 \begin{bmatrix} 
 1 \\ 
 -1 \\  
 3\\
 \end{bmatrix}
 \end{gather}
   $$
This makes this matrix linearly dependent and hence the determinant is 0.

Source: https://study.com/academy/lesson/linear-independence-definition-examples.html#:~:text=A%20matrix%20with%20a%20determinant,one%20of%20the%20other%20equations.

7. What’s the difference between the covariance matrix $A^TA$ and the Gram matrix $AA^T$?

Ans. The covariance matrix $A^TA$ represents the pairwise covariances between the columns of the matrix A. It is symmetric and positive definite and used to measure the relationships between the different variables in data.

On the other hand, the Gram matrix $AA^T$ represents the dot product between the rows of the matrix A. In kernel methods, 
