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
