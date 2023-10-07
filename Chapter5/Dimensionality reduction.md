

## 5.1.3 Dimensionality Reduction


1. [E] Why do we need dimensionality reduction?

Ans. Dimensionality reduction is a technique used in machine learning to reduce the number of features or dimensions in a dataset while retaining as much of the important information as possible. Here are some reasons why we need dimensionality reduction:

a) Curse of Dimensionality: As the number of features or dimensions in a dataset increases, the amount of data required to cover the space increases exponentially. This is known as the curse of dimensionality, and it can lead to overfitting, increased computational complexity, and reduced model performance. Dimensionality reduction can help mitigate this problem by reducing the number of features and making the data more manageable.

b) Visualization: High-dimensional data is difficult to visualize, making it challenging to understand and interpret. Dimensionality reduction can help by projecting the data onto a lower-dimensional space that can be easily visualized. This allows us to gain insights into the data and identify patterns that may not be apparent in the high-dimensional space.

c) Redundancy: High-dimensional data often contains redundant or irrelevant features that add noise and complexity to the dataset. Dimensionality reduction can help remove these features and focus on the most important ones, improving model performance and reducing overfitting.

d) Memory and Storage: High-dimensional data requires more memory and storage, making it more challenging to work with. Dimensionality reduction can help reduce the size of the dataset, making it easier to store and process.

e) Interpretability: In some cases, high-dimensional data may be difficult to interpret or explain. Dimensionality reduction can help simplify the data and make it more interpretable, allowing us to gain insights and make informed decisions.

In summary, dimensionality reduction is important in machine learning because it helps mitigate the curse of dimensionality, improves visualization, removes redundancy, reduces memory and storage requirements, and improves interpretability

2. [E] Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?

Ans. No, Eigendecomposition of a matrix is not always unique. It is not always unique when two eigenvalues are the same. It is unique only when all the eigenvalues are unique. 

Example from https://github.com/starzmustdie/ml-interview-questions-and-answers/blob/main/ML_interview_questions_and_answers.pdf

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/c7b07de8-6aa9-47d5-81bd-6b6d82bd1595)

Reference: https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut4_slides.pdf

3.  [M] Name some applications of eigenvalues and eigenvectors.

Ans. Some of the important applications are:

a. Singular Value Decomposition for image compression.

b. Spectral clustering used in data analysis.

c. Dimensionality reduction/PCA.

d. Low rank factorization for collaborative prediction.

e. Google Page Rank algorithm.

Reference: https://math.stackexchange.com/questions/1520832/real-life-examples-for-eigenvalues-eigenvectors

4.  [M] We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range 0-1 and one is in the range 10 - 1000. Will PCA work on this dataset?

Ans. In PCA we are interested in the components that maximize the variance. If one component (e.g. human height) varies less than another (e.g. weight) because of their respective scales (meters vs. kilos), PCA might determine that the direction of maximal variance more closely corresponds with the ‘weight’
axis, if those features are not scaled. Since a change in height of one meter should be considered much more important than the change in weight of one kilogram, the previous assumption would be incorrect. Therefore, it is important to standardize the features before applying PCA.

5. [H] Under what conditions can one apply eigendecomposition? What about SVD?

Ans. Eigendecomposition is possible only for (square) diagonalizable matrices. On the other hand, the Singular Value Decomposition (SVD) always exists (even for non-square matrices).

(i).  What is the relationship between SVD and eigendecomposition?

Ans. SVD is a more general matrix factorization technique than eigendecomposition. It is applicable to any matrix while eigendecomposition can only be applied to square matrices. 

![image](https://github.com/Anirudh257/Solutions-to-Machine-Learning-Interviews-Book-By-Chip-Huyen/assets/16001446/151dd734-5f1e-4084-a344-3cfbeee77837)

The left singular vectors of $A$ are the eigenvectors of $AA^T$ and the right singular vectors of $A$ are the eigenvectors of $A^TA$. If $\lambda$ is a an eigenvalue of $AA^T$ (or $A^TA$), the eigenvalues (and thus the singular values) are non-negative.
