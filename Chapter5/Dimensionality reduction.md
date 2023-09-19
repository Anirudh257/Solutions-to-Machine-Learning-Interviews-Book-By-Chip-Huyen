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
