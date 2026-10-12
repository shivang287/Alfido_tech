# Iris Classification Using Machine Learning

## Project Overview

This project builds a machine learning classifier to predict an Iris
flower species from four measurements: - Sepal length - Sepal width -
Petal length - Petal width

The target/output column is `species`, with three classes: Setosa,
Versicolor, and Virginica.

## Algorithms

Three classification algorithms are trained and compared: 1. k-Nearest
Neighbors (k-NN) 2. Logistic Regression 3. Decision Tree

## Tools and Libraries

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Joblib
-   Google Colab / Jupyter Notebook

## Workflow

1.  Load and inspect the dataset.
2.  Perform exploratory data analysis (EDA).
3.  Visualize feature relationships and class separability.
4.  Split data into training and testing sets.
5.  Train all three classifiers.
6.  Evaluate using accuracy, precision, recall, F1-score, and confusion
    matrices.
7.  Select the best model based on test accuracy.
8.  Save the selected model and test inference on a sample flower.

## Files

-   `Iris_Classification.ipynb` --- notebook with code, plots, training,
    and evaluation.
-   `iris_best_model.joblib` --- saved best-performing model.
-   `model_comparison.csv` --- model comparison metrics.

## How to Run

1.  Open the notebook in [Google
    Colab](https://colab.research.google.com/) or Jupyter Notebook.
2.  Run the cells from top to bottom.
3.  Ensure the required Python libraries are installed.
4.  Review the comparison table and confusion matrices.

## Example Inference

The saved model expects four measurements in this order: sepal length,
sepal width, petal length, petal width.

Example input: `[5.1, 3.5, 1.4, 0.2]`

``` python
import joblib
import pandas as pd

model = joblib.load("iris_best_model.joblib")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

prediction = model.predict(sample)
print("Predicted species:", prediction[0])
```

If your notebook uses different feature-column names, change the example
to match the exact column names used during training.

## Results

The actual metrics are saved in `model_comparison.csv`. Refer to that
file for the accuracy, precision, recall, and F1-score of each model.
Results depend on the dataset and train-test split.

## Author

Shivang

## Task

Beginner-level machine learning classification task for Alfido Tech.
