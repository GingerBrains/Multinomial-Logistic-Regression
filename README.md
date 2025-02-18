# MLG2023 - Machine Learning Framework in C#

## Overview
This repository contains a C# implementation of a machine learning framework designed for **Multivariate Linear Regression (MLR)**. The framework includes utilities for **data processing, vector operations, and training/testing machine learning models**. It is designed to be **modular and extensible**, allowing for future enhancements such as logistic regression or artificial neural networks.

## Features
- **Data Processing**: Load and preprocess datasets from CSV files.
- **Vector Operations**: Support for vector arithmetic (addition, subtraction, dot product, etc.).
- **Multivariate Linear Regression (MLR)**:
  - Train and test MLR models.
  - Support for stochastic, batch, and mini-batch gradient descent.
  - Configurable learning rate, epochs, and batch size.
- **SoftMax Activation**: Used for multi-class classification.
- **Random Number Generation**: Utilities for generating random numbers for weight initialization.

## Code Structure
The project is organized into the following classes:

### 1. RNG (Random Number Generator)
- Generates random integers and doubles for weight initialization.

### 2. MLRRow
- Represents a single row of data for MLR, containing input and output vectors.

### 3. MLR (Multivariate Linear Regression)
- Implements the MLR algorithm.
- Methods for training, testing, and evaluating the model.
- Supports splitting data into training and testing sets.

### 4. DataProcessor
- Handles loading and preprocessing of datasets from CSV files.
- Supports different file types (e.g., LR, MLR, ANN).

### 5. DataRow
- Represents a single row of data with input and output values.

### 6. Vector
- Implements vector operations (addition, subtraction, dot product, etc.).
- Provides utility methods like `Zero()` and `ToString()`.

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/MLG2023.git
```

- Open the project in your preferred C# IDE (e.g., **Visual Studio**).
- Build the project to resolve dependencies.

## Usage

### 1. Loading Data
Use the `DataProcessor` class to load your dataset:

```csharp
DataProcessor dp = new DataProcessor();
dp.Ft = FileType.MLR;
dp.Filelocation = "path/to/your/dataset.csv";
dp.Header = true;
dp.LoadData();
```

### 2. Training the Model
Create an instance of the `MLR` class and configure it:

```csharp
MLR mlr = new MLR();
mlr.ImportData(dp.Alldata);
mlr.Nepochs = 50;
mlr.Split = 0.7; // 70% training, 30% testing
mlr.LearningRate = 0.01;
mlr.BatchSize = 1; // Stochastic gradient descent
mlr.SplitData();
mlr.Train();
```

### 3. Testing the Model
Evaluate the model on the test dataset:

```csharp
mlr.Test();
```

### 4. Vector Operations
Use the `Vector` class for arithmetic operations:

```csharp
Vector a = new Vector(3);
a[0] = 1; a[1] = 2; a[2] = 3;
Vector b = new Vector(3);
b[0] = 4; b[1] = 5; b[2] = 6;
Vector c = a + b;
Console.WriteLine(c); // Output: (5.00, 7.00, 9.00)
```

## Example Dataset
The framework is designed to work with CSV files. For example, the `Iris.csv` dataset can be used for multi-class classification. Ensure the dataset has the following format:

```csv
SepalLength,SepalWidth,PetalLength,PetalWidth,Species
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
...
```

## Dependencies
- **.NET Framework or .NET Core**
- No external libraries are required.

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## Acknowledgments
- The `Vector` class provides a foundation for linear algebra operations.
- The `SoftMax` function is used for multi-class classification.
- The framework is designed to be **simple and extensible** for educational purposes.
