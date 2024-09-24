# Analysis-of-Factors-Affecting-Vehicle-Selling-Price
## Overview
This project implements a multiple linear regression model to predict car selling prices based on various factors, including vehicle attributes such as mileage, engine size, fuel type, and year of manufacture. The project uses statistical methods to provide insights into the key factors that affect the price of a used car.

## Features
- **Multiple Linear Regression Model**: Predicts the selling price of a car based on multiple input features.
- **Data Cleaning and Preprocessing**: Handles missing values, feature encoding, and normalization for improved model performance.
- **Model Evaluation**: Evaluates the accuracy of the model using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.

## Dataset
The dataset used for this project includes information such as:
- Car make and model
- Year of manufacture
- Kilometers driven
- Engine size and fuel type
- Sale price of the car

## Project Structure
```bash
Car-Selling-Price-Prediction/
│
├── Project_Code.R            # Main R script for analysis and modeling
├── car.csv                   # Dataset used for the analysis
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies
```

## How It Works
- **Data Preprocessing**: The dataset is cleaned, missing values are handled, and necessary transformations such as encoding categorical variables are applied.
- **Feature Engineering**: Features like year, mileage, engine size, and fuel type are used as inputs to the model.
- **Modeling**: Multiple linear regression is applied to predict the car selling price.
- **Evaluation**: The model is evaluated using various performance metrics to ensure accuracy and robustness.

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- R 4.0+
- Required R packages (`dplyr`, `ggplot2`, `caret`, `MASS`, etc.)

### Clone the Repository
```bash
git clone https://github.com/your-username/car-selling-price-prediction.git
cd car-selling-price-prediction
```

### Install Dependencies
Install the required R packages by running the following command in your R environment:
```bash
install.packages(c("dplyr", "ggplot2", "caret", "MASS"))
```

### Running the Analysis
To run the project, open the Project_Code.R file in RStudio or any R environment and execute the code. The script will:
-Load the data.
-Preprocess the data (handle missing values, normalize features).
-Train the regression model.
-Evaluate the model on test data.

### Key Libraries Used
-dplyr: For data manipulation.
-ggplot2: For data visualization.
-caret: For model training and evaluation.
-MASS: For regression modeling.

### Future Enhancements
Model Optimization: Explore other regression models like Ridge or Lasso to improve performance.
Feature Addition: Add more features such as the car's location or service history to improve prediction accuracy.

### Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request for enhancements or bug fixes.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
