# Data-Driven Materials Optimization

Data-driven predictive modeling and regression-based optimization using experimental materials and nanocomposites data.

## Motivation
This project demonstrates how experimental materials data can be leveraged using data-driven modeling and machine learning techniques to predict material properties and optimize material formulations.

The overarching goal is to bridge experimental materials science with applied data science and interpretable predictive modeling.

## Dataset
- Experimental dataset inspired by published nanocomposite studies  
- Number of samples: ~30  
- Input features: processing parameters and material composition variables  
- Target variables: mechanical and material performance metrics  

**Note:** The dataset used in this repository is anonymized, simplified, and intended solely for methodological demonstration. It does not contain confidential or proprietary experimental data.

## Methods
- Data preprocessing and feature scaling (standardization)
- Baseline linear regression and interaction modeling
- Regularized regression using LASSO for feature selection
- Model evaluation using R² and RMSE
- Interpretation of model coefficients and dominant variables

## Results
- Strong predictive performance achieved on experimental-style datasets
- Identification of key features influencing material performance
- LASSO regression enabled effective feature selection and model simplification
- Interaction effects highlighted nonlinear structure–property relationships

## Project Structure


data-driven-materials-optimization/
├── notebooks/
│   ├── 01_eda_and_linear_regression.ipynb
│   ├── 02_feature_selection_lasso.ipynb
│   └── 03_nonlinear_models_and_optimization.ipynb
├── README.md
├── LICENSE
└── .gitignore


## Skills Demonstrated
- Applied machine learning for materials and engineering problems
- Regression modeling and feature engineering
- Statistical analysis and model interpretation
- Python (NumPy, Pandas, scikit-learn)
- Translating experimental data into predictive insights

## Relation to Research
This project is inspired by methodologies commonly used in peer-reviewed research on data-driven modeling and optimization of nanocomposites.

The implementation is intentionally simplified and adapted for demonstration purposes in an applied data science and research-oriented context.
