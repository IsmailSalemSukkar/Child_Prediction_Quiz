# Predicting Number of Children using GSS Survey Data
This model's accuracy is ±1, and was done mostly for fun so please do not use this in any official or medical way. This app was also written very quickly so attempting to break it will break it.

## Overview

This repository contains code and documentation for a project in which data from the General Social Survey (GSS), conducted by the National Opinion Research Center (NORC), was utilized to predict the number of children a person might have based on various factors including religiousness, idealized family size, social class, working status, and beliefs about a parent being a stay-at-home parent. The project employed supervised machine learning techniques to build a predictive model.

## Data Source

The primary data source for this project is the General Social Survey (GSS) conducted by NORC. The GSS is a nationally representative survey of adults in the United States, designed to monitor changes and constants in attitudes, behaviors, and attributes of American society. 

## Methodology

### Variables Used
- **Religiousness:** This variable measures the level of religiosity of individuals, capturing their religious beliefs, practices, and affiliations.
- **Idealized Family Size:** This variable reflects individuals' perceptions or desires regarding the ideal number of children in a family.
- **Social Class:** This variable categorizes individuals based on their socio-economic status, which can influence family planning decisions.
- **Working Status:** This categorical variable indicates whether individuals are employed or not, which may affect their family planning choices.
- **Beliefs about a Parent Being a Stay-at-home Parent:** This variable captures individuals' attitudes and beliefs regarding parental roles in childcare and family dynamics.


## Repository Structure

- **data:** Contains the datasets used in the analysis, including the GSS survey data.
- **notebooks:** Jupyter notebooks containing the code for data preprocessing, exploratory data analysis, model training, and evaluation.
- **models:** Saved models generated during the training process.
- **ChildPredictor.py:** This script is a console-based application designed to interactively prompt users with relevant questions for predicting the number of children they might have.


## Conclusion

This project demonstrates the application of supervised machine learning techniques to predict the number of children individuals might have based on their religiousness, religion, and idealized family size. By leveraging data from the GSS survey conducted by NORC, valuable insights into the factors influencing family size in the United States can be gained. Admittedly, more could have been done and more variables are needed to be able to determine this number with more accuracy. With an accuracy of ±1 and an R-squared of 0.2, there is much more work needed to be done.
