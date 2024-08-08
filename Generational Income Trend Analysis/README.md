# **Generational Income Trend Analysis**

# Objective
This project aimed to examine the interrelation of age and income among different generations, providing insights to aid policy formulation for varied age demographics. We sought to understand how factors such as age, education, and work hours contribute to income levels, with a focus on generational differences.
Methodology
We utilized data from the General Social Survey (GSS), which provides a representative sample of the U.S. adult population through multistage sampling. Our analysis involved several key steps:

# Exploratory Data Analysis (EDA):

Processed invalid responses and NA values
Applied log transformation to relevant variables
Handled categorical variables


Ordinary Least Squares (OLS) Regression:

Estimated the relationship between age, education, work hours, and income
Utilized robust standard errors to account for potential heteroscedasticity


# Model Comparison:

Developed four different models to explore various aspects of the relationships
Used Stargazer tables for clear presentation of results



# Tools and Technologies:

R for statistical analysis and modeling
Stargazer package for creating publication-quality regression tables

# Results
Key findings from our analysis include:

A statistically significant positive relationship between age and income
The variance in income explained by age, education, and work hours ranged from 3.3% to 30.2% (adjusted R-squared)
Age consistently emerged as a predictor of income, likely reflecting career progression

# Significance/Impact
This project provides valuable insights into the complex relationships between age, education, work hours, and income:

Policy Implications: The findings suggest that policy approaches to economic well-being should be multi-faceted, considering factors beyond just age.
Educational Importance: The significant role of education in income determination underscores the importance of skill development and lifelong learning initiatives.
Generational Perspectives: By examining income across generations, the project offers a nuanced view of economic well-being that can inform targeted interventions and support programs.
Methodological Insights: The project demonstrates the application of OLS regression in socio-economic analysis, including the use of robust standard errors to address potential issues like heteroscedasticity.

Limitations and Future Directions
We acknowledged several limitations in our approach:

Potential multicollinearity and non-linearity in the regression model
Observed heteroscedasticity and non-normal distribution of residuals
Possible omitted variable bias, not accounting for factors like industry or geography

Future research could address these limitations by:

Exploring non-linear modeling techniques
Incorporating additional relevant variables
Employing more advanced econometric methods to handle complex data structures

This project showcases the application of statistical analysis to real-world socio-economic questions, demonstrating the ability to derive meaningful insights from complex datasets while acknowledging the limitations of the chosen methodological approach.
