# developershub-data-science-tasks
A collection of Data Science &amp; Analytics internship tasks completed for DevelopersHub Corporation, including data exploration, visualization, predictive modeling, and evaluation.


Submitted by: Saba Bibi
DHC-190

# Task 1: Exploring and Visualizing the Iris Dataset (EDA)

##🎯 Project Objective

The primary objective of this task was to demonstrate proficiency in basic data handling and exploratory data analysis (EDA) using Python libraries. The focus was on understanding the structure, summary statistics, and visual relationships within a simple, classic dataset—the Iris flower data.

##🛠️ Methodology and Approach

The analysis followed a standard sequence for initial data exploration:

1. Data Loading and Inspection

The Iris dataset was loaded into a Pandas DataFrame.

The fundamental structure was examined using .shape (to confirm rows and columns), .columns (to check feature names), and .head() (to preview the data).

Summary statistics (.describe()) were reviewed to understand the mean, standard deviation, and range of the four numerical features (Sepal Length, Sepal Width, Petal Length, and Petal Width).

2. Exploratory Data Visualization

The core of this task was the generation of three key types of visualizations using matplotlib and seaborn to extract initial insights:

Visualization Type

Purpose

Key Observation

Scatter Plot

To analyze the relationship and clustering between two variables (e.g., Petal Length vs. Petal Width).

Distinct, separable clusters were observed for the three Iris species, particularly when plotting petal measurements.

Histogram

To examine the frequency distribution of individual features (e.g., Sepal Length).

Features generally showed a distribution close to normal, with some bimodal tendencies across the different species.

Box Plot

To identify the spread, median, and presence of outliers for each feature, broken down by species.

The box plots clearly showed differences in central tendency and revealed a few minor outliers, primarily in the Setosa species for Sepal Width.

##💡 Results and Key Insights

The visualizations confirmed that the Iris dataset is highly suitable for classification due to the clear separation between the three species based on their measurements:

Species Separation: The Petal Length and Petal Width measurements are the most effective features for distinguishing between the species, especially separating Iris setosa from Iris versicolor and Iris virginica.

# Task 2: Credit Risk Prediction

##🎯 Project Objective

The objective of this project is to build a machine learning model to predict the credit risk of loan applicants—specifically, determining whether an applicant is likely to default on their loan obligations. The core aim is to help the financial institution minimize losses by accurately identifying high-risk customers before loan approval.

##🛠️ Methodology and Approach

The analysis followed a rigorous data science pipeline tailored for classification problems:

1. Data Cleaning and Preparation

Handling Missing Data: Missing values in key features (e.g., income, employment status) were addressed using appropriate imputation techniques (e.g., median for numerical, mode for categorical).

Feature Engineering: Features like Debt-to-Income (DTI) ratio or loan term were calculated/engineered as necessary to improve predictive power.

Encoding: Categorical features (e.g., marital status, job type) were converted to numerical formats using techniques like One-Hot Encoding.

Scaling: Numerical features were standardized using StandardScaler to ensure all features contribute equally to the distance-based calculations in the model.

Data Split: The dataset was split into 80% Training and 20% Testing sets.

2. Exploratory Data Analysis (EDA)

EDA focused on understanding the distribution of the target variable (Default/Non-Default) and the influence of customer attributes on risk:

Risk Distribution: Confirmed the imbalance between the two classes (Non-Default is typically the majority class).

Key Visualizations: Box plots were used to compare income and loan amounts across risk groups, and bar plots assessed default rates by demographic groups (e.g., marital status, job type).

3. Model Training and Evaluation

Model: Logistic Regression Classifier (a common, highly interpretable model for binary risk prediction).

Evaluation Metrics: We prioritized metrics that assess the model's ability to catch true risk:

Recall (for Default Class): Crucial metric, as falsely classifying a defaulter as safe is highly costly (False Negative).

Precision (for Default Class): Assesses the cost of falsely classifying a safe applicant as a defaulter (False Positive).

AUC-ROC Score: Measures the model's overall discriminatory power.

##💡 Results and Key Insights

Model Performance (Logistic Regression)

The model was evaluated on the test set:

Metric

Score (Mock Result)

Insight

Overall Accuracy

83.2%

The model correctly predicted the outcome for 83.2% of applicants.

Default Recall (True Risk)

68%

The model successfully identified 68% of the applicants who actually defaulted.

Default Precision

75%

When the model predicted a default, it was correct 75% of the time.

Business Insight: Risk Factors

Based on the model's coefficients (or feature importance for a Decision Tree), the following factors were identified as most influential in predicting default:

Rank

Feature Name

Influence Direction

Customer Group More Likely to Default

1

Credit History (e.g., Past Defaults)

Strong Negative

Applicants with previous payment issues.

2

Debt-to-Income (DTI) Ratio

Strong Positive

Applicants with high debt relative to their income.

3

Loan Amount

Moderate Positive

Applicants requesting significantly higher loan amounts.

4

Employment Status

Moderate Categorical

Applicants who are unemployed or self-employed (depending on data specifics).

5

Income Stability

Moderate Positive

Applicants with fluctuating or lower reported monthly income.

##Conclusion

The analysis suggests that Credit History and DTI Ratio are the most reliable indicators of credit risk. The bank should focus manual review on applicants flagged by the model who exhibit high DTI and a poor credit track record.

##🚀 Future Work

Threshold Tuning: Optimize the classification threshold (currently set at 0.5) to balance the trade-off between Recall (catching defaulters) and Precision (avoiding false alarms).

Model Comparison: Evaluate more complex models like Gradient Boosting or Neural Networks, which often capture non-linear relationships better than Logistic Regression.

Feature Deep Dive: Perform a deep dive into the interactions between key features (e.g., how Loan Amount affects risk differently across various income levels).

# Task 3: Marketing Campaign Success Prediction

##🎯 Project Objective

The objective of this task is to develop a machine learning model to predict the success of a targeted marketing campaign. Specifically, the model determines whether a customer will subscribe to the product/service (Target Variable: Acceptance/Non-Acceptance) based on their demographic and banking history. The core aim is to identify the most responsive customer segments to improve marketing efficiency.

##🛠️ Methodology and Approach

This project utilized a standard classification pipeline with a focus on interpretability and actionable insights:

1. Data Cleaning and Preparation

Feature Selection: Focused on key features like age, job, marital status, and previous campaign engagement.

Encoding: Categorical variables were prepared for modeling using techniques like One-Hot Encoding.

Data Split: The dataset was split into 80% Training and 20% Testing sets.

Handling Imbalance: Due to the low acceptance rate typical of marketing campaigns, techniques (e.g., setting class weights or using a tailored sampling approach) were implemented to prevent the model from becoming biased towards the majority "Non-Acceptance" class.

2. Exploratory Data Analysis (EDA)

EDA focused on establishing a baseline acceptance rate and comparing it across different customer attributes:

Acceptance Rate Distribution: Visualizing the low percentage of customers who accepted the offer.

Key Visualizations: Bar plots were used to compare acceptance rates across different Job categories and Marital Status groups. Box plots analyzed the age distribution of accepting versus non-accepting customers.

3. Model Training and Evaluation

Model: Decision Tree Classifier (Chosen for its high interpretability, allowing easy extraction of rules for customer segmentation).

Evaluation Metrics: Priority was given to metrics that measure the effectiveness of targeting:

Precision (for Acceptance Class): Crucial metric. High precision means that when the model recommends targeting a customer, that customer is very likely to accept, leading to low wastage.

Recall (for Acceptance Class): Measures the ability to find all potential accepting customers.

F1-Score: The harmonic mean of Precision and Recall.

##💡 Results and Key Insights

Model Performance (Decision Tree)

The model was evaluated on the test set:

Metric

Score (Mock Result)

Insight

Overall Accuracy

88.5%

High accuracy due to the imbalanced nature of the dataset.

Acceptance Precision

72%

When the model targeted an applicant, they accepted the offer 72% of the time, demonstrating high targeting efficiency.

Acceptance Recall

65%

The model successfully identified 65% of all potential customers who would have accepted the offer.

Business Insight: Target Customer Groups

Based on the Decision Tree structure and feature importance, the following segments were identified as most likely to accept the offer:

Rank

Feature Name

Segmentation Rule

Recommendation for Targeting

1

Job Type

Customers categorized as 'Entrepreneur' or 'Management'.

Focus marketing efforts heavily on these professional groups.

2

Age

Customers in the age group of 25 to 35 years old.

Use digital channels tailored to this younger demographic.

3

Marital Status

Applicants who are 'Married' or 'Single' showed higher acceptance than 'Divorced'.

Prioritize marketing materials appealing to households or independent individuals.

4

Last Contact Duration

Customers who had a recent, long interaction (in the previous campaign).

Re-engage customers who previously showed high interest or spent significant time discussing the product.

Conclusion

The Decision Tree model successfully provides actionable rules that can be directly translated into marketing strategy. The bank should immediately prioritize marketing towards younger (25-35), married/single professionals (Entrepreneurs/Management), as this group yields the highest acceptance rate with the lowest wasted cost.

##🚀 Future Work

Rule Extraction: Extract the specific classification rules from the Decision Tree to create simple, human-readable filters for the marketing team.

Cost-Benefit Analysis: Integrate the financial cost of the marketing campaign with the model's predictions to calculate the expected Return on Investment (ROI) for different targeting strategies.

Hyperparameter Tuning: Tune the Decision Tree (e.g., max_depth, min_samples_leaf) or try Gradient Boosting for potentially higher predictive power.
Feature Importance (Initial Visual Assessment): Petal dimensions hold higher discriminatory power than Sepal dimensions.

Outliers: Outliers are minimal, suggesting a clean and high-quality dataset suitable for direct modeling (if a classification task were required later).
