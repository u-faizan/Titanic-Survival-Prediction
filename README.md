# Titanic-Survival-Prediction


Problem Statement:
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. Of the 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters. One of the main goals is to build a predictive model that answers the question: “What sorts of people were more likely to survive?” using passenger data (i.e., name, age, gender, socio-economic class, etc.).

Objective:
The objective of this project is to develop a machine learning model capable of predicting the survival of passengers on the Titanic based on their characteristics such as age, gender, class, etc. The model aims to automate the prediction process, offering a practical solution for identifying passengers who were more likely to survive the disaster.

Project Details:

Key Features:
The essential characteristics used for prediction include:

Passenger Class (Pclass)
Gender (Sex)
Age
Number of Siblings/Spouses Aboard (SibSp)
Number of Parents/Children Aboard (Parch)
Ticket Fare (Fare)
Port of Embarkation (Embarked)
Machine Learning Model:
The project involves the creation and training of multiple machine learning models to accurately predict survival. The models used include Logistic Regression, Decision Tree, and Random Forest.

Significance:
This project demonstrates the application of machine learning techniques to historical data to predict outcomes and derive insights, which can have broader applications in various fields such as safety analysis, risk management, and historical data analysis.

Project Summary:

Project Description:
The Titanic Survival Prediction project focuses on developing a machine learning model to predict the survival of passengers on the Titanic based on specific characteristics. The project involves extensive data exploration, preprocessing, feature engineering, and model evaluation.

Objective:
The primary goal of this project is to leverage machine learning techniques to build a classification model that can accurately predict the survival of passengers on the Titanic. The model aims to automate the prediction process, offering a practical solution for identifying passengers who were more likely to survive the disaster.

Key Project Details:

Dataset: The dataset contains information about the passengers on the Titanic, including their socio-economic status (class), sex, age, port of embarkation, and other relevant attributes.
Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features to prepare the dataset for modeling.
Feature Engineering: Creating new features such as FamilySize to improve the model's performance.
Model Selection: Training and evaluating multiple machine learning models to select the best one based on performance metrics.
Results:
Accuracy was chosen as the primary evaluation metric for the Titanic Survival Prediction model. The final list of models and their accuracies are as follows:

Logistic Regression: 83.24%
Decision Tree: 81.56%
Random Forest: 85.78%
Conclusion:
In the Titanic Survival Prediction project, the Random Forest model has been selected as the final prediction model due to its highest accuracy. The project aimed to predict the survival of passengers on the Titanic based on various characteristics. After extensive data exploration, preprocessing, and model evaluation, the following conclusions can be drawn:

Data Exploration: Through a thorough examination of the dataset, insights were gained into the characteristics and distributions of features. Gender and passenger class were found to be significant factors influencing survival.
Data Preprocessing: Data preprocessing steps, including handling missing values and encoding categorical variables, were performed to prepare the dataset for modeling.
Model Selection: After experimenting with various machine learning models, the Random Forest was chosen as the final model due to its simplicity, interpretability, and good performance in predicting survival.
Model Training and Evaluation: The Random Forest model was trained on the training dataset and evaluated using accuracy metrics. The model demonstrated satisfactory accuracy in predicting the survival of passengers.
Practical Application:
The Titanic Survival Prediction model can be applied in real-world scenarios, such as historical data analysis and risk management, to predict outcomes based on specific characteristics.

How to Use:

Clone the repository:
bash
Copy code
git clone https://github.com/u-faizan/titanic-survival-prediction.git
cd titanic-survival-prediction
Place the Titanic dataset (titanic.csv) in the project directory.

Run the script:

Copy code
python titanic_survival_prediction.py
Dependencies:
The following Python packages are required to run the code:

pandas
matplotlib
scikit-learn
You can install the necessary packages using the following command:

Copy code
pip install pandas matplotlib scikit-learn
Author:
Umar Faizan

License:
This project is licensed under the MIT License.

