# UNIL_Geneva_DSML

![Black and White Monthly News Email Header](https://github.com/thebrisly/UNIL_Geneva_DSML/assets/84352348/fe6feaca-7e43-4f0d-9ffa-d2ec38c5416b)


## 1. Introduction
Welcome to the UNIL_Geneva_DSML (Data Science and Machine Learning) project! 

Our primary goal is to develop and test various machine learning models on French text data to determine the level of difficulty. For instance, given a sentence like "J'aime les pommes" (I like apples), our model aims to predict the corresponding language proficiency level, such as A1.

## 2. Objective
The main objective of this project is to explore and implement different machine learning algorithms to effectively categorize the difficulty levels of French texts. By leveraging a diverse range of models, we seek to enhance our understanding of how well these algorithms can perform in the context of language proficiency assessment. Once we implemented a great algorithm, we need to create a real-world application with a user-friendly interface. 

## 3. Methodology
To achieve our goal, we split things up into 4 different stages :

### 3.1 Data Collection & Preprocessing
Our initial step involved a comprehensive analysis of the dataset, examining variables, and gaining a global understanding of its structure. 

After careful consideration, we made a strategic decision not to preprocess the data. The rationale behind this choice was to work with the dataset in its entirety, preserving its original form and ensuring that our models are exposed to the unaltered linguistic nuances present in the French texts. This approach aims to maintain the integrity of the data and assess the models on their ability to handle real-world, unprocessed language data effectively.

### 3.2 Model selection & Architecture
Experimenting with various machine learning models, such as natural language processing (NLP) models, deep learning architectures, and traditional classifiers.

#### 3.2.1 Initial models
We commenced our experimentation with fundamental machine learning models commonly encountered in class, including Linear Regression, Decision Tree, KNN Neighbours, and Random Forest. The table below outlines the performance metrics achieved by each model through training and evaluation on our prepared dataset:

| Models    | Linear Regression | Decision Tree | KNN Neighbours | Random Forest |
|:---------:|:---------:|:---------:|:---------:|:---------:|
| Precision | 0.4753  |  0.3097 | 0.3972 | 0.4194
| Recall    | 0.4745  |  0.3131  | 0.3160 | 0.4177
| F1-score | 0.4694 | 0.3102 | 0.2948 | 0.4045 
| Accuracy | 0.4760 | 0.3146 | 0.3198 | 0.4188

In our case, the Linear Regression model is by far the most performant of the basic models. It has the highest metrics where the true positives are maximized, as well as the false negatives and false positives that are minimized. Considering the simple and linear relation between the wording and the language level,  the use of more complex models, such as decision trees and random forest are less reasonable, and in our case less efficient. 

#### 3.2.2 Advanced model
After trying out basic models, we realized we needed something more advanced for better results. So, we did some research and found BERT models, which are really good at understanding language. This step was important for making our language proficiency assessment work even better.

### 3.3 Optimization with CamemBERT
After implementing CamemBERT, an advanced language model, we aimed to optimize our language proficiency assessment. Here are the performance metrics for CamemBERT:

| Models    | CamemBERT
|:---------:|:---------:|
| Precision | [enter value] 
| Recall    | [enter value] 
| F1-score | [enter value]
| Accuracy | [enter value] 


### 3.4 Development of a User-Friendly Interface
Creating a small website that uses our model and can predict the level of any sentence given to it (with a % of failure - of course).

## 4. Improvements

## 5. Conclusion


## 6. Team
This project has been created by Lisa Hornung & Laura Fabbiano at the University of Lausanne (Unil) during the course "Data Science & Machine Learning" for the Master in Information System.
