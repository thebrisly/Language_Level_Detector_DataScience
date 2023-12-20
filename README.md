# UNIL_Geneva_DSML

<sub><small> 0. Run this project at home

<sub>You can run the different models we've created at home. To do so, run the following commands : </sub>

```git clone git@github.com:thebrisly/UNIL_Geneva_DSML.git```

<sub>Then via your terminal, go on the folder directory (hint : you can use cd to easily change the directory) and go on the model directory of your choice. For example : </sub>

``` cd 2_MODELS/1_logistic_regression ```

<sub>Then, run the python programm that you selected, like this : </sub>

<sub> ``` python3 logistic_regression1.py ``` Make sure to have all the libraries (sklearn, pandas, pytorch, streamlit...)  installed before running something :-)  </sub>

<sub> Enjoy ! </small></sub>

-----------------------------------------------------

![Black and White Monthly News Email Header](https://github.com/thebrisly/UNIL_Geneva_DSML/assets/84352348/fe6feaca-7e43-4f0d-9ffa-d2ec38c5416b)

[ LINK TO THE YOUTUBE VIDEO HERE ](mettre_le_lien_ici)

## 1. Introduction
Welcome to the UNIL_Geneva_DSML (Data Science and Machine Learning) project! 

Our primary goal is to develop and test various machine learning models on French text data to determine the level of difficulty. For instance, given a sentence like "J'aime les pommes" (I like apples), our model aims to predict the corresponding language proficiency level, such as A1.

## 2. Objective 🎯
The main objective of this project is to explore and implement different machine learning algorithms to effectively categorize the difficulty levels of French texts. By leveraging a diverse range of models, we seek to enhance our understanding of how well these algorithms can perform in the context of language proficiency assessment. Once we implemented a great algorithm, we need to create a real-world application with a user-friendly interface. 

## 3. Methodology
To achieve our goal, we split things up into 3 different stages :

### 3.1 Data Collection & Preprocessing
Our initial step involved a comprehensive analysis of the dataset, examining variables, and gaining a global understanding of its structure. 

After careful consideration, we made a strategic decision not to preprocess the data. The reason behind this choice was to work with the dataset in its entirety, preserving its original form and ensuring that our models are exposed to the unaltered linguistic nuances present in the French texts. 

This approach aims to maintain the *integrity* of the data and assess the models on their ability to handle real-world, unprocessed language data effectively.

### 3.2 Model selection & Architecture
In this section, we explored different types of machine learning models, such as natural language processing (NLP) models, deep learning architectures, and traditional classifiers.

#### 3.2.1 Initial models
We commenced our experimentation with fundamental machine learning models commonly encountered in class, including [Linear Regression](https://github.com/thebrisly/UNIL_Geneva_DSML/tree/main/2_MODELS/1_logistic_regression), [Decision Tree](https://github.com/thebrisly/UNIL_Geneva_DSML/tree/main/2_MODELS/2_decision_tree), [KNN Neighbours](https://github.com/thebrisly/UNIL_Geneva_DSML/tree/main/2_MODELS/3_knn), and [Random Forest](https://github.com/thebrisly/UNIL_Geneva_DSML/tree/main/2_MODELS/4_random_forest).


Each model was constructed consistently, employing a 20/80 test-train split and utilizing functions from the sklearn library for implementation. This approach ensured uniformity and comparability across our experiments.

After running each model, here are the results. The table below outlines the performance metrics achieved by each model through training and evaluation on our prepared dataset:

| Models    | Linear Regression | Decision Tree | KNN Neighbours | Random Forest |
|:---------:|:---------:|:---------:|:---------:|:---------:|
| Precision | 0.4753  |  0.3097 | 0.3972 | 0.4194
| Recall    | 0.4745  |  0.3131  | 0.3160 | 0.4177
| F1-score | 0.4694 | 0.3102 | 0.2948 | 0.4045 
| Accuracy | 0.4760 | 0.3146 | 0.3198 | 0.4188

In our case, the Linear Regression model is by far the most performant of the basic models. It has the highest metrics where the true positives are maximized, as well as the false negatives and false positives that are minimized. Considering the simple and linear relation between the wording and the language level,  the use of more complex models, such as decision trees and random forest are less reasonable, and in our case less efficient. 


| ![Image 1](https://github.com/thebrisly/UNIL_Geneva_DSML/blob/main/images/BarPlot%20Accuracies.jpg) | ![Image 2](https://github.com/thebrisly/UNIL_Geneva_DSML/blob/main/images/BarPlot%20F1-Scores.jpg) | ![Image 3](https://github.com/thebrisly/UNIL_Geneva_DSML/blob/main/images/BarPlot%20Precisions.jpg )| ![Image 4](https://github.com/thebrisly/UNIL_Geneva_DSML/blob/main/images/BarPlot%20Recalls.jpg) |
| --- | --- | --- | --- |

Linear regression works well for our project because it looks for simple linear relationships between words in our French sentences and language proficiency levels. This means that it is effective for understanding how language difficulty evolves in a direct way with linguistic features.

On the other hand, decision trees (the "worst" tested model) attempt to understand relationships in a more complex way, by creating more detailed rules for making decisions. In our case, where the relationship between language and difficulty can best be described by straight lines (as linear regression does), decision trees can sometimes be too complicated to capture these simple relationships. This can lead to poorer performance, as the model may overlearn less important details instead of focusing on general trends.


#### 3.2.2 Advanced model
After trying out basic models, we realized we needed something more advanced for better results. So, we did some research and found BERT models, which are really good at understanding language. We found "CamemBERT" which is the french version of the BERT model. 

This step was important for making our language proficiency assessment work even better and have a better score in the kaggle competition.

### 3.3 Optimization with CamemBERT 🧀

CamemBERT is a smart computer program designed to understand and work with the French language. It helps computers make sense of French text by learning patterns and relationships in the language.

#### 3.3.1 Tokenization and padding

To prepare the data for CamemBERT, we employed tokenization and padding techniques. 

Tokenizing the training data is a really important step because it transforms the textual information into a numerical format that the machine learning model can interpret. By assigning each word a unique identifier, we create a structured representation that enables the model to efficiently process and analyze language nuances. We used the CamemBERT Tokenizer to do that.


#### 3.3.2 Training and evaluation

The training process involved multiple epochs, allowing the model to adapt and learn the intricate patterns in the language data. We utilized the AdamW optimizer with a learning rate of 2e-5 for effective training. The performance of the optimized CamemBERT model was evaluated on a separate validation set, ensuring robustness and generalization.

#### 3.3.3 Results and insights

The optimized CamemBERT model demonstrated improved language proficiency classification compared to the initial basic models. We couldn't print different metric on the python script so we just took the accuracy from the kaggle board. For a more complete picture, we should also check precision, recall, and F1 score. 

With the camemBERT model we got this accuracy :

|  | Kaggle |
|:---------:|:---------:|
| Accuracy | 0.609 |

To achieve this score, here's what we did it:

##### ---------- Best training settings ---------- 
The final model that we used is composed of :
- Epochs: We conducted training over 8 iterations to impart extensive knowledge to the model.
- Learning Rate: Used a learning rate of 2e-5 to help the model learn just right.
- Optimizer: Picked AdamW to make our model even smarter.

These settings were selected through experimentation, providing the best balance for effective language proficiency assessment with our CamemBERT model.
  
##### ---------- Exploration of alternative approaches ---------- 
Before finding these parameters, we explored various alternative approaches in our quest for an optimized model:

- Dynamic learning rate adjustment
We experimented with a strategy to dynamically adjust the learning rate, aiming to adapt to the evolving knowledge of the model. The idea was to fine-tune the learning process as the model gained experience. However, to our surprise, this approach did not lead to improved results. After careful evaluation, we decided to exclude this strategy from our methodology, as it did not contribute positively to the language proficiency assessment task.

- Iterative experimentation
Our journey involved extensive experimentation, exploring different combinations of parameters, training iterations, and learning rate strategies. This iterative process was crucial in refining our model and identifying the most effective configuration for accurate language proficiency prediction.

Despite the diverse approaches explored, the finalized parameters mentioned earlier emerged as the most successful, providing a robust and reliable solution for our language assessment task. And even though we couldn't get all the metrics, the relatively high accuracy suggests our advanced language model did well in assessing language proficiency. 

This result can be explained with the fact that CamemBERT is better at understanding the French language than traditional models because it's designed specifically for that purpose. It learns the details and connections in the language really well.


## 4 Development of a User-Friendly Interface

<img src="https://github.com/thebrisly/UNIL_Geneva_DSML/assets/84352348/8234aae5-c532-411b-99ee-6d8b54949ef4" width="500">

In our project, we aimed to create a user-friendly interface to make our language proficiency model accessible to everyone. Initially, we tried using Streamlit, a tool for building interactive web applications with Python. However, we encountered a challenge – our final model turned out to be quite large, making it difficult to run the website smoothly.

We tried a number of different ways to overcome this challenge. After encountering difficulties loading the Streamlit tunnel, we decided to change our environment from Google Colab to Pycharm. Thanks to this change, we were able to run the user interface, but then encountered difficulties with the weight of the models too :-( 

After using other methods, such as uploading the file through different tunnels - GitHub for large files, Google Drive or SwissTransfer - we managed to get the user interface working, in part. Unfortunately, it's not possible to predict the language level, but it's still possible to see how beautiful it is! 


## 5. Improvements

*--> If we had more time, what would we do?*

To make our model even better, we could have tried a few things : 

First, we could have added more variety to our data through data augmentation. This means creating different versions of our sentences to help the model learn from a wider range of language styles. We tried to do it, but without any success - so we decided to delete it. 

Another trick, that has been recommendend on the Kaggle competition page, is using advanced techniques like advanced text embedding. We implemented a straightforward approach by tokenizing the text, but there's room for employing a more advanced technique for text embedding. This involves representing words in a way that helps the model understand their meanings better. These improvements could have bumped up our accuracy. We also tried a bit, but without any success (we didn't know how to implement it well).

Also, working with a larger and more diverse set of data could have helped our model become more adaptable to different kinds of language challenges. Trying out different settings for our model, testing out new ideas, and using more specific language resources could have been useful too. There are lots of things we could explore to make our model even more accurate and powerful.


## 6. Conclusion

To conclude, we explored different machine learning models for evaluating the difficulty levels of French texts, progressing from basic models like linear regression to advanced models like CamemBERT. Despite the difficulties encountered in obtaining all measurements, the optimized CamemBERT model showed a little improvement in accuracy (going from 0.470 with linear regression to 0.609), suggesting its effectiveness in evaluating language skills. 

Although our current accuracy is decent, we could still improve it by increasing the data, using advanced techniques such as text integration, and working with a larger and more diverse dataset.

Anyway. We enjoyed a lot this project and we're happy with the result, because we came 3rd out of 28 in the Kaggle competition 😎
<img width="1184" alt="image" src="https://github.com/thebrisly/UNIL_Geneva_DSML/assets/84352348/5051a767-e24d-4c60-910f-05966da42c2d">


## 7. Team
This project has been created by Lisa Hornung & Laura Fabbiano at the University of Lausanne (Unil) during the course "Data Science & Machine Learning" for the Master in Information System.
