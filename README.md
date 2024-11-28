# Customer-Churn-Prediction

## Algorithm Used :

<h2>Supervised Learning</h2>
--> Basically supervised learning is when we teach or train the machine using data that is well-labelled. <br><br>
--> Which means some data is already tagged with the correct answer.<br><br>
--> After that, the machine is provided with a new set of examples(data) so that the supervised learning algorithm analyses the training data(set of training examples) and produces a correct outcome from labeled data.<br><br>

<h3>i) K-Nearest Neighbors (KNN) </h3>
<br>
--> K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning.<br><br>
--> It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection..<br><br>
--> In this algorithm,we identify category based on neighbors.<br><br>

<h3>ii) Support Vector Machines (SVM) </h3>
<br>
--> The main idea behind SVMs is to find a hyperplane that maximally separates the different classes in the training data. <br><br>
--> This is done by finding the hyperplane that has the largest margin, which is defined as the distance between the hyperplane and the closest data points from each class. <br><br>
--> Once the hyperplane is determined, new data can be classified by determining on which side of the hyperplane it falls. <br><br>
--> SVMs are particularly useful when the data has many features, and/or when there is a clear margin of separation in the data.<br><br>

<h3>iii) Naive Bayes Classifiers</h3>
<br>
--> Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. <br><br>
--> It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.<br><br>
--> The fundamental Naive Bayes assumption is that each feature makes an independent and equal contribution to the outcome.<br><br>

<h3>iv) Decision Tree</h3>
<br>
--> It builds a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.<br><br>
--> It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is met, such as the maximum depth of the tree or the minimum number of samples required to split a node.<br><br>
--> The goal is to find the attribute that maximizes the information gain or the reduction in impurity after the split.<br><br>

<h3>v) Random Forest</h3>
<br>
--> It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.<br><br>
--> Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.<br><br>
--> The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.<br><br>


<h3>vi) Logistic Regression</h3>
<br>
--> Logistic regression is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class or not. <br><br>
--> It is a kind of statistical algorithm, which analyze the relationship between a set of independent variables and the dependent binary variables. <br><br>
--> It is a powerful tool for decision-making.<br><br>
--> For example email spam or not. <br>

---
# Libraries Used 📚 💻
<p>Short Description about all libraries used.</p>
To install python library this command is used- <br><br>

```
pip install library_name
```

<ul>
<li>NumPy (Numerical Python) – Enables with collection of mathematical functions
to operate on array and matrices. </li>
  <li>Pandas (Panel Data/ Python Data Analysis) - This library is mostly used for analyzing,
cleaning, exploring, and manipulating data.</li>
  <li>Matplotlib - It is a data visualization and graphical plotting library.</li>
<li>Scikit-learn - It is a machine learning library that enables tools for used for many other
machine learning algorithms such as classification, prediction, etc.</li>
  <li>Seaborn - It is an extension of Matplotlib library used to create more attractive and
informative statistical graphics.</li>
</ul>

