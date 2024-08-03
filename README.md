# Breast Cancer Classification

This project aims to classify breast cancer tumors as either malignant or benign using machine learning techniques.
The dataset used contains features extracted from cell nuclei in breast cancer patients.The primary algorithm employed for classification is XGBoost, 
with additional models like Logistic Regression and Random Forest being used for comparison.

# Table of Contents
- Dataset
- Installation
- Usage
- Model
- Evaluation
- Results
- Conclusion
- Contributing
- License

# Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset consists of 569 instances, each representing a breast tumor case. It contains 32 columns, including one target variable (`diagnosis`) and 31 features.
The features are derived from digitized images of fine needle aspirates (FNAs) of breast masses, detailing characteristics such as mean, standard error, and worst-case values of measurements like radius, texture, perimeter, area, and smoothness. 
The target variable indicates whether the tumor is malignant (M) or benign (B). This dataset is commonly used to train and evaluate machine learning models for breast cancer detection.

# Installation
To run this project, you need to have Python installed along with the necessary libraries.

# Usage
1. **Loading the Dataset**
   The dataset is loaded using the `pandas` library, and the first 20 rows are displayed to understand the structure.

   ```python
   df = pd.read_csv("data.csv")
   df.head(20)
   ```

2. **Data Exploration** 
   - Count the number of rows and columns.
   - Check for missing values.
   - Describe the dataset for summary statistics.
   - Visualize the count of malignant and benign cases using a count plot.

   ```python
   df.shape
   df.isnull().sum()
   df.describe()
   sns.countplot(df['diagnosis'], label='count')
   ```

3. **Data Preprocessing**  
   - Encode categorical data (diagnosis: M=1, B=0).
   - Create pair plots to visualize relationships between features.
   - Visualize correlation between features using a heatmap.

   ```python
   from sklearn.preprocessing import LabelEncoder
   df.iloc[:,1] = LabelEncoder_Y.fit_transform(df.iloc[:,1].values)
   sns.pairplot(df.iloc[:,1:6], hue="diagnosis")
   sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='.0%')
   ```

4. **Feature Scaling**  
   Standardize the feature values to have a mean of 0 and a standard deviation of 1.

   ```python
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.fit_transform(X_test)
   ```

5. **Model Building**  
   Three models are trained and evaluated:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier (Primary model)

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from xgboost import XGBClassifier

   # Function to train models
   def models(X_train, Y_train):
       log = LogisticRegression(random_state=0)
       forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
       log.fit(X_train, Y_train)
       forest.fit(X_train, Y_train)
       return log, forest

   model = models(X_train, Y_train)
   xgb = XGBClassifier(random_state=42)
   xgb.fit(X_train, Y_train)
   ```

## Evaluation
The models' performance is evaluated using accuracy scores. The XGBoost model, being the primary model, shows the highest accuracy.

```python
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

## Results
- Logistic Regression Training Accuracy 
- Random Forest Classifier Training Accuracy
- XGBoost Training Accuracy

## Conclusion
The project demonstrates the effectiveness of using machine learning algorithms for breast cancer classification. 
The successful implementation of XGBoost model, in particular is done.
achieved a perfect accuracy score on the training data.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

