# Churn-Analysis-with-Supervised-and-Unsupervised-Learning
This project analyzes churned user behavior, provides actionable recommendations to reduce churn, builds a fine-tuned machine learning model for churn prediction, and segments users into groups based on behavior. Insights are used to design targeted promotions for each group.
# Introduction
Customer churn poses significant challenges for businesses, impacting not only revenue but also long-term growth and customer acquisition costs. Retaining existing customers is often more cost-effective than acquiring new ones, making churn analysis a crucial aspect of business strategy. Churned users, those who discontinue their association with the company, can reveal critical insights into service gaps, customer dissatisfaction, or competitive pressures. By understanding the patterns and behaviors of these customers, businesses can proactively address underlying issues, refine their offerings, and implement strategies that enhance customer loyalty.

Analyzing churned users is not just about reducing attrition; it also opens up opportunities to increase customer lifetime value and improve brand reputation. Effective churn management can lead to personalized interventions, targeted promotions, and improved customer satisfaction, ultimately driving better business outcomes.

The dataset used for this analysis includes variables such as customer demographics, purchase history, satisfaction scores, and activity levels. Key features like Churn status, Tenure, PreferredPaymentMethod, and SatisfactionScore provide a comprehensive view of customer behavior and preferences, forming the basis for detailed segmentation and predictive modeling.
# Objectives
- *Analyze the patterns and behavior of churned users:* Identify the factors and behaviors common among churned customers and suggest strategies to reduce churn.

- *Build a Machine Learning model for predicting churned users:* Develop and fine-tune a predictive model to forecast customer churn based on the available data.

- *Segment churned users based on behavior:* Group churned users by their behaviors and characteristics, highlighting differences between these groups, to inform targeted retention strategies and promotional offers.
# Processing
## Prepare the data & EDA
Conect data from DriveConnect to data from my Google Drive, specifically an Excel file containing the dataset for churned users analysis.
``` python
rom google.colab import drive
drive.mount('/content/drive')

churn_predicting_source = pd.read_excel('/content/drive/MyDrive/Machine Learning/churn_prediction.xlsx')
churn_predicting = churn_predicting_source.copy()
```
Examine data characteristics, including the data type of each column, and check for the presence of null, missing, or error values using the dataprep.eda library and the create_report method.
``` python
!pip install dataprep
from dataprep.eda import create_report
create_report(churn_predicting)
```
After connecting to the dataset and performing an initial inspection, several key observations and insights can be made:

- The dataset contains a total of 20 columns, with 9 numerical columns and 11 categorical columns, spread across 5,630 rows. This structure provides a diverse set of features that could be leveraged for both predictive and classification tasks.

- Missing data accounts for approximately 1.6% of the dataset, which translates to about 1,856 cells. While the amount of missing data is relatively small, the fact that the "Churn" column—critical for predicting customer behavior—has no missing values is a major advantage. The completeness of this column ensures that the core target variable for any machine learning models is fully intact, making it much easier to build reliable predictive models or perform clustering without dealing with significant data gaps.

- The missing data primarily resides in the columns with floating-point data types (float64), which is common for real-valued variables, and these columns also feature discrete values. Given the nature of this missing data, it would be prudent to apply imputation techniques to fill in these gaps, ensuring that machine learning algorithms can function optimally without bias from incomplete data.

- Notably, the dataset contains no duplicate rows, which is a positive aspect as it avoids over-representation of certain data points and ensures the integrity of model training.

- The dataset is also assumed to be free from outliers, which simplifies the analysis by eliminating the need for additional outlier detection or cleaning steps. However, it would still be beneficial to run a quick check for any anomalies that might have been overlooked during the initial inspection.
## Predict churned users by using supervised learning model
In Supervised Learning models, it is crucial for columns containing data to be in numerical format (typically int64 or float64) to ensure accurate predictions. This often necessitates the process of encoding categorical variables into numerical values.

There are two effective methods for encoding categorical data. One approach involves manually mapping the categorical values to numeric codes, such as converting categories into 0, 1, 2, 3, and so on. While this method can be very effective, it requires the programmer to perform additional steps, making it more complex and time-consuming.

An alternative, simpler method is to use the get_dummies function from the Pandas library.
``` python
churn_predicting_dummies = pd.get_dummies(churn_predicting)
```
This method automatically converts categorical variables into a series of binary (0 or 1) columns, each representing one category. This method is straightforward, efficient, and widely used in data preprocessing for machine learning, as it significantly reduces the time and effort needed to prepare the data for training.

In applying a Supervised Learning model to predict future customer churn, the selection of relevant features plays a critical role in the accuracy and effectiveness of the model.

Notably, certain columns are excluded from the training dataset, such as CustomerID and Churn. The Churn column, representing whether a customer has left or not, serves as the target variable, making it unnecessary in the input features. Meanwhile, CustomerID is a system-generated identifier with no inherent relationship to customer behavior or churn. For instance, there is no meaningful connection between an even-numbered ID and a higher likelihood of churn compared to an odd-numbered one, rendering this column irrelevant to the prediction task.

The remaining columns, however, contain valuable information that can significantly influence churn predictions. CityTier, for example, suggests that customers in higher-tier cities (e.g., Tier 1) may have a higher likelihood of churn, possibly due to factors such as more available alternatives or differing service expectations. WarehouseToHome, representing the distance between the customer’s residence and the warehouse, is another important feature. A greater distance may lead to lower engagement with the ordering process, thus increasing the chances of churn. Additionally, HourSpendOnApp, SatisfactionScore, and Complain provide strong indications of customer loyalty. For instance, customers who spend more time on the app and rate their satisfaction highly are generally more engaged and less likely to churn, while those who frequently complain may signal dissatisfaction and an increased risk of leaving.

Incorporating columns like Gender and MaritalStatus further enriches the model. While these variables might not directly predict churn, they can offer additional insights into customer behavior. Married customers, for example, may demonstrate higher loyalty, as they could have more stable purchasing habits, while female customers might exhibit stronger buying patterns at specific times of the year. Therefore, despite not being directly related to churn, these columns are still useful in understanding broader customer trends and ensuring the model captures a comprehensive view of customer behavior.
``` python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

X = churn_predicting_dummies.drop(['CustomerID', 'Churn'], axis=1).values
y = churn_predicting_dummies['Churn'].values
```
Next, the dataset needs to be split into the desired ratio, typically 80% for the training data and 20% for the testing data. This division ensures that the model is trained on a large enough portion of the data while also being evaluated on a separate, unseen subset to test its generalizability.
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
```
Before training the model, it is essential to address any missing values. Since the columns with missing values are numeric and specifically have a float64 data type, the most effective approach is to use the SimpleImputer method. In this case, missing values (represented as NaN) will be imputed with the mode—the most frequent value in that column. This imputation strategy helps ensure that the dataset is complete, preventing issues during model training and enabling the model to learn more effectively from the available data.
``` python
imp = SimpleImputer(strategy='most_frequent')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
```
To optimize the machine learning model, it's important to standardize the data using StandardScaler. This transforms the values into z-scores, with a mean of 0 and a standard deviation of 1, ensuring all features are on the same scale and improving the model's performance.
``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
The most important step in predicting customer churn is selecting the right model. While several models can be suitable, it’s essential to choose the best one. Each model has its own set of parameters that need to be tuned before training, as the chosen model may lead to issues like Overfitting or Underfitting. Therefore, it’s crucial to test different models to identify the best fit.

Since the target variable is categorical (0 or 1), three models are suggested: KNeighborsClassifier, DecisionTreeClassifier, and RandomForestClassifier. To optimize the model parameters, the training data will be split into smaller parts for increased experimentation using KFold cross-validation. The optimal model will be chosen based on the highest Balanced Accuracy score.
``` python
kf = KFold(n_splits=5, shuffle=True, random_state=85)
grid_search_knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}, cv=kf, scoring='balanced_accuracy')
grid_search_knn.fit(X_train_scaled, y_train)

param_grid_dtc = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dtc = GridSearchCV(RandomForestClassifier(), param_grid_dtc, cv=kf, scoring='balanced_accuracy')
grid_search_dtc.fit(X_train_scaled, y_train)

param_grid_rfc = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search_rfc = GridSearchCV(RandomForestClassifier(), param_grid_rfc, cv=kf, scoring='balanced_accuracy')
grid_search_rfc.fit(X_train_scaled, y_train)
```

The next step is to automate the model selection process and apply it to the current dataset. Automation is necessary because the optimization of models can vary with each run of the program, resulting in different Balanced Accuracy scores each time. By automating the selection, the process ensures consistent and efficient model evaluation, allowing for the identification of the best-performing model based on the current data, without the need for manual intervention.
``` python
dic = {
    'knn': {'param': grid_search_rfc.best_params_,
            'best_score': grid_search_knn.best_score_},
    'dtc': {'param': grid_search_dtc.best_params_,
            'best_score': grid_search_dtc.best_score_},
    'rfc': {'param': grid_search_rfc.best_params_,
            'best_score': grid_search_rfc.best_score_}
}
best_score = [grid_search_knn.best_score_, grid_search_dtc.best_score_, grid_search_rfc.best_score_]
for i in dic.keys():
  if dic[i]['best_score'] == max(best_score):
    best_model = i
  else:
    continue

if best_model == 'knn':
  print("Model will be used: KNN")
  model_using = KNeighborsClassifier()
  model_using.set_params(**dic[best_model]['param'])
elif best_model == 'dtc':
  print("Model will be used: Decision Tree Classifier")
  model_using = DecisionTreeClassifier()
  model_using.set_params(**dic[best_model]['param'])
else:
  print("Model will be used: Random Forest Classifier")
  model_using = RandomForestClassifier()
  model_using.set_params(**dic[best_model]['param'])

model_using.fit(X_train_scaled, y_train)
y_pred = model_using.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
```
Once the model is trained and optimized, when new data needs to be predicted, it can simply be fed into the predict method. The model will then return the predicted results, providing an efficient way to forecast outcomes without the need for retraining each time new data is available.
## Segment these churned users into groups by using unsupervised learning model
For the clustering task related to predicting customer churn, the initial data preparation steps are similar to those used in the prediction task:

- First, categorical values should be converted to numerical values using get_dummies.

- Handle missing values with SimpleImputer to ensure that the dataset is complete.

- Normalize the data by transforming it into z-scores to standardize the features.

Additionally, since this is a clustering task focusing on customer churn, we will filter the dataset to include only customers who have churned, i.e., those with churn = 1. This step helps to isolate the relevant data for clustering and ensures that the model focuses on the group of users who have left.

Clustering involves categorizing customers into different groups, and the criteria for this classification are not directly known; it is often referred to as a "black box" in machine learning. However, one thing that can be controlled is the input data used for clustering.

In the current dataset, there are too many columns (20 columns), and if all are used (including the Churn column), it may be difficult to distinguish between clusters or identify meaningful differences based on the available features. To address this, Principal Component Analysis (PCA) is applied.

The goal of PCA is to identify the directions (principal components) in which the data varies the most and project the data onto these directions. This reduces the number of necessary features while retaining most of the important information. The key aspect of PCA is that it **preserves features with the highest variance**, ensuring that the most significant variations in the data are kept for further analysis.

![PCA_Churn_Users.png](https://github.com/khangtran85/Churn-Analysis-with-Supervised-and-Unsupervised-Learning/blob/main/PCA_Churn_Users.png)

From the PCA plot, it is necessary to select the optimal number of principal components for clustering. The analysis reveals that five principal components have variances greater than 1.5. However, for simplicity and standard practice, two principal components will be chosen. This means the original data can be represented in a two-dimensional space while retaining most of its variability.

Furthermore, based on the elbow method, three clusters are identified as the ideal number. Choosing a smaller number of clusters not only makes differentiation more straightforward but also enhances practical applications. This approach allows for more targeted and cost-effective strategies when implementing campaigns tailored to each cluster.

![Number_of_cluster.png](https://github.com/khangtran85/Churn-Analysis-with-Supervised-and-Unsupervised-Learning/blob/main/Number_of_cluster.png)

The result is a visualized cluster plot:

![Clusters_char.png](https://github.com/khangtran85/Churn-Analysis-with-Supervised-and-Unsupervised-Learning/blob/main/Clusters_char.png)

The final step (or perhaps an additional step) is to identify which cluster is the most significant.
``` python
df_pca2 = pd.DataFrame(pca_data_2, columns=['PC1', 'PC2', 'PC3'])
df_pca2['Cluster'] = labels_2

X_2 = df_pca2.drop('Cluster', axis=1)
y_2 = df_pca2['Cluster']

from sklearn.metrics import classification_report
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=5)
rf2 = RandomForestClassifier(n_estimators=100, random_state=85)
rf2.fit(X_train_2, y_train_2)
y_pred_2 = rf2.predict(X_test_2)
print(classification_report(y_test_2, y_pred_2))

importances = rf2.feature_importances_
features_names = X_2.columns
feature_importances = sorted(zip(importances, features_names), reverse=True)

for importance, feature_name in feature_importances:
    print(f"{feature_name}: {importance}")
```
# Recommendation
- For churn prediction, as previously analyzed, the model's ability to process new datasets for forecasting is highly advantageous. It allows businesses to proactively identify customers at risk of churning. By recognizing these at-risk individuals, companies can evaluate engagement trends and implement targeted strategies to strengthen user retention. For instance, offering personalized incentives, loyalty rewards, or exclusive promotions could help re-engage these customers. Additionally, enhancing customer service and addressing complaints promptly may foster a deeper sense of trust and satisfaction, increasing the likelihood of retaining these users as loyal customers.

- In the case of user segmentation, distinguishing the unique characteristics of each cluster remains a complex challenge due to the "black box" nature of clustering algorithms. Although the reasoning behind the segmentation might be unclear, supplementary analysis can help identify the most impactful clusters. By understanding these key clusters, businesses can tailor their strategies to cater to the specific needs and behaviors of each group. For example, high-value customers might benefit from premium services or early access to products, while less engaged groups could be targeted with educational campaigns or incentives to explore new features. By customizing strategies for each cluster, organizations can enhance user experiences and maximize overall customer value.
