# Big-Mart-Sales-Detection
This project aims to predict the sales of products at various outlets of Big Mart using machine learning techniques. The dataset includes product information, outlet details, and sales data. We employ exploratory data analysis (EDA) and use an XGBoost Regressor to predict the target variable, Item_Outlet_Sales. 
### README: Big Mart Sales Detection

---

#### Project Description:
This project aims to predict the sales of products at various outlets of Big Mart using machine learning techniques. The dataset includes product information, outlet details, and sales data. We employ exploratory data analysis (EDA) and use an XGBoost Regressor to predict the target variable, `Item_Outlet_Sales`.

#### Necessary Files:
1. **Train.csv** - This CSV file contains the dataset, which includes various features about the items, outlets, and their sales. Ensure that this file is available in the same directory as the script.
2. **BigMartSalesPrediction.py** - The Python script that contains the model training, prediction, and evaluation code (based on the code provided).
3. **requirements.txt** (optional) - List of dependencies needed to run the project, including packages like pandas, numpy, seaborn, matplotlib, scikit-learn, and xgboost.

#### Steps to Run the Project:
1. **Install Dependencies**:  
   Before running the code, ensure you have the necessary Python libraries installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```
   or manually install the following:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost
   ```

2. **Load Dataset**:  
   Ensure the `Train.csv` file is available in the working directory. The script reads this dataset and uses it for analysis and prediction.

3. **Data Preprocessing**:  
   - Handle missing values in the `Item_Weight` and `Outlet_Size` columns.
   - Use label encoding to convert categorical variables into numeric form.

4. **Exploratory Data Analysis (EDA)**:  
   Visualize the distribution of important features such as `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Item_Outlet_Sales`, etc., using seaborn and matplotlib.

5. **Model Training**:  
   The features and target variable are split into training and test sets. XGBoost Regressor is used to train the model on the training data.

6. **Model Evaluation**:  
   The model's performance is evaluated using the R² score, both for training and test data.

#### Outputs:
- **Accuracy Scores**:  
   The model prints the R² score for both the training and test data, indicating how well the model has fit.

#### Key Functions:
- **train_test_split()**: Split the dataset into training and test sets.
- **XGBRegressor()**: Train the XGBoost regression model on the data.
- **r2_score()**: Evaluate the model's prediction accuracy.

---

