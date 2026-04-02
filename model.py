import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Models

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)

# metrics

from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score,f1_score)

from analysis import generate_summary, suggest_improvements

# for AI insights

from analysis import generate_summary, suggest_improvements

st.set_page_config('ML & AL insight App')

st.title('📊 Auto Ml + AI insights APP')
st.subheader(':green[To learn the given data and to fit the ML models and to get AI insights using Gemini]')

file = st.file_uploader('Upload the csv file here 📥', type=['csv'])

if file:
    df = pd.read_csv(file)
    st.write('### 📌 Data Preview')
    st.dataframe(df.head())
    
    target = st.selectbox(':blue[Select target here]', df.columns)
    
    if target:
        
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()
        
        # preprocessing 
        
        num_cols = X.select_dtypes(include = ['int64','float64']).columns.tolist()
        cat_cols = X.select_dtypes(include = ['object']).columns.tolist()
        
        
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna('Missing')
        
        # Encoding
        
        X = pd.get_dummies(X, columns= cat_cols, drop_first= True, dtype=int)
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Detect the problem type
            
        if df[target].dtype == 'object' or len(np.unique(y)) < 15:
            problem_type = 'Classification'
        else:
            problem_type = 'Regression'

        st.write(f'### 🔍 Problem type : {problem_type}')
        
        
        # Split the data
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        for i in xtrain.columns:
            xtrain[i] = scaler.fit_transform(xtrain[[i]])
            xtest[i] = scaler.transform(xtest[[i]])
            
        # Models
        results=[]
        if problem_type == 'Regression':
            models={'Linear Regression' : LinearRegression(),
                    'Random Forest':RandomForestRegressor(),
                    'Gradient Boosting': GradientBoostingRegressor()}
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model Name':name,
                                'R2 Score': round(r2_score(ytest,ypred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(ytest,ypred)),3)})
                
                
        else:
            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            for name, model in models.items():
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                results.append({
                    'Model Name': name,
                    'Accuracy': round(accuracy_score(ytest, ypred), 3),
                    'Precision': round(precision_score(ytest, ypred, average='weighted'), 3),
                    'Recall': round(recall_score(ytest, ypred, average='weighted'), 3),
                    'F1 score': round(f1_score(ytest, ypred, average='weighted'), 3),
                })
                
        results_df = pd.DataFrame(results)
        st.write("### :red[📊 Model Results]")
        st.dataframe(results_df)

        if problem_type == 'Regression':
            st.bar_chart(results_df.set_index('Model Name')['R2 Score'])
            st.bar_chart(results_df.set_index('Model Name')['RMSE'])

        else:
            st.bar_chart(results_df.set_index('Model Name')['Accuracy'])
            st.bar_chart(results_df.set_index('Model Name')['F1 score'])

        # ==========
        # AI insights
        # ==========
        

        if st.button(":blue[📜 Generate Summary]"):
            summary = generate_summary(results_df)
            st.write(summary)

        if st.button(":blue[📈 Suggest Improvements]"):
            improve = suggest_improvements(results_df)
            st.write(improve)

        # DOWNLOAD

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV here", csv, "model_results.csv")