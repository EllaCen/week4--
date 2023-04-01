# 總分為 170 分
# 及格分數為 90 分


#--------------- 簡答題 -------------------
# 問題（簡答）：資料集內遇到 missing value，請問該如何處理？ 請提供至少兩種情況的處理方式。
# 15分
"""
方法一：可使用一下套件將遺漏值以平均值代替
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

方法二：可將有遺漏值的資料刪除
如dataset = dataset.dropna(subset=['sales'] , how = 'any')

方法三：使用fillna
df['sales'].fillna(value='Not Found', inplace=True)
"""

# 問題（簡答）：資料集內有類別的資料為了進行預測，請問該如何處理？
# 15分
"""
應將其作one hot encoding
可使用
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
或是
str_columns1 = detect_str_columns(X_train)
X_train = get_dummies(str_columns1, X_train)
str_columns2 = detect_str_columns(X_test)
X_test = get_dummies(str_columns2, X_test)

若是有序的應用label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

來將其轉換成數字以利分析
"""

# 問題（簡答）：請問訓練模型時，出現 data leakage 是什麼？該如何避免？
# 15分
"""
數據洩漏，是指在train資料集中參雜了未知資料集的資訊
首先在切分資料集時，不可以隨意抽取的方式區分
再者使用標準化時，需先再train資料集裡建立標準化模型(fit.treansform)，
test資料集轉換必須使用train資料集的標準化模型，使用treansform即可
"""

# 問題（簡答加分題）：請問 GBDT模型跟Deep Learning的差異，以及在應用上各自的優缺點？
# 25分
"""
<請在此作答>
"""

# 至此總分70

# ---------------  機器學習 實作題  -------------------
# 問題：請看bank.pdf檔案，並依照過往所學之機器學習知識，對bank_train.csv與bank_test.csv執行分析
# 也可以參考pdf裡面的Tip1與Tip2
# 60分
import os
import util
from util import get_dummies, detect_str_columns,model_testRF,results_summary_to_dataframe,plot_confusion_matrix,logistic_model,logistic_importance,logistic_conf,model_profit_fun,model_profit_newdata_fun
from util import profit_linechart, profit_linechart_all
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, accuracy_score,classification_report
import numpy as np
from xgboost import XGBClassifier
import shutil

data1 = pd.read_csv('bank_train.csv')
data2 = pd.read_csv('bank_test.csv')

X_train = data1.drop(columns='buy')
y_train = data1['buy']
X_test = data2.drop(columns='buy')
y_test = data2['buy']

train_uid = X_train['UID']
test_uid = X_test['UID']

del X_train['UID']
del X_test['UID']

str_columns1 = detect_str_columns(X_train)
X_train = get_dummies(str_columns1, X_train)
str_columns2 = detect_str_columns(X_test)
X_test = get_dummies(str_columns2, X_test)


LR_all_df, LR_model_profit_df, LR_y_test_df = model_profit_fun(
        clf = LogisticRegression(),
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        sales_price = 3450, 
        marketing_expense = 312, 
        product_cost = 1250, 
        plot_name = 'LogisticRegression')

from util import move_file
move_file(dectect_name = 'LogisticRegression', folder_name = 'LogisticRegression_利潤結果')

XGB_all_df, XGB_model_profit_df, XGB_y_test_df = model_profit_fun(
                                                    
        clf =  XGBClassifier(n_estimators=300 ,random_state = 0,nthread = 8)   , 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        sales_price = 3450, 
        marketing_expense = 312, 
        product_cost = 1250, 
        plot_name = 'XGB' ) 

move_file(dectect_name = 'XGB', folder_name = 'XGB_利潤結果')

RF_all_df, RF_model_profit_df, RF_y_test_df = model_profit_fun(
    
        clf = RandomForestClassifier(n_estimators = 100, random_state = 0), 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        sales_price = 3450, 
        marketing_expense = 312, 
        product_cost = 1250, 
        plot_name = 'Random_Forest' ) 

move_file(dectect_name = 'Random_Forest', folder_name = 'RF_利潤結果')

profit_linechart_all(y_test_df= [XGB_y_test_df,
                                 RF_y_test_df,
                                 LR_y_test_df] ,
                    sales_price = 3450,
                    marketing_expense = 312,
                    product_cost = 1250,)

'使用隨機森林模型，因為其預測利潤最高，對購買機率大於等於13%的人行銷，預期可賺1378008'
'隨機森林的混淆矩陣顯示，針對預測會買的人進行行銷，利潤高達720952，相較於全市場賺1203768'

RFClassifier_test_df=pd.DataFrame(y_test.values ,columns =['客戶對A商品【實際】購買狀態'])
RFClassifier_test_df['客戶對A商品【預測】購買機率'] = RF_y_test_df['Random_Forest_pred']
test_uid = test_uid.reset_index()  
test_uid = test_uid.drop(columns = ['index'])
RFClassifier_test_df = pd.concat([test_uid, RFClassifier_test_df  ], axis = 1)
RFClassifier_test_df=RFClassifier_test_df.sort_values('客戶對A商品【預測】購買機率', ascending = False)

'執行人員可透過顧客推薦名單制定行銷策略'
RFClassifier_test_df_13_up = RFClassifier_test_df[RFClassifier_test_df['客戶對A商品【預測】購買機率']>=0.13]
RFClassifier_test_df_13_up.to_csv('顧客產品推薦名單.csv', encoding='utf-8-sig')

'並進一步找出顧客會購買的關鍵因素(畫出決策樹)，做行銷策略的調整、新服務的開發'


# 至此總分170

# --------------- Streamlit 實作題 -------------------
# 問題：請查看 training3.JPG 配合 diabetes_prediction_ml_app 資料夾
# 請將紅框處轉變成「箭頭」所指之處
# 請將藍框處轉變成「箭頭」所指之處
# 請將完成的檔案，deploy到streamlit cloud(https://share.streamlit.io/)，並在此附上網址
# 請注意streamlit cloud的Python版本限制在 <= 3.10
# PS: 請在deploy時看一下 Deploy! 按鈕上面的 advenced settings...，並記得將瀏覽權限設為 public
# 40分



# 至此總分210