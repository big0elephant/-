import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import KFold,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)

cd E:\桌面\bonc_2024\工信部投诉\train_20240519
data = pd.read_excel('data_train.xlsx')

##数据字典
data_dic = pd.read_excel('字典.xlsx')
data_dic = data_dic.iloc[1:]  ##去除开头行
data_dic.columns = ['col_name','col_meaning']
data_dic

data.head()

df = data.copy()

##查看样本比例
for col in ["if_gxbts"]: 
    t_rat = pd.DataFrame(data[col].value_counts())
    
    t_rat_per = (t_rat / len(data[col])) * 100
    t_rat_per = t_rat_per.rename(columns={col: f"Percent"})
    print('-----'*5)
    rat_table = pd.concat([t_rat, t_rat_per], axis=1)
    
    print(rat_table)

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) +
    " columns that have missing values.")
    return mis_val_table_ren_columns

miss_df=missing_values_table(df)
miss_df

print(miss_df[miss_df['% of Total Values'] >= 70].index)

df = df.drop(['DUTY_TYPE', 'RETURNMONEY', 'REAL_RETURNMONEY', 'RECALL_WAY',
       'IS_RETURNMONEY', 'TOTAL_SCORE', 'SERVICE', 'DIS_DETAIL', 'CLIENT_MAN',
       'DIS_TYPE', 'duanduan_time', 'IS_SEND_TELECOM', 'IS_CALL_TELECOM',
       'IS_REASON', 'GIS_STATION_CELL', 'APPEAL_DATE', 'discontented_reason',
       'offen_5g_rest_cell_top3', 'offen_5g_work_cell_top3',
       'offen_5g_cell_top3', 'offen_5g_rest_cell_top2',
       'offen_5g_work_cell_top2', 'offen_5g_rest_cell_top1',
       'offen_5g_cell_top2', 'offen_5g_work_cell_top1', 'offen_5g_cell_top1'],axis=1)

df.shape

##查找唯一值
del_cols = []
for col in df.columns:
    if df[col].nunique() == 1 :
        del_cols.append(col)
        print(col)
        print('----------------------')

df = df.drop(del_cols,axis=1)

#drop_col : 带有_id,nbr,date以及数据单一的无用字段
#obj_col : DataFrame中类别为object 类型的字段
#num_col : DataFrame中类别为 数值类型的字段
drop_cols = [col for col in df.columns if ('_id' in col ) | ('_nbr' in col) | ('_date' in col )|('_new' in col)]
obj_cols = df.select_dtypes(include =['object']).columns
num_cols = df.select_dtypes(exclude =['object','datetime64']).columns.tolist()
drop_cols,obj_cols,num_cols

print(df['time_interval'].value_counts().head())

## 剔除无关字段,包括日期、编码、id、文本等
del_cols = ["BUSI_PHONE","SHEET_NO","DEAL_TIME","COM_CONTENT","FIRST_RESPONSE","CREATE_DATE","WORK_SHEET","DEAL_RESULT",
"term_type","itv_type_name","offen_4g_cell_top1","offen_4g_cell_top2","offen_4g_cell_top3","IS_REPEAT",
"offen_4g_work_cell_top1","offen_4g_work_cell_top2","offen_4g_work_cell_top3","offen_4g_rest_cell_top1",
"offen_4g_rest_cell_top2","offen_4g_rest_cell_top3",'RISK_INFO','is_5g_cover','g5_mask_type','xxzx_label435','adsl_zc_reason']
df = df.drop(del_cols,axis=1)
df.shape

##-----------------------------------------------数据清洗-----------------------------------------------##
##-----异常值处理-----##
num_col = [col for col in num_cols if col not in del_cols]

def deloutliers(num_col, df, l=0.01, u=0.99):
    for col in num_col:
        if len(df[col]) > 20:
            a = df[col].quantile(l)
            b = df[col].quantile(u)
            df[col] = df[col].map(lambda x: b if x > b else a if x < a else x)
    return df

deloutliers(num_col,df)

##-----数据处理-----##
class_col = df.select_dtypes(include =['object']).columns

for col in class_col:
    print(pd.value_counts(df[col]).head(10))

for col in ['IF_JG', 'YY_TS_UP', 'TS_TYPE', 'TS_QW', 'LABEL']:
    print(pd.value_counts(df[col]))

df.replace('未知',0,inplace = True)
df.replace('未获取',-1,inplace = True)
df.replace('是',1,inplace = True)
df.replace('否',0,inplace = True)
df.replace('5个以上',5,inplace=True)
df['time_interval'] = df['time_interval'].map({u'早上':1,u'上午':2,u'中午':3,u'下午':4,u'晚上':5,u'夜间':6})
df['xxzx_label434'] = df['xxzx_label434'].map({u'一星':1,u'二星':2,u'三星':3,u'四星':4,u'五星':5,u'六星':6,u'七星':7}).fillna(-1)
df['prd_sex'] = df['prd_sex'].map({u'男':0,u'女':1})
df['prd_age'] = df['prd_age'].map({u'青少年':1,u'中年':2,u'老年':3})
df['flux_use_type'] = df['flux_use_type'].map({u'溢出型':1,u'抑制型':2,u'0':0})

df['IF_JG'] = df['IF_JG'].fillna(-1)
df['YY_TS_UP'] = df['YY_TS_UP'].map({u'高':2,u'较高':1,u'一般':0})

df['TS_TYPE'] = df['TS_TYPE'].map(lambda x: u'其他' if x not in [u'服务不满意',u'费用争议',u'服务不满意，费用争议',u'涉嫌欺诈'] else x) 
df['TS_TYPE'] = df['TS_TYPE'].map({u'服务不满意':1,u'费用争议':2,u'服务不满意，费用争议':3,u'涉嫌欺诈':4,u'其他':5})

df['TS_QW'] = df['TS_QW'].map(lambda x: u'其他' if x not in [u'期待问题解决',u'期待问题解决、希望得到补偿',u'希望得到补偿',u'发泄不满情绪'] else x) 
df['TS_QW'] = df['TS_QW'].map({u'期待问题解决':1,u'期待问题解决、希望得到补偿':2,u'希望得到补偿':3,u'发泄不满情绪':4,u'其他':5})
df = df[df['rel_chrg_level'] > 0]


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) +
    " columns that have missing values.")
    return mis_val_table_ren_columns

miss_df=missing_values_table(df)
miss_df

df= df.drop(['flux_use_type','LABEL'],axis=1)

df = df.drop('RECALL_RESULT',axis=1)
for col in ['CLIENT_LEVEL','prd_age']:  ##回访结果/客户等级/用户年龄
    df[col] = df[col].fillna(df[col].mode()[0])  
for col in ['prd_sex','REASON_TYPE','COM_SOURCE','RISK_INDEX']:
    df[col] = df[col].fillna(-1)

df.isnull().sum()

for col in df.columns:
    df = df.fillna(0)
    df[col] = df[col].astype(np.float32)
df.info()

##-----------------------------------------建模-----------------------------------------##
for col in df.columns:
    print(col)

feature_col = ["COM_TYPE","COM_SOURCE","TOUSU_COUNT","SEND_TIMES","RECALL_TIMES","FLOW_FACT_TIME","HURRY_CNT","CLIENT_LEVEL",
               "REASON_TYPE","RISK_INDEX","last_m_flux","last_m_5gflux","vedio_perfer","short_video_level","fav_zhibo_level","game_perfer",
               "music_perfer","reader_perfer","waimai_fans","licai_fans","term_times","retail_price","rt_sa_stitch","last1_m_5gkgs","is_main_net",
               "avg_3m_tnet_dur","speed_id","xxzx_label49","xxzx_label162","bb_zc_flag","aqiyi_member","is_txhy","youku_member","shejiao_fans",
               "jtyw_nums","is_act3_kd","kd_cons","last_m_itv_dur","call_out_1m_num","speed1000_flag","is_home_stu","call_comm_nums",
               "avg_prd_calling_num","avg_prd_called_num","last3_limit_spee","last_flow_chrg","yd_last_flux_all","last_pay_flux_pac_chrg",
               "dt_tq_fans","xxzx_label161","zyb_use_flux","prd_age","prd_sex","net_fee","xxzx_label274","xxzx_label275","xxzx_label278",
               "call_out_3m_num","c29","xxzx_label434","rel_chrg_level","c60","xxzx_label333","pnt_cust_eff_point","owe_flag","acct_own_chrg",
               "own_mon","left_balance","two_kd_flag","tf_amount","sf_amount","lj_charge","xxzx_label377","fwzj_flag","anchor_flag",
               "jz_design_flag","gwy_flag","kdy_flag","ywry_flag","wmy_flag","teach_flag","wysj_flag","tszyh_flag","wlxs_flag","skzj_flag",
               "ts_1y_cnt","gxb_wh_ts","time_interval", "IF_JG","YY_TS_UP","TS_TYPE","TS_QW",'if_gxbts']  ##"days_after","dif_nbr",


## 随机森林算法
feature_col.remove('if_gxbts')
x = df[feature_col].fillna(-1)

y = df['if_gxbts']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 0)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rfc = RandomForestClassifier(random_state=12,n_estimators = 200,
                             class_weight={0:1,1:6.7},
                             min_samples_split=30, 
                             min_samples_leaf=10,
                             max_depth=5,n_jobs = -1)
rfc.fit(xtrain,ytrain)

import sklearn.metrics as metrics
pre1 = rfc.predict(xtrain)  
print(metrics.confusion_matrix(ytrain, pre1, labels=[0, 1]))    
print(metrics.classification_report(ytrain, pre1,digits = 4))

pre_test = rfc.predict(xtest)
print(metrics.confusion_matrix(ytest, pre_test, labels=[0, 1]))   

print(metrics.classification_report(ytest, pre_test,digits = 4))


##特征重要性
res = pd.concat([pd.DataFrame(xtest.columns, columns=['feature_names']), pd.DataFrame(rfc.feature_importances_, columns=['importances'])], axis=1)
feature = res.sort_values('importances', ascending=False).round(4).reset_index(drop = True)
feature


##特征分析-附大模型+top5
def compute_woe1( x, y, na = -1):
    try:
        tb = pd.crosstab(x.fillna(-1),y,dropna=False,margins=True) 
    except:
        tb = pd.crosstab(x,y,dropna=False,margins=True)
    pos = pd.value_counts(y)[1]
    neg = pd.value_counts(y)[0]
    tb['rat'] = tb[1]/tb['All']
    bad_rat = tb[0]/neg
    good_rat = tb[1]/pos
    tb['woe'] = np.log(good_rat/bad_rat) * 100
    tb['woe'] = tb['woe'].replace(-np.inf,0)
    tb['woe'] = tb['woe'].replace(np.inf,0)
    bad_dev = good_rat - bad_rat
    iv = sum(bad_dev * tb['woe']) /100
    return tb,iv

for col in ['IF_JG', 'YY_TS_UP', 'TS_TYPE', 'TS_QW']:
    print(compute_woe1(df[col],df['if_gxbts']))

a= df['TOUSU_COUNT'].map(lambda x: u'5及以上' if x not in [1,2,3,4] else x) 
print(compute_woe1(a,df['if_gxbts']))

print(compute_woe1(df['gxb_wh_ts'],df['if_gxbts']))


a = pd.cut(df['avg_prd_calling_num'],[-np.inf,0,5,10,15,20,np.inf])
print(compute_woe1(a,df.if_gxbts))




