
'''
///工信部投诉倾向建模脚本·初///
'''
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import KFold,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
pd.set_option('display.max_rows', 1000)
import warnings
warnings.filterwarnings('ignore')

cd E:\桌面\bonc_2024\工信部投诉
data = pd.read_csv('最新训练数据.csv',encoding='gbk')
data = data.drop_duplicates()

##数据字典
data_dic = pd.read_excel('工信部投诉分析模型数据表字典.xlsx')
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

a = print(miss_df[miss_df['% of Total Values'] >= 80].index)

df = df.drop(a,axis=1)

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

a = miss_df[miss_df['% of Total Values'] >= 80].index
df = df.drop(a,axis=1)

##查找唯一值
del_cols = []
for col in df.columns:
    if df[col].nunique() == 1 :
        del_cols.append(col)
        print(col)
        print('----------------------')
df = df.drop(del_cols,axis=1)

drop_cols = [col for col in df.columns if ('_id' in col ) | ('_nbr' in col) | ('_date' in col )|('_new' in col)]  #drop_col : 带有_id,nbr,date以及数据单一的无用字段
obj_cols = df.select_dtypes(include =['object']).columns  #obj_col : DataFrame中类别为object 类型的字段
num_cols = df.select_dtypes(exclude =['object','datetime64']).columns.tolist()  #num_col : DataFrame中类别为 数值类型的字段
print('drop_cols:','\n',drop_cols,'\n','-'*50)
print('obj_cols:','\n',obj_cols,'\n','-'*50)
print('num_cols:','\n',num_cols)


## 剔除无关字段,包括日期、编码、id、文本等
del_cols = ["BUSI_PHONE","SHEET_NO","DEAL_TIME","COM_CONTENT","FIRST_RESPONSE","CREATE_DATE","WORK_SHEET","DEAL_RESULT",
"term_type","itv_type_name","offen_4g_cell_top1","offen_4g_cell_top2","offen_4g_cell_top3","IS_REPEAT",'adsl_zc_reason',
"offen_4g_work_cell_top1","offen_4g_work_cell_top2","offen_4g_work_cell_top3","offen_4g_rest_cell_top1",'RISK_INFO',
"offen_4g_rest_cell_top2","offen_4g_rest_cell_top3","offen_5g_cell_top1","offen_5g_cell_top2","offen_5g_cell_top3",
"offen_5g_work_cell_top1","offen_5g_work_cell_top2","offen_5g_work_cell_top3","offen_5g_rest_cell_top1",'xxzx_label434',
"offen_5g_rest_cell_top2",'is_5g_cover','g5_mask_type','xxzx_label435']
df = df.drop(del_cols,axis=1)
df.shape

##-----------------------------------------------数据清洗-----------------------------------------------##
##-----异常值处理-----##
def deloutliers(data, l = 0.01, u = 0.99):
    a=data.quantile(l)
    b=data.quantile(u)
    data=data.map(lambda x:b if x>b else a if x < a else x)
    return data
    
num_col = [col for col in num_cols if col not in del_cols]

col_meanings = {}
for col in num_col:
    meaning = data_dic[data_dic['col_name'] == col]['col_meaning'].values
    if len(meaning) > 0:
        col_meanings[col] = meaning[0]
# 输出特征对应的含义
for col, meaning in col_meanings.items():
    print(f"'{col}'：{meaning}")
    print('最小值：',min(df[col].fillna(-1)))
    print('最大值：',max(df[col].fillna(-1)))
    print('-'*50)
    
##异常数据
df = df[df['rel_chrg_level'] > 0]
for col in ['FLOW_FACT_TIME','last_m_flux','last_m_5gflux','avg_3m_tnet_dur','last_m_itv_dur','avg_prd_calling_num','avg_prd_called_num',
            'last_flow_chrg','yd_last_flux_all','last_pay_flux_pac_chrg','net_fee','c29','rel_chrg_level','pnt_cust_eff_point','acct_own_chrg',
            'left_balance','ts_cs','ts_1y_cnt','rel_chrg_level','pnt_cust_eff_point','acct_own_chrg','left_balance','ts_cs','ts_1y_cnt',
            'rel_chrg_level','pnt_cust_eff_point','acct_own_chrg','left_balance','ts_cs','ts_1y_cnt']:
    df[col] = deloutliers(df[col])

df.describe()

##-----数据处理-----##
class_col = df.select_dtypes(include =['object']).columns

for col in class_col:
    print(pd.value_counts(df[col]).head(10))

for col in ['IF_JG', 'YY_TS_UP', 'TS_TYPE', 'TS_QW', 'LABEL']:
    print(pd.value_counts(df[col].head()))

df.replace('未知',0,inplace = True)
df.replace('未获取',-1,inplace = True)
df.replace('是',1,inplace = True)
df.replace('否',0,inplace = True)
df.replace('5个以上',5,inplace=True)
# df.replace(' ',-1,inplace = True)
# df.replace(' NULL ',-1,inplace = True)
df['time_interval'] = df['time_interval'].map({u'早上':1,u'上午':2,u'中午':3,u'下午':4,u'晚上':5,u'夜间':6})
# df['xxzx_label434'] = df['xxzx_label434'].map({u'一星':1,u'二星':2,u'三星':3,u'四星':4,u'五星':5,u'六星':6,u'七星':7})
df['prd_sex'] = df['prd_sex'].map({u'男':0,u'女':1})
df['prd_age'] = df['prd_age'].map({u'青少年':1,u'中年':2,u'老年':3})
df['flux_use_type'] = df['flux_use_type'].map({u'溢出型':1,u'抑制型':2,u'0':0})

df['IF_JG'] = df['IF_JG'].fillna(-1)
df['YY_TS_UP'] = df['YY_TS_UP'].map({u'高':2,u'较高':1,'一般':0})

df['TS_TYPE'] = df['TS_TYPE'].map(lambda x: u'其他' if x not in ['服务不满意','费用争议','服务不满意，费用争议','涉嫌欺诈'] else x) 
df['TS_TYPE'] = df['TS_TYPE'].map({u'服务不满意':1,u'费用争议':2,u'服务不满意，费用争议':3,u'涉嫌欺诈':4,u'其他':5})

df['TS_QW'] = df['TS_QW'].map(lambda x: u'其他' if x not in ['期待问题解决','期待问题解决、希望得到补偿','希望得到补偿','发泄不满情绪'] else x) 
df['TS_QW'] = df['TS_QW'].map({u'期待问题解决':1,u'期待问题解决、希望得到补偿':2,u'希望得到补偿':3,u'发泄不满情绪':4,u'其他':5})

df['c622'] = df['c622'].fillna(0)

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

df= df.drop(['offen_5g_rest_cell_top3','LABEL'],axis=1)

df = df.drop('RECALL_RESULT',axis=1)
for col in ['CLIENT_LEVEL','prd_age','time_interval']:  ##回访结果/客户等级/用户年龄
    df[col] = df[col].fillna(df[col].mode()[0])  
df['flux_use_type'] = df['flux_use_type'].fillna(0)  #流量使用抑制/溢出
for col in ['prd_sex','REASON_TYPE','COM_SOURCE']:
    df[col] = df[col].fillna(-1)

df.isnull().sum()

df.replace(' ',-1,inplace = True)
df.replace(' NULL ',-1,inplace = True)
df.replace(' 未知 ',0,inplace = True)
df.replace(' 未获取 ',-1,inplace = True)
df.replace(' 是 ',1,inplace = True)
df.replace(' 否 ',0,inplace = True)

for col in df.columns:
    df = df.fillna(0)
    df[col] = df[col].astype(np.float32)
df.info()

##-----------------------------------------建模-----------------------------------------##
feature_col = ['COM_TYPE', 'COM_SOURCE', 'TOUSU_COUNT', 'SEND_TIMES', 'RECALL_TIMES',
       'FLOW_FACT_TIME',  'HURRY_CNT', 'CLIENT_LEVEL',
       'REASON_TYPE', 'RISK_INDEX', 'last_m_flux', 'last_m_5gflux',
       'vedio_perfer', 'short_video_level', 'fav_zhibo_level', 'game_perfer',
       'music_perfer', 'reader_perfer', 'waimai_fans', 'licai_fans',
       'term_times', 'retail_price', 'rt_sa_stitch', 'last1_m_5gkgs',
       'is_main_net', 'avg_3m_tnet_dur', 'speed_id', 'xxzx_label49',
       'xxzx_label162', 'bb_zc_flag', 'aqiyi_member', 'is_txhy',
       'youku_member', 'shejiao_fans', 'jtyw_nums', 'is_single', 'is_act3_kd',
       'kd_cons', 'last_m_itv_dur', 'call_out_1m_num', 'flux_use_type',
       'speed1000_flag', 'is_home_stu', 'call_comm_nums',
       'avg_prd_calling_num', 'avg_prd_called_num', 'last3_limit_spee',
       'last_flow_chrg', 'yd_last_flux_all', 'last_pay_flux_pac_chrg',
       'dt_tq_fans', 'xxzx_label161', 'zyb_use_flux', 
       'net_fee', 'xxzx_label274', 'xxzx_label275', 'xxzx_label278',
       'call_out_3m_num', 'c29', 'c622', 'rel_chrg_level', 'c60',
       'xxzx_label333', 'pnt_cust_eff_point', 'owe_flag', 'acct_own_chrg',
       'own_mon', 'left_balance', 'two_kd_flag', 'tf_amount', 'sf_amount',
       'lj_charge', 'xxzx_label377', 'fwzj_flag', 'anchor_flag',
       'jz_design_flag', 'zwry_flag', 'gwy_flag', 'kdy_flag', 'ywry_flag',
       'wmy_flag', 'teach_flag', 'wysj_flag', 'wlxs_flag', 'tszyh_flag','time_interval',
       'skzj_flag', 'is_gxb', 'ts_cs', 'ts_1y_cnt', 'gxb_wh_ts', 'wanhao_ts', 'IF_JG', 'YY_TS_UP', 'TS_TYPE', 'TS_QW'] 

## 随机森林算法
feature_col.remove('is_gxb')
feature_col.remove('wanhao_ts')
feature_col.remove('ts_cs')
x = df[feature_col].fillna(-1)

y = df['if_gxbts']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 0)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rfc = RandomForestClassifier(random_state=12,n_estimators = 200,
                             class_weight={0:1,1:10},
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




