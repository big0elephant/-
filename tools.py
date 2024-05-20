import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DataCreate:
    def __init__(self) -> None:
        pass
    def createData(self,):
        from sklearn.datasets import make_classification
        # 创建复杂不平衡数据集
        X, y = make_classification(n_samples=10000,  # 样本数量
                                n_features=100,    # 特征总数量
                                n_informative=30,  # 信息性的特征数量
                                n_redundant=10,   # 冗余信息，有效特征的随机线性组合
                                n_repeated=5,     # 重复信息，随机提取n_informative和n_redundant
                                n_classes=3,      # 类别数量
                                n_clusters_per_class=2,  # 每个类别的簇数量
                                weights=[0.05, 0.15, 0.8],  # 类别权重，制造类别间不平衡
                                flip_y=0.01,  # 样本标签随机翻转的比例
                                class_sep=0.8,  # 类别分隔的程度
                                hypercube=True,
                                shift=0.1,  # 将整个特征空间随机移动
                                scale=1.0,  # 随机缩放特征
                                shuffle=True,  # 打乱样本和特征的顺序
                                random_state=42)  # 随机数生成器的种子

        # 转换为 DataFrame 以便使用
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        data['target'] = y
        return data

class DataDetect:
    def __init__(self) -> None:
        pass

    #  IV值大于0.02的变量通常意味着预测能力强
    def compute_woe1(self, x, y, na = -1):
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

    # 计算数据缺失率
    def missing_values_table(self, data, threshold=0):
        mis_val = data.isnull().sum()
        mis_val_percent = 100 * data.isnull().sum() / len(data)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : 'NanRate'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[
        (mis_val_table_ren_columns.iloc[:,1] != 0) & (mis_val_table_ren_columns.iloc[:,1] > threshold)].sort_values('NanRate', ascending=False).round(1)
        print ("含空值的列有 " + str(data.shape[1]) + " 列.\n"\
                "有 " + str(mis_val_table_ren_columns.shape[0]) +" 列的空值阈值超过了 > {}%.".format(threshold))
        
        #使用过滤后的列创建新的DataFrame
        high_missing_features = mis_val_table_ren_columns[mis_val_table_ren_columns['NanRate'] > threshold].index.tolist()
        print('空值大于阈值的特征有:',high_missing_features)
        return high_missing_features

    def feature_distribution(self, data_series, colName):
        '''
        本函数以直方图形式提供单个特征值的分布
        参数:
        - data_series: pandas Series,需要分析的特征数据。
        - bins: int 或 sequence of scalars,可选参数,默认为10。用于直方图的区间数量。
        '''
        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 5))
        category_counts = data_series.value_counts()
        # 绘制直方图
        ax.bar(category_counts.index, category_counts.values, color='blue', edgecolor='black')
        ax.set_title('{} Categories'.format(colName))
        ax.set_xlabel('Categories')
        ax.set_ylabel('frequency')

        # 调整布局防止重叠
        plt.tight_layout()
        # 显示图表
        plt.show()

    # 统计特征的结果数量
    def value_count(data):
        one_col = []  ## 一元特征
        binary_col = []  ## 二元
        multi_col = []  ## 多元
        num_col = []  ## 大于十种结果的特征
        for col in [col for col in data.columns if 'label1' not in col]:
            temp = pd.value_counts(data[col])
            if len(temp) == 1:
                print('----',col,'----')
                one_col.append(col)
            elif len(temp) == 2:
                binary_col.append(col)
            elif len(temp) < 10 :
                multi_col.append(col)
            else :
                num_col.append(col)
        return [one_col, binary_col, multi_col, num_col]

class FeatureExpansion:
    def __init__(self) -> None:
        pass

    def extend(self, dataTrainExtendNormal, uniqueId = 'id',if_testData = 0, trained_feature = 0, debugmode = 0):
        '''
        特征拓展核心函数
        1. 首次应用是对trainData进行训练,得到trained_feature(即拓展的特征列). 
        2. 对testdata则是必须使用得到trained_feature进行再次特征拓展,得到同样的拓展特征,以便模型的预测.
        '''
        import featuretools as ft
        import numpy as np

        # 调试模式默认不开启
        if debugmode == 1:
            aggregationList=['sum']
            transformList=['divide_numeric']  
        else:
            method = ft.list_primitives()
            aggregationList = method[method['type'] == 'aggregation']['name'].tolist()
            transformList = method[method['type'] == 'transform']['name'].tolist()
        # 创建EntitySet并添加dataframe
        es = ft.EntitySet(id = 'id')
        es = es.add_dataframe(dataframe_name='data', dataframe=dataTrainExtendNormal, index=uniqueId)

        # 使用DFS构建特征
        
        trans_primitives=transformList
        agg_primitives=aggregationList
        if if_testData == 0:
            feature_matrix, feature_names = ft.dfs(entityset=es, 
                                                target_dataframe_name = 'data', 
                                                max_depth = 5, 
                                                verbose = 1,
                                                agg_primitives=agg_primitives,
                                                trans_primitives=trans_primitives,
                                                n_jobs = -1)
            # 检查并处理异常值
            feature_matrix[~np.isfinite(feature_matrix)] = 0
            return  feature_matrix, feature_names
        else: 
            feature_matrix = ft.calculate_feature_matrix(entityset=es, 
                                                features=trained_feature)
            # 检查并处理异常值
            feature_matrix[~np.isfinite(feature_matrix)] = 0
            return  feature_matrix
    
class FillNa:
    def __init__(self,) -> None:
        pass
        
    # KNN和mice空值填充
    def fillNa_fancyimpute(self, data, mode = 'MICE', iter = 100, if_testData=0, imputerModel = None):
        '''
        1. 对MICE,一旦训练imputer实例,可以直接使用其transform方法对测试数据进行空值填充,因为fit步骤已经在训练数据上完成了
        2. 对于KNN,每次调用时都会重新计算最近邻,无需保留训练过程中的实例状态。
        '''
        from fancyimpute import KNN
        from fancyimpute import IterativeImputer

        if mode == 'MICE':
            if if_testData == 0:
                imputer = IterativeImputer(max_iter=iter)
                
                filled = imputer.fit_transform(data)
                
                dataNew = pd.DataFrame(filled, columns=data.columns)
                return dataNew, imputer
            else:
                imputer = imputerModel
                filled = imputer.transform(data)
                dataNew = pd.DataFrame(filled, columns=data.columns)
                return dataNew, imputer
        elif mode == 'KNN':
            imputer = KNN(max_iter=iter)
            filled = imputer.fit_transform(data)
            dataNew = pd.DataFrame(filled, columns=data.columns)
            return dataNew, imputer

    # 简单数学填充
    def fillNa_simpleImputer(self, data, mode='most_frequent'):
        """
        使用SimpleImputer填充DataFrame中的缺失值

        参数:
        - data: 输入DataFrame
        - mode: 填充策略,'mean', 'median', 'most_frequent', 'constant'
        - return_cols: 是否返回原始列名,默认False
        
        返回:
        填充后的DataFrame
        """
        import pandas as pd
        from sklearn.impute import SimpleImputer
        # 参数校验
        if mode not in ['mean', 'median', 'most_frequent', 'constant']:
            raise ValueError('mode 参数必须为 "mean", "median", "most_frequent", "constant"')
  
        if mode == 'constant':   
            imputer = SimpleImputer(strategy=mode, fill_value=0)
        else:
            imputer = SimpleImputer(strategy=mode)
        filled = imputer.fit_transform(data)  
        return pd.DataFrame(filled, columns=data.columns)
    
class SampleFunction:
    
    def __init__(self) -> None:
        pass

    # 超采样SMOTE (Synthetic Minority Over-sampling Technique)
    def oversample_SMOTE(self, X_train, y_train):
        from imblearn.over_sampling import ADASYN, SMOTE
        import pandas as pd
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    # 超采样ADASYN (Adaptive Synthetic Sampling)
    def oversample_ADASYN(self, X_train, y_train):
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN()
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    # imblearn超采样和欠采样
    def sample_Imblearn(self, X_train, y_train, mode):
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        if mode == 1:
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train) 
        elif mode == -1:
            rus = RandomUnderSampler(random_state=0)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

# X代表的是需要计算重要性的特征,即我们通常说的自变量、输入变量。它是一个二维数组,shape为(样本数, 特征数)。
# y代表的是学习任务的标签或目标,即所要预测的变量。它是一个一维数组,shape为(样本数,),表示每个样本的目标值。


class importantceCalculate:
    def __init__(self) -> None:
        pass
        from scipy.stats import stats

    # 计算皮尔逊及斯皮尔曼系数
    def featureImportance_corrs(self,data, resultColumn, threshold, mode='person'):
        print('支持的参数有: pearson,spearman')
        rhoListTemp = []
        for i in data.columns:
            rho = data[resultColumn].corr(data[i], method= mode)
            if abs(rho) > 0.01:
                rhoListTemp.append([i,abs(rho)])            
        if len(rhoListTemp) == 0:
            print('当前阈值过低,没有过滤出特征。')
        rhoMatrix = pd.DataFrame(rhoListTemp)
        return rhoMatrix
        

    # 计算基尼系数
    '''
    基尼系数范围为0至1,值越大表示该特征包含的样本越不均衡,区分结果的能力越强
    gini<0.2,特征对结果的区分能力较弱;[0.2, 0.4]有一定的区分能力;
    [0.4, 1]该特征包含非常不均衡的样本,因此可以很好地区分不同的结果
    '''
    def featureImportance_gini(self, data, resultColumn = 'result'):
        print('当前调用Gini Importance方法.')
        gini = []
        for col in data.columns:
            p = data.groupby(col)[resultColumn].count()/len(data)
            giniTemp = 1 - (p**2).sum()
            gini.append([col, giniTemp])
        gini = pd.DataFrame(gini, columns=['features','gini'])
        return gini

    # 计算相关系数
    def featureImportance_importantce(self, data, resultColumn = 'result', mode=1):
        from scipy.stats import stats
        print('当前调用相关系数 Importantce方法')
        print('当前mode为1,是丢弃空行模式') if mode == 1 else print('当前mode为0,是非丢弃空行模式')
        corrs = []
        for col in data.columns:
            if col != resultColumn:
                data_cleaned = data[[col, resultColumn]].dropna()
                if len(data_cleaned) > 2:
                    if mode == 1:
                        feature1 = data_cleaned[col].dropna()
                        result = data_cleaned[resultColumn].dropna()
                    elif mode == 0:
                        feature1 = data_cleaned[col]
                        result = data_cleaned[resultColumn]
                    r = stats.pearsonr(feature1, result)
                corrs.append([col, r[0], r[1]])
        corrs = pd.DataFrame(corrs, columns=['features','R_values', 'P_values'])
        return corrs


    # 计算互信息  
    def featureImportance_mis(self, data, resultcolumns, bins = 10):
        print('当前调用互信息计算方法')
        def mi(x, y, bins):
            print(bins)
            epsilon = 1e-8  # 增加平滑项
            hist_xy = np.histogram2d(x, y, bins)[0]  
            px = np.histogram(x, bins[0])[0] / len(x)
            px_s = np.where(px==0, epsilon, px)
            py = np.histogram(y, bins[1])[0] / len(y)
            py_s = np.where(py==0, epsilon, py)
            g_xy = hist_xy / len(x)
            g_xy_smoothed = np.where(g_xy == 0, epsilon, g_xy)

            mi = np.sum(g_xy_smoothed * np.log(g_xy_smoothed / ((px_s[:, None]) * (py_s[None, :] )))) 
            return mi

        mis = []
        for col in data.columns:
            if col != resultcolumns:
                r = mi(data[col], data[resultcolumns], bins = [len(pd.unique(data[col])), len(pd.unique(data[resultcolumns]))])
                mis.append((col, r))
        mis = pd.DataFrame(mis, columns=['features','mis'])
        return mis

    # WOE（Weight of Evidence）和IV（Information Value）:
    import numpy as np
    import pandas as pd
    def featureImportance_WOE_IV(self, data, resultColumn, na = -1, reserveDigits = 4):
        datax = data.drop([resultColumn], axis=1)
        y = data[resultColumn]
        infMatrix = []
        for col in datax.columns:
            x = datax[col]
            try:
                tb = pd.crosstab(x.fillna(na),y,dropna=False,margins=True)
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
            infMatrix.append([col, tb, round(iv, 4)], )
        resultMatrix = pd.DataFrame(infMatrix, columns=['features','tb', 'iv'])
        return resultMatrix

    #############################################################################
    ####################以下均通过实例化后的模型进行计算###########################
    #############################################################################

    # Permutation Importance:
    def featureImportance_permutation(self, model, X, y):
        # model 必须是已经实例化的模型
        print('当前调用Permutation Importance方法,传入的必须是已经fit后的model。')
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(model, X, y)
        feature_importances = perm_importance.importances_mean
        return feature_importances

    # SHAP Values:
    def featureImportance_shap(self, modelType, model, X):  
        import shap 
        print('当前调用shap方法,传入的必须是已经fit后的model.朴素贝叶斯和AdaBoost目前不支持')
        print('对于Decision Tree、Random Forest、XGBoost、LightGBM和CatBoost, 请选择shap.TreeExplainer\n对于Linear Model、Logistic Regression请选择shap.LinearExplainer')
        print('对于SVM、KNN请选择shap.KernelExplainer\n对于Neural Network,请选择shap.DeepExplainer')
        if modelType == 'shap.TreeExplainer':
            explainer = shap.TreeExplainer(model)
        elif modelType == 'shap.LinearExplainer':
            explainer = shap.LinearExplainer(model)
        elif modelType == 'shap.KernelExplainer':
            explainer = shap.KernelExplainer(model)
        elif modelType == 'shap.DeepExplainer':
            explainer = shap.DeepExplainer(model,X) # 调用成功
        shap_values = explainer.shap_values(X)
        return shap_values


    # 特征选择法:
    def featureImportance_selectN(self, model, X, y, N = 5):
        print('当前调用特征选择法,传入的必须是已经fit后的model.')
        from sklearn.feature_selection import RFE
        selector = RFE(model, n_features_to_select=N) 
        selector.fit(X, y)
        return selector

    # 决策规则学习
    # pip install skope-rules
    def featureImportance_skRules(self, X, y): 
        import six
        import sys
        sys.modules['sklearn.externals.six'] = six # sklearn版本较新问题
        from skrules import SkopeRules
        rf = SkopeRules(max_depth=3, precision_min=0.6).fit(X, y)
        print(rf.score_top_rules(X[:10])) # 或者是这个
        rules = rf.rules_[:10] # 输出的规则
        # feature_importances = rf.feature_importance() # 没有这个函数
        return rules

    # 卷积神经网络Attention机制
    def featureImportance_abcpyStatistics(self, data):
        import pandas as pd
        from tensorflow import keras
        import tensorflow as tf
        # data.columns = ['feature_1', 'feature_2', ..., 'result']
        X = data[['feature_1', 'feature_2', ...]]  
        y = data['result']   
        inputs = keras.Input(shape=X.shape[1])
        x = keras.layers.Dense(16, activation="relu")(inputs)
        attention = keras.layers.Attention()([x, x]) # attention层没有参数
        outputs = keras.layers.Dense(X.shape[1], activation="softmax")(attention)  # 新增一层全连接
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y,epochs=50,batch_size=32)

        importance = abs(model.layers[-1].weights[1])
        return importance


