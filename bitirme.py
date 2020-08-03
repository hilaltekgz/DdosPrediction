import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

#csv dosyasını okuma
data = pd.read_csv(r"C:\Users\hlltk\PycharmProjects\ddospredict\verisetii.csv")
del data['a'] #id silindi
#print(data)


print(data.head()) #ilk 5 elemanı verir ‘istege bağlı olarak head() fonk. içine istediğiniz kadarınıda alabilir.

print ("Rows     : " ,data.shape[0]) # Rows     :  311012 kaç satır olduğunu verir.
print ("Columns  : " ,data.shape[1]) # Columns  :  42kaç sutun olduğunu gösterir.
print ("\nFeatures : \n" ,data.columns.tolist()) # sutun isimlerini göstermek için kullanılır.
print ("\nMissing values :  ", data.isnull().sum().values.sum()) # kayıp değer olup olmadığını kontrol edder. Missing values :   0
print ("\nUnique values :  \n",data.nunique()) # Türlerden (özelliklerden) ‘benzersiz’ olanları göstermek

del data['num_outbound_cmds'] # benzersiz değerlerin sayısı 1 olduğu için silindi.

target_col = data["result"] # etiketli veri.

# Etiketli olan veride 0 ve 1 değerlerinin sayısını öğrenmek için aşağıda ki kod satırları oluşturuldu.
sayi = 0 # değişken belirleme
sayi1= 0
for i in target_col:
    if i==0:
        sayi = sayi+1
    else:
        sayi1 = sayi1+1

print('Result 1 ve 0 count:',sayi,sayi1) # Result 1 ve 0 count: 64556 64556


cat_cols = data.nunique()[data.nunique() < 6].keys().tolist() # kategorik kolonları bulmak için kullanıldı. (Kategorik veri nedir label encoder encoder açıklanabilir.)


cat_cols = [x for x in cat_cols if x not in target_col] # kategorik olan verileri etiketli veri hariç cat_cols değişkenine atama işlemi yaptık.
cat = cat_cols
kat = pd.get_dummies(data=data['protocol_type']) # protocol_type değişkeni kategorik veri olduğu için get_dummies metodu ile onehotencoder işlemi yaptık.
kat_ser = pd.get_dummies(data=data['service']) # aynı şekilde onehotencoder işlemi yapıldı.
kat_flag = pd.get_dummies(data=data['flag']) # aynı şekilde onehotencoder işlemi yapıldı.
print("encoder")
#Kategortik veriler üzerinde onehotencoder işlemi yapıldıktan sonra aşağıda tekrar yapılan kategortik veriler incelendi.#
print(kat)
print(kat.nunique())
print(kat_ser)
print(kat_ser.nunique())
print(kat_flag)
print(kat_flag.nunique())
"""
'protocol_type' kategorik parametre için onehot encoder işleminden sonra bu şekilde oldu.
        icmp  tcp  udp
0          0    0    1
1          0    0    1
2          0    0    1
3          0    0    1
4          0    0    1
...      ...  ...  ...
311007     0    0    1
311008     0    0    1
311009     0    0    1
311010     0    0    1
311011     0    0    1

[311012 rows x 3 columns]

"""
byt = data['dst_bytes'] # dst_bytes parametresini byt değişkenine atama işlemi
byt = pd.DataFrame(byt) #değişkeni dataframe' e çavirme işlemi (Dataframe nedir araştırıp yazabilirsin.)
byt.to_numpy() # oluşturulan dataframe numpy' a çevrildi çevrilmesinin sebebi standart scaler yan, normalizasyon yapabilmek için numpy türünü çevirmem işlemi yapmamız gerekir.
src = data['src_bytes'] # Bu değişkenler içinde yukarıdaki işlemler aynı şekilde yapılmıştır.
src = pd.DataFrame(src)
src.to_numpy()

dst = data['dst_host_count']# Bu değişkenler içinde yukarıdaki işlemler aynı şekilde yapılmıştır.
dst = pd.DataFrame(dst)
dst.to_numpy()

dst_count = data['dst_host_srv_count']
dst_count = pd.DataFrame(dst_count)
dst_count.to_numpy()
from sklearn import preprocessing # minmazScaler sklern kütüphanesi preprocessing kütüphanesinin içerisinde bulunduğpu için küyüphane içe aktarma işlemi yapıldı.
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) #preprocessing içerisindeki minmaxscaler için x_scaler adında bir özellik oluştuduk.
x_scaled = min_max_scaler.fit_transform(byt) #ve yukarıda numpy çevirdiğimiz parametrelere minmaxscaler uyguladık.
y_scaled = min_max_scaler.fit_transform(src)#ve yukarıda numpy çevirdiğimiz parametrelere minmaxscaler uyguladık.
dst_scaled = min_max_scaler.fit_transform(dst)#ve yukarıda numpy çevirdiğimiz parametrelere minmaxscaler uyguladık.
dst_count_scaled = min_max_scaler.fit_transform(dst_count)#ve yukarıda numpy çevirdiğimiz parametrelere minmaxscaler uyguladık.
#scaled = pd.DataFrame(x_scaled,columns=['tenure','MonthlyCharges','TotalCharges'])
x_scaled = pd.DataFrame(x_scaled) # Bunları tekrar hepsini birlşetirmek için dataframe'e çevirme işlemi yapıyoruz.
y_scaled = pd.DataFrame(y_scaled)
dst_scaled = pd.DataFrame(dst_scaled)
dst_count_scaled = pd.DataFrame(dst_count_scaled)
# print("maxminscaled")
# print(dst_scaled)
# print(x_scaled)
# print(y_scaled)
a =pd.concat([kat,kat_flag,kat_ser,x_scaled,y_scaled,dst_scaled,dst_count_scaled],axis =1) # hepsini tek bir dataframede birleiştiriyoruz.

"""
Aşağıdaki del fo2nsiyonu ile ise yukarıdaki kod satırlarında yaptığımız düzenlemeler sonucunda eski parameteleri silip işlem görmüş yani kategorik verileri normalizasyon uyguladığımız 
yeni a datadrame'i ekleyeceğiz.
"""
del data['dst_bytes']
del data['protocol_type']
del data['src_bytes']
del data['dst_host_count']
del data['dst_host_srv_count']
del data['service']
del data['flag']
print(a.head())
data = pd.concat([data,a],axis=1) # birleştirme işlemi


data = pd.DataFrame(data) #dataframe' çevirme işlemi
data.dropna()


"""
Selectkbest ile Özellik seçimi, verilerimizde hedef değişkene en fazla katkıda bulunan özellikleri seçtiğimiz bir tekniktir. Başka bir deyişle, hedef değişken için en iyi tahminleri seçiyoruz.
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,  k = 'all')
fit = bestfeatures.fit(data,target_col)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['parameters','result']  #datafranesutublarının isimlerini oluşturuyoruz.
print("Feature Selection")
print(featureScores.nlargest(100,'result'))  #100 özellik için değişkemlerin hedef değişken arasındaki ilişkilerini bu şekilde kontrol edebiliyotuz.
del data['other']

# Veriseti 64556 246456 bu şekilde 1 - 0 sayısı ddngesiz olduğu için smote işlemi ile oranlama yapıyoruz. (Adasyn ve smoote açıklanabilir.Dökümanda internetten ne işe yaradoğını güzelce anlatabilirsiniz.)
from imblearn.over_sampling import SMOTE
train_pct_index = int((0.66 * len(data))) # eğitim ve test setini %66 ile %33 olarak ayarllama işlemi
X_train, X_test = data [: train_pct_index], data [train_pct_index:]
y_train, y_test = target_col [: train_pct_index], target_col [train_pct_index:]
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train) #daha sonra smote işlemi ile 0 ve 1 lerin sayısını dengeliyorz. Dengelemz isek verisetimiz yoğun olana tarafa doğru eğitilebilir.
#data.to_csv(r'C:\Users\hlltk\PycharmProjects\ddospredict\dataa.csv') # oluşturulan yani veri ön işleme adımları tamamlanan veriyi csv formatına çeviriyoruz.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression # logistik regresyon işlemi için kütüphane yükleme işlemi
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} # GridSearchCv ise bize hiper parametre seçme kolaylığı sunuyor. Bu sayede logistik regresyon parametrelerini etkili bir şekilde kullanbiliyotuz.
threshold = 0.5 # threshold değerini değiştirebilirz. Bunu kullanmamızın amacı ml algortimaları varsayılan olarak 0.5 > olanlara 1 küçük olanalara ise 0 olarak değer atıyor.
lr=LogisticRegression(random_state=42) # logisticRegression nesnesini oluşturuyoruz.
print("---")
print(lr.best_params_)#en iyi parametreleri öğreniyoruz.

lr_best=LogisticRegression(C=1.0,penalty='l1') # öğrendikten sonra paraametreleri ona göre belirliyorum
lr_best.fit(X_train_res, y_train_res) # eğitim setlerini logisticregression algoritması ile fit ediyoruz.
y_pred_lr = (lr_best.predict_proba(X_test)[:,1] >= 0.6).astype(bool) # threshol değerini ayarlayarak tahmin işlemi yapıyoruz.
from sklearn.metrics import confusion_matrix,accuracy_score # modeli değerlendirmek için kütüphaneleri yükleme işlemi ypıyoruz.
accuracy = accuracy_score(y_test, y_pred_lr) # acc. değerini hesaplamak için kullanıyoruz.
print(round(accuracy,4,)*100, "%") #Burada da accuracy değerini ekrana yazdırıyoruz.
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr) # karmaşıklık matrisi için
print(confusion_matrix_lr) # karmaşıklık matrisini tazıdırıyoruz.

from sklearn.metrics import classification_report # ayrıntılı şekilde çıktıalamk için kullanıyoruz.
print(classification_report(y_test, y_pred_lr))


####################################################################
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,4,5,6,7,8],
    'criterion' :['gini', 'entropy']
} # param_grid değişkeni ile RFC algortimasının tüm paramterelerini listeye ekliyoruz.
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42)  # RFC nesnesi oluşturuyoruz.

rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5) # cv çapraz doğrulama bölme . Ve hiper paramtreleri belirleme işlemi yapıyoruz.
print(rfc.best_params_)

rfc_best=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 50, max_depth=8, criterion='gini') # belirlenen hiperparametrelere göre algoritmanın parametrelerini dolduruyoruz.
rfc_best.fit(X_train, y_train)
y_pred_rfc = (rfc_best.predict_proba(X_test)[:,1] >= 0.5).astype(bool)
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test, y_pred_rfc)
print(round(accuracy,4,)*100, "%")
confusion_matrix_forest = confusion_matrix(y_test, y_pred_rfc)
print(confusion_matrix_forest)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))

#################################################################

thres = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # istersek 0'dan 1 e kadar threshold değerlerini değiştirip hangisinde daha iy, sonuç verdiğini karşılatırıp threshold değerini ayarlayabilriiz.Ondan dolayı bir diziye 0dan 1 e kadar değerleri yazıp bunları tek tek değiştirerek sonu.ları alacağız.
from sklearn.neighbors import KNeighborsClassifier as KNN
for i in thres:
    knn = KNN()
    knnGrid = {
        'n_neighbors': range(1, 12),
    }
    knn = GridSearchCV(estimator=knn, param_grid=knnGrid, cv=5)

    print(knn.best_params_)

    knn_best = KNN(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
    print(knn_best.fit(X_train_res, y_train_res))

    y_pred_knn = (knn_best.predict_proba(X_test)[:, 1] >= i).astype(bool)
    print(knn_best.predict_proba(X_test)[:, 1] == i)
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_knn)
    print(round(accuracy, 4, ) * 100, "%")

    confusion_matrix_forest = confusion_matrix(y_test, y_pred_knn)
    print(confusion_matrix_forest)
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred_knn))

#############################################################################


from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

for i in thres:
    parameters = {'n_estimators':[100,200,300],'learning_rate':[1.0,2.0,4.0]}
    abc = AdaBoostClassifier()
    abc_grid = GridSearchCV(abc, parameters)
    # Train Adaboost Classifer
    model = abc_grid.fit(X_train_res, y_train_res)
    y_pred = (model.predict_proba(X_test)[:, 1] >= i).astype(bool)
    confusion_matrix_forest = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_forest)
    print("Adaboost Accuracy:",metrics.accuracy_score(y_test, y_pred))
    report=classification_report(y_test,abc_grid.predict(X_test))

    print("Report",report)
    deger = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("precisin - recall - fscore",deger)


print('------------------SVM------------------------------------------')
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print('------------------SVM------------------------------------------')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
