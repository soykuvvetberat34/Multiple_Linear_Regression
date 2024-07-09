import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veriyi okuma
datas = pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\çoklu doğrusal regresyon\\deneme.csv")

# Eksik verilerin olduğu yerlere sütundaki tüm verilerin ortalamasını koyma
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = datas.iloc[:, 1:4].values
imputer = imputer.fit(age)  # Değerleri eğitir
age = imputer.transform(age)  # Değerleri değiştirir
age = age.astype(int)  # Tüm yaş değerlerini int yapar
print(age)
age_df = pd.DataFrame(age, columns=["boy", "kilo", "yaş"])

# Encoding işlemleri
# Ülke isimleri için encoding işlemi
countrys = datas.iloc[:, 0].values.reshape(-1, 1)
# One-Hot encoding
OHE = OneHotEncoder()
OHE_Countrys = OHE.fit_transform(countrys).toarray()
OHE_Countrys = OHE_Countrys.astype(int)
df_countrys = pd.DataFrame(data=OHE_Countrys, columns=OHE.categories_[0])
print(df_countrys)

# Cinsiyet değerleri için encoding işlemi
genders = datas.iloc[:, -1].values.reshape(-1, 1).ravel()#reshape(-1, 1) ifadesi tek boyutlu diziyi iki boyutlu hale getirir. Ancak LabelEncoder kullanırken .ravel() kullanarak verileri tek boyutlu diziye dönüştürmek gereklidir.
# Label encoding
LE_gender = LabelEncoder()
genders_LE = LE_gender.fit_transform(genders)
gender_df = pd.DataFrame(genders_LE, columns=["genders"])
print(gender_df)

# Verileri birleştirme
res1 = pd.concat([df_countrys, age_df], axis=1)
res2 = pd.concat([res1, gender_df], axis=1)
print(res2)
res3=res2
# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(res1, gender_df, test_size=0.33, random_state=0)

# Model oluşturma ve eğitme
regressor = LinearRegression()
regressor.fit(x_train, y_train)  # x_train verileri ile y_train verileri arasındaki ilişkiyi anla ve eğit
predict = regressor.predict(x_test)  # Çıktıyı y_test verileri ile kıyaslayacağız
print(predict)


#boy verilerini tahmin etme
boy=datas.iloc[:,1].values.reshape(-1,1)
print(len(boy))
res3.drop('boy',axis=1,inplace=True)
x_train2,x_test2,y_train2,y_test2=train_test_split(res3,boy,test_size=0.33,random_state=1)
regressor2=LinearRegression()
regressor2.fit(x_train2,y_train2)
predct2=regressor2.predict(x_test2)
predct2=predct2.astype(int)
print(predct2)
print(y_test2)







