import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"C:\Users\DELL\Desktop\trainingData.csv")

# PCA'SIZ!!!!!!
# Bağımsız değişkenleri seçin (Attribute 1-520 sütunları)
X = dataset.iloc[:, 0:520]

# Bağımlı değişken olarak 'BUILDINGID' sütununu seç
y_building = dataset["BUILDINGID"]

y_floor = dataset["FLOOR"]

# Bağımlı değişken olarak 'RELATIVEPOSITION' sütununu seç
y_position = dataset["RELATIVEPOSITION"]

# Eğitim ve test kümesine ayırma
X_train, X_test, y_train_building, y_test_building = train_test_split(X, y_building, test_size=0.2, random_state=0)
X_train, X_test, y_train_floor, y_test_floor = train_test_split(X, y_floor, test_size=0.2, random_state=0)
X_train, X_test, y_train_position, y_test_position = train_test_split(X, y_position, test_size=0.2, random_state=0)


# K-NN modelini oluşturma ve eğitme
knn_building = KNeighborsClassifier(n_neighbors=5)
knn_floor = KNeighborsClassifier(n_neighbors=5)
knn_position = KNeighborsClassifier(n_neighbors=5)

knn_building.fit(X_train, y_train_building)
knn_floor.fit(X_train, y_train_floor)
knn_position.fit(X_train, y_train_position)

# Test kümesi üzerinde tahmin yapma
y_pred_building = knn_building.predict(X_test)
y_pred_floor = knn_floor.predict(X_test)
y_pred_position = knn_position.predict(X_test)

# Modelin doğruluk skorlarını hesaplama
accuracy_building = accuracy_score(y_test_building, y_pred_building)
accuracy_floor = accuracy_score(y_test_floor, y_pred_floor)
accuracy_position = accuracy_score(y_test_position, y_pred_position)

print("K-NN modelinin doğruluk skoru PCA'SIZ (BUILDINGID):", accuracy_building)
print("K-NN modelinin doğruluk skoru PCA'SIZ (FLOOR):", accuracy_floor)
print("K-NN modelinin doğruluk skoru PCA'SIZ (RELATIVEPOSITION):", accuracy_position)


# WAP özelliklerini seçin (Attribute 1-520 sütunları)
X_wap = dataset.iloc[:, 0:520].values

# WAP özellikleriını gerektirni ölçeklendirin (PCA, veri setinin ölçekli olmasir)
scaler = StandardScaler()
X_wap_scaled = scaler.fit_transform(X_wap)

# PCA uygulayın ve boyutu belirleyin (örneğin, 100 boyutlu alt uzaya dönüştürün)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_wap_scaled)

# PCA sonuçlarını DataFrame'e dönüştürün
pca_df = pd.DataFrame(data=X_pca, columns=[f"PCA_{i+1}" for i in range(100)])

# WAP özelliklerini çıkarın ve PCA sonuçlarını ve diğer özellikleri birleştirin
dataset = pd.concat([pca_df, dataset.iloc[:, 520:530]], axis=1)


# Her bir PCA bileşeni için Box Plot oluşturarak aykırı değerleri görselleştirme
plt.figure(figsize=(12, 6))
sns.boxplot(data=pca_df)
plt.title('PCA Sonrası Aykırı Değerler - Box Plot')
plt.xlabel('PCA Bileşeni')
plt.ylabel('Değer')


X_PCA = pca_df

y_building_PCA = dataset["BUILDINGID"]

y_floor_PCA = dataset["FLOOR"]

y_position_PCA = dataset["RELATIVEPOSITION"]

# Eğitim ve test kümesine ayırma
X_train_PCA, X_test_PCA, y_train_building_PCA, y_test_building_PCA = train_test_split(X_PCA, y_building_PCA, test_size=0.2, random_state=0)
X_train_PCA, X_test_PCA, y_train_floor_PCA, y_test_floor_PCA = train_test_split(X_PCA, y_floor_PCA, test_size=0.2, random_state=0)
X_train_PCA, X_test_PCA, y_train_position_PCA, y_test_position_PCA = train_test_split(X_PCA, y_position_PCA, test_size=0.2, random_state=0)

# K-NN modelini oluşturma ve eğitme
knn_building_PCA = KNeighborsClassifier(n_neighbors=5)
knn_floor_PCA = KNeighborsClassifier(n_neighbors=5)
knn_position_PCA = KNeighborsClassifier(n_neighbors=5)

knn_building_PCA.fit(X_train_PCA, y_train_building_PCA)
knn_floor_PCA.fit(X_train_PCA, y_train_floor_PCA)
knn_position_PCA.fit(X_train_PCA, y_train_position_PCA)

# Test kümesi üzerinde tahmin yapma
y_pred_building_PCA = knn_building_PCA.predict(X_test_PCA)
y_pred_floor_PCA = knn_floor_PCA.predict(X_test_PCA)
y_pred_position_PCA = knn_position_PCA.predict(X_test_PCA)

# Modelin doğruluk skorlarını hesaplama
accuracy_building_PCA = accuracy_score(y_test_building_PCA, y_pred_building_PCA)
accuracy_floor_PCA = accuracy_score(y_test_floor_PCA, y_pred_floor_PCA)
accuracy_position_PCA = accuracy_score(y_test_position_PCA, y_pred_position_PCA)

print("K-NN modelinin doğruluk skoru PCA'LI (BUILDINGID):", accuracy_building_PCA)
print("K-NN modelinin doğruluk skoru PCA'LI (FLOOR):", accuracy_floor_PCA)
print("K-NN modelinin doğruluk skoru PCA'LI (RELATIVEPOSITION):", accuracy_position_PCA)

plt.show()
