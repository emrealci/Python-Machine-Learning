import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv(r"C:\Users\DELL\Desktop\trainingData.csv")

# PCA'SIZ!!!!!!
# Bağımsız değişkenleri seçin (Attribute 1-520 sütunları)
X = dataset.iloc[:, 0:520]

# Bağımlı değişken olarak 'BUILDINGID' sütununu seçin
y_building = dataset["BUILDINGID"]
y_floor = dataset["FLOOR"]
y_position = dataset["RELATIVEPOSITION"]

# Eğitim ve test kümesine ayırma
X_train, X_test, y_train_building, y_test_building = train_test_split(X, y_building, test_size=0.2, random_state=0)
X_train, X_test, y_train_floor, y_test_floor = train_test_split(X, y_floor, test_size=0.2, random_state=0)
X_train, X_test, y_train_position, y_test_position = train_test_split(X, y_position, test_size=0.2, random_state=0)

# ANN modelini oluşturma ve eğitme
ann_building = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
ann_floor = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
ann_position = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

ann_building.fit(X_train, y_train_building)
ann_floor.fit(X_train, y_train_floor)
ann_position.fit(X_train, y_train_position)

# Test kümesi üzerinde tahmin yapma
y_pred_building = ann_building.predict(X_test)
y_pred_floor = ann_floor.predict(X_test)
y_pred_position = ann_position.predict(X_test)

# Modelin doğruluk skorlarını hesaplama
accuracy_building = accuracy_score(y_test_building, y_pred_building)
accuracy_floor = accuracy_score(y_test_floor, y_pred_floor)
accuracy_position = accuracy_score(y_test_position, y_pred_position)

print("ANN modelinin doğruluk skoru PCA'SIZ (BUILDINGID):", accuracy_building)
print("ANN modelinin doğruluk skoru PCA'SIZ (FLOOR):", accuracy_floor)
print("ANN modelinin doğruluk skoru PCA'SIZ (RELATIVEPOSITION):", accuracy_position)

# PCA uygulama kodu
X_wap = dataset.iloc[:, 0:520].values
scaler = StandardScaler()
X_wap_scaled = scaler.fit_transform(X_wap)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_wap_scaled)

# PCA sonuçlarıyla yeni veri kümesi oluşturulması
pca_df = pd.DataFrame(data=X_pca, columns=[f"PCA_{i+1}" for i in range(100)])
dataset_pca = pd.concat([pca_df, dataset.iloc[:, 520:530]], axis=1)

# PCA'lı veri kümesinin ayrılması
X_train_pca, X_test_pca, y_train_building_pca, y_test_building_pca = train_test_split(X_pca, y_building, test_size=0.2, random_state=0)
X_train_pca, X_test_pca, y_train_floor_pca, y_test_floor_pca = train_test_split(X_pca, y_floor, test_size=0.2, random_state=0)
X_train_pca, X_test_pca, y_train_position_pca, y_test_position_pca = train_test_split(X_pca, y_position, test_size=0.2, random_state=0)

# PCA'lı ANN modelini oluşturma ve eğitme
ann_building_pca = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
ann_floor_pca = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
ann_position_pca = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

ann_building_pca.fit(X_train_pca, y_train_building_pca)
ann_floor_pca.fit(X_train_pca, y_train_floor_pca)
ann_position_pca.fit(X_train_pca, y_train_position_pca)

# PCA'lı test kümesi üzerinde tahmin yapma
y_pred_building_pca = ann_building_pca.predict(X_test_pca)
y_pred_floor_pca = ann_floor_pca.predict(X_test_pca)
y_pred_position_pca = ann_position_pca.predict(X_test_pca)

# PCA'lı modelin doğruluk skorlarını hesaplama
accuracy_building_pca = accuracy_score(y_test_building_pca, y_pred_building_pca)
accuracy_floor_pca = accuracy_score(y_test_floor_pca, y_pred_floor_pca)
accuracy_position_pca = accuracy_score(y_test_position_pca, y_pred_position_pca)

print("ANN modelinin doğruluk skoru PCA'LI (BUILDINGID):", accuracy_building_pca)
print("ANN modelinin doğruluk skoru PCA'LI (FLOOR):", accuracy_floor_pca)
print("ANN modelinin doğruluk skoru PCA'LI (RELATIVEPOSITION):", accuracy_position_pca)
