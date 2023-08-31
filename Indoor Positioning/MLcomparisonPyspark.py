from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import time
import os

# Spark oturumu oluşturma
spark = SparkSession.builder.appName("LiveML").getOrCreate()

# CSV dosyasının path'i
file_path = (r"C:\Users\Emre\Desktop\liveData.csv")

# Bağımsız değişkenlerin sütun adları
start_number = 1
end_number = 519

feature_columns = []

for number in range(start_number, end_number + 1):
    padded_number = str(number).zfill(3)
    feature_columns.append(f'WAP{padded_number}')

# Hedef değişkenin sütun adı
target_column = "BUILDINGID"

# Klasör oluştur (varsa tekrar oluşturmayacak)
result_folder = "model_results"
os.makedirs(result_folder, exist_ok=True)

# Veriyi okuyarak DataFrame'e dönüştürme
def read_csv_and_create_df(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

# Veriyi işleyerek özellik vektörleri oluşturma
def preprocess_data(df):
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(df)
    return assembled_df

# Modeli eğit ve tahminleri yapma
def train_and_evaluate_regression_models(data):
    models = [
        RandomForestRegressor(featuresCol="features", labelCol=target_column, numTrees=10),
        LinearRegression(featuresCol="features", labelCol=target_column)
    ]
    
    model_accuracies = {}
    
    for model in models:
        model_name = model.__class__.__name__
        print(f"{model_name} modeli eğitiliyor...")
        trained_model = model.fit(data)
        
        print(f"{model_name} modeli değerlendiriliyor...")
        predictions = trained_model.transform(data)
        evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
        r2 = evaluator.evaluate(predictions)
        
        model_accuracies[model_name] = r2
        
        print(f"{model_name} Model Doğruluk Skoru (R-kare):", r2)
        print("\n")
        
        # Accuracy değerlerini dosyaya yazma
        with open(os.path.join(result_folder, f"{model_name}_accuracy.txt"), "a") as f:
            f.write(str(r2) + ",")
        
    return model_accuracies

# Modeli eğit ve tahminleri yapma
def train_and_evaluate_classification_models(data):
    models = [
        LogisticRegression(featuresCol="features", labelCol=target_column),
        RandomForestClassifier(featuresCol="features", labelCol=target_column, numTrees=10)
    ]
    
    model_accuracies = {}
    
    for model in models:
        model_name = model.__class__.__name__
        print(f"{model_name} modeli eğitiliyor...")
        trained_model = model.fit(data)
        
        print(f"{model_name} modeli değerlendiriliyor...")
        predictions = trained_model.transform(data)
        evaluator = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        model_accuracies[model_name] = accuracy
        
        print(f"{model_name} Model Doğruluk Skoru (Accuracy):", accuracy)
        print("\n")
        
        # Accuracy değerlerini dosyaya yazma
        with open(os.path.join(result_folder, f"{model_name}_accuracy.txt"), "a") as f:
            f.write(str(accuracy) + ",")
        
    return model_accuracies

def main():
    try:
        while True:
            # Veriyi oku ve DataFrame'e çevirme
            data_df = read_csv_and_create_df(file_path)
            
            # Veriyi işleme
            processed_data_df = preprocess_data(data_df)
            
            print("Regresyon Modelleri:")
            regression_accuracies = train_and_evaluate_regression_models(processed_data_df)
            
            print("Sınıflandırma Modelleri:")
            classification_accuracies = train_and_evaluate_classification_models(processed_data_df)
            
            time.sleep(3)
    except KeyboardInterrupt:
        print("İşlem durduruldu.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
