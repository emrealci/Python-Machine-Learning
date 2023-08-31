from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time

#------------------------------#
#SparkSession# --> Spark işlemlerini başlatma, DataFrame oluşturma, veri okuma/yazma
#gibi temel işlemleri gerçekleştirmek için
#------------------------------#
#VectorAssembler# --> Genellikle makine öğrenimi modellerine giriş olarak veri hazırlamak için kullanılır.
#veriyi vector haline getirir.
#------------------------------#
#RandomForestRegressor# --> Random Forest algoritması ile regresyon problemleri için kullanılır
#------------------------------#

# Spark oturumu oluşturma
spark = SparkSession.builder.appName("LiveML").getOrCreate()

# CSV dosyasının path'i
file_path = r"C:\Users\DELL\Desktop\fake_data.csv"

# Bağımsız değişkenlerin sütun adları
feature_columns = ["yas", "deneyim"]

# Hedef değişkenin sütun adı
target_column = "maas"

# Veriyi okuyarak DataFrame'e dönüştür
def read_csv_and_create_df(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    #header=True --> PySpark otomatik olarak ilk satırı sütun isimleri olarak kabul eder
    #inferSchema=True --> PySpark otomatik olarak her sütunun veri tipini tahmin eder
    #inferSchema=False --> tüm sütunları varsayılan olarak string veri tipi olarak kabul eder
    return df


# Veriyi (bağımsız değişkenleri) işleyerek özellik vektörleri oluşturma
def preprocess_data(df):
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(df)
    #assembled_df --> df DataFrame'i, orijinal bagımsız degiskenler sütunlarına ek olarak 
                      #vektör olarak birleştirilmiş yeni bir sütun içerir. 
    return assembled_df
    # Vektöre çevirmek şart mı?
         # - Bazı PySpark algoritmaları, veriyi vektör formatında bekler.
         # - Bu nedenle doğrudan değişkenleri kullanmanız bazı algoritmalarla uyumsuzluk yaratabilir.
         
         
# Modeli eğit
def train_model(data):
    rf = RandomForestRegressor(featuresCol="features", labelCol=target_column, numTrees=10)
    model = rf.fit(data)
    return model


# Modeli değerlendir ve sonuçları yazdırma
def evaluate_model(model, data):
    predictions = model.transform(data)
    #evaluator = RegressionEvaluator(labelCol="gercek_deger_sutunu", predictionCol="tahmin_sutunu", metricName="r2")
    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
    #
    r2 = evaluator.evaluate(predictions)
    return r2

    #Regresyon problemleri için yaygın olarak kullanılan metriklerden bazıları şunlardır:
    
        #Ortalama Mutlak Hata (MAE): Tahmin edilen değerler ile gerçek değerler arasındaki mutlak farkların ortalamasıdır.
        #Ortalama Kare Hata (MSE): Tahmin edilen değerler ile gerçek değerler arasındaki kare farkların ortalamasıdır.
        #Kök Ortalama Kare Hata (RMSE): MSE'nin kareköküdür, tahminlerin gerçek değerlerden ne kadar uzak olduğunu ölçer.
        #R-kare (Coefficient of Determination): Tahmin edilen değerlerin varyansının gerçek değerlerin varyansına oranıdır.


def main():
    try:
        while True:
            # Veriyi oku ve DataFrame'e çevirme
            data_df = read_csv_and_create_df(file_path)
            
            # Veriyi işleme
            processed_data_df = preprocess_data(data_df)
            
            # Modeli eğitme
            model = train_model(processed_data_df)
            
            # Modeli değerlendirme
            r2 = evaluate_model(model, processed_data_df)
            
            print("Model Doğruluk Skoru (R-kare):", r2)
            
            time.sleep(3)
    except KeyboardInterrupt:
        print("İşlem durduruldu.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
