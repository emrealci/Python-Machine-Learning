import pandas as pd
import time

def append_rows(source_file, target_file, chunk_size, interval):
    chunk_num = 0
    
    while True:
        df = pd.read_csv(source_file, skiprows=chunk_num*chunk_size, nrows=chunk_size)
        if df.empty:
            break
        
        chunk_num += 1
        df.to_csv(target_file, mode='a', header=chunk_num == 1, index=False)
        
        print(f"{chunk_num}. {chunk_size} satır {target_file} dosyasına yazıldı.")
        time.sleep(interval)

if __name__ == "__main__":
    source_csv = (r"C:\Users\Emre\Desktop\Aselsan Proje 2 ML\trainingData.csv")  # Kaynak CSV dosyanızın adını burada belirtin
    target_csv = (r"C:\Users\Emre\Desktop\liveData.csv")   # Hedef CSV dosyanızın adı
    chunk_size = int(input("Her seferinde kaç satır alınsın:"))
    interval = int(input("Kaç saniye aralıkla alınsın:"))
    print("")
    append_rows(source_csv, target_csv, chunk_size, interval)
