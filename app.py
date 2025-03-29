import os
import requests
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class OllamaParallelProcessor:
    def __init__(self, ports=[11439, 11440], model_name="llama3.2:3b", max_workers=2):
        self.ports = ports
        self.model_name = model_name
        self.max_workers = max_workers
        self.db_directory = "pdf_concurrency/chroma_dbs"
        
        # Chroma DB klasörünü oluştur
        if not os.path.exists(self.db_directory):
            os.makedirs(self.db_directory)
        
        # Mevcut veritabanlarını sakla
        self.available_dbs = self.get_available_dbs()
        
        # Ollama servislerinin çalışıp çalışmadığını kontrol et
        self.check_ollama_services()
    
    def check_ollama_services(self):
        """Ollama servislerinin çalışıp çalışmadığını kontrol eder"""
        print("Ollama servislerini kontrol ediyorum...")
        for port in self.ports:
            try:
                url = f"http://0.0.0.0:{port}/api/tags"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Port {port} çalışıyor!")
                else:
                    print(f"UYARI: Port {port} cevap veriyor ancak HTTP kodu: {response.status_code}")
            except Exception as e:
                print(f"HATA: Port {port} çalışmıyor veya erişilemiyor: {str(e)}")
    
    def get_available_dbs(self):
        """Mevcut ChromaDB veritabanlarını listeler"""
        if not os.path.exists(self.db_directory):
            return []
        
        return [d for d in os.listdir(self.db_directory) if os.path.isdir(os.path.join(self.db_directory, d))]
    
    def generate_embeddings_batch(self, texts, port):
        """Metin grubunun embeddings'lerini oluşturur"""
        url = f"http://0.0.0.0:{port}/api/embeddings"
        
        results = []
        for text in texts:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": text
                }
                response = requests.post(url, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding")
                    if embedding:
                        results.append(embedding)
                    else:
                        print(f"UYARI: Embedding sonucu bulunamadı: {result}")
                else:
                    print(f"HATA: Embedding başarısız. Status: {response.status_code}")
            except Exception as e:
                print(f"HATA: Embedding oluşturulurken bir istisna oluştu: {str(e)}")
        
        return results

    def process_pdf(self, pdf_path):
        """PDF'i işler ve bir ChromaDB veritabanı oluşturur"""
        if not os.path.exists(pdf_path):
            return f"Hata: {pdf_path} bulunamadı"
        
        try:
            print(f"PDF dosyası yükleniyor: {pdf_path}")
            # PDF dosyasının adını al (uzantısız)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # ChromaDB yolunu oluştur
            db_path = os.path.join(self.db_directory, pdf_name)
            
            # PDF'i yükle
            loader = PyPDFLoader(pdf_path)
            print("PDF yükleniyor...")
            pages = loader.load()
            print(f"PDF yüklendi. {len(pages)} sayfa.")
            
            # Belgeyi daha küçük bölütlere ayır (chunk boyutunu azalttık)
            print("Belge bölütlere ayrılıyor...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,  # Daha küçük chunk boyutu
                chunk_overlap=25  # Daha az örtüşme
            )
            splits = text_splitter.split_documents(pages)
            print(f"Belge {len(splits)} bölüte ayrıldı.")
            
            # Alternatif olarak, ChromaDB yerine sadece metinleri ve embeddings'leri saklayabilirsiniz
            print("Direk Ollama API kullanarak embeddings oluşturuluyor...")
            
            # Embeddings için alternatif yöntem
            try:
                embeddings = OllamaEmbeddings(
                    base_url=f"http://0.0.0.0:{self.ports[0]}",
                    model=self.model_name
                )
                
                # Test etmek için küçük bir örnek
                test_embedding = embeddings.embed_query("Test metni")
                print(f"Test embedding boyutu: {len(test_embedding)}")
                
                # ChromaDB'yi daha küçük belge gruplarıyla oluştur
                print("ChromaDB aşamalı olarak oluşturuluyor...")
                
                # Önce boş bir veritabanı oluştur
                db = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings
                )
                
                # Belgeleri daha küçük gruplara böl (örn. 10'ar belge)
                batch_size = 10
                num_batches = (len(splits) + batch_size - 1) // batch_size  # Ceil division
                
                # Her grubu teker teker işle ve ilerlemeyi göster
                for i in tqdm(range(num_batches), desc="ChromaDB oluşturuluyor"):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(splits))
                    current_batch = splits[start_idx:end_idx]
                    
                    # Bu grup için db'ye ekle
                    db.add_documents(documents=current_batch)
                    
                    # Her gruptan sonra verileri diske kaydet
                    db.persist()
                    
                    # İlerlemeyi göster
                    print(f"Grup {i+1}/{num_batches} tamamlandı ({end_idx}/{len(splits)} belge)")
                    
                    # Her gruptan sonra kısa bir bekleme (sistem rahatlaması için)
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"ChromaDB oluşturulurken hata: {str(e)}")
                raise e
            
            # Mevcut veritabanı listesini güncelle
            self.available_dbs = self.get_available_dbs()
            
            return f"{pdf_name} için veritabanı başarıyla oluşturuldu"
            
        except Exception as e:
            return f"PDF işlenirken hata oluştu: {str(e)}"
    
    def send_request(self, port, prompt, request_id):
        """Belirli bir porta istek gönderir"""
        try:
            url = f"http://0.0.0.0:{port}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False  # Akış yerine tam cevap al
            }
            
            response = requests.post(url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "port": port,
                    "request_id": request_id,
                    "status": "success",
                    "response": result.get("response", "")
                }
            else:
                return {
                    "port": port,
                    "request_id": request_id,
                    "status": "error",
                    "error": f"HTTP Hata: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "port": port,
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    def query_db_with_parallel_llms(self, db_name, query, num_results=5):
        """Bir veritabanını sorgular ve paralel olarak iki LLM'den cevap alır"""
        if db_name not in self.available_dbs:
            return f"Hata: {db_name} veritabanı bulunamadı. Mevcut veritabanları: {', '.join(self.available_dbs)}"
        
        try:
            # ChromaDB yolunu oluştur
            db_path = os.path.join(self.db_directory, db_name)
            
            # Embeddings için Ollama kullan (ilk portu kullan)
            embeddings = OllamaEmbeddings(
                base_url=f"http://0.0.0.0:{self.ports[0]}",
                model=self.model_name
            )
            
            # ChromaDB'yi yükle
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            # Sorguyu çalıştır ve en alakalı sonuçları al
            results = db.similarity_search(query, k=num_results)
            
            # İlgili metinleri birleştir
            context = "\n\n".join([doc.page_content for doc in results])
            
            # LLM için girdiyi hazırla
            system_prompt = "Aşağıdaki bilgiler ışığında kullanıcının sorusunu cevapla:"
            full_prompt = f"{system_prompt}\n\nBİLGİLER:\n{context}\n\nSORA: {query}\n\nCEVAP:"
            
            # Sonuçları saklamak için liste
            responses = []
            
            # Her port için benzersiz bir istek ID'si oluştur (sadece 11439 ve 11440 için)
            request_ids = [str(uuid.uuid4()) for _ in self.ports]
            
            # ThreadPoolExecutor ile paralel istekler gönder
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Her port için görevleri gönder
                futures = [executor.submit(self.send_request, port, full_prompt, request_id) 
                          for port, request_id in zip(self.ports, request_ids)]
                
                # Tamamlanan görevleri topla
                for future in futures:
                    result = future.result()
                    responses.append(result)
            
            # Başarılı yanıtları filtrele
            successful_responses = [r for r in responses if r["status"] == "success"]
            
            if not successful_responses:
                return "Hiçbir modelden geçerli bir yanıt alınamadı."
            
            # İlk başarılı yanıtı döndür (veya başka bir strateji uygulayabilirsiniz)
            # Örneğin, tüm yanıtları birleştirebilir veya en uzun/kısa olanı seçebilirsiniz
            return successful_responses[0]["response"]
            
        except Exception as e:
            return f"Sorgu işlenirken hata oluştu: {str(e)}"

# Kullanım örneği
def main():
    # Sadece 11439 ve 11440 portlarını kullan
    processor = OllamaParallelProcessor(ports=[11439, 11440])
    
    while True:
        print("\n--- Ollama PDF İşleyici ---")
        print("1. PDF Dosyasını İşle")
        print("2. Veritabanı Sorgula")
        print("3. Mevcut Veritabanlarını Listele")
        print("4. Çıkış")
        
        choice = input("Seçiminiz (1-4): ")
        
        if choice == "1":
            pdf_path = input("PDF dosyasının tam yolunu girin: ")
            print(f"PDF işlemi başlıyor: {pdf_path}")
            result = processor.process_pdf(pdf_path)
            print(result)
            
        elif choice == "2":
            if not processor.available_dbs:
                print("Hiç veritabanı yok. Önce bir PDF işleyin.")
                continue
                
            print("Mevcut veritabanları:", ", ".join(processor.available_dbs))
            db_name = input("Hangi veritabanını sorgulamak istiyorsunuz?: ")
            
            if db_name not in processor.available_dbs:
                print(f"Hata: {db_name} veritabanı bulunamadı.")
                continue
                
            query = input("Sorgunuzu girin: ")
            result = processor.query_db_with_parallel_llms(db_name, query)
            print("\nSorgu Sonucu:")
            print(result)
            
        elif choice == "3":
            if not processor.available_dbs:
                print("Hiç veritabanı yok.")
            else:
                print("Mevcut veritabanları:", ", ".join(processor.available_dbs))
                
        elif choice == "4":
            print("Programdan çıkılıyor...")
            break
            
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()