import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import base64
from PIL import Image
import io

load_dotenv()

class PharmaGuardAgents:
    def __init__(self):
        # API Keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llava_url = os.getenv("LLAVA_API_URL", "http://localhost:11434/api/generate")

        # Models
        self.orchestrator = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.google_api_key)
        self.fast_analyser = ChatGroq(model="llama3-70b-8192", groq_api_key=self.groq_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        
        self.vector_store_path = "data/chroma_db"
        self.corpus_path = "data/corpus/"
        
        os.makedirs(self.corpus_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)

    def vision_scanner(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        [Vision-Scanner]: LLaVA mimarisini kullanarak görseli tara.
        """
        # Base64 encode image for LLaVA API (Assumed Ollama-like structure)
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = """
        Analyze this medicine package image. Extract the following information in JSON format:
        {
            "brand_name": "Commercial name of the medicine",
            "active_ingredient": "Chemical name/active substance",
            "dosage": "mg or ml value",
            "form": "Tablet, syrup, etc.",
            "barcode": "Barcode if visible"
        }
        If text is not readable, say 'NOT_READABLE' for that field. 
        Focus strictly on accuracy.
        """

        try:
            # Attempt local Ollama LLaVA call
            response = requests.post(
                self.llava_url,
                json={
                    "model": "llava:v1.6",
                    "prompt": prompt,
                    "stream": False,
                    "images": [encoded_image],
                    "format": "json"
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get('response', '{}')
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # Clean the result if it contains markdown code blocks
                    cleaned_result = result.strip().replace("```json", "").replace("```", "").strip()
                    try:
                        return json.loads(cleaned_result)
                    except:
                        return {"error": "Görsel analiz sonucu beklenen formatta değil (JSON hatası)."}
            else:
                # Fallback to Gemini Vision if LLaVA is not available, 
                # but following the prompt's LLaVA instruction as priority.
                return {"error": f"LLaVA error: {response.status_code}"}
        except Exception as e:
            return {"error": f"LLaVA connection failed: {str(e)}"}

    def rag_specialist(self, query: str) -> str:
        """
        [RAG-Specialist]: PDF prospektüsleri semantik olarak tara.
        """
        # Load or Create Vector DB
        if not os.path.exists(self.vector_store_path):
            # Check if corpus has files
            files = [f for f in os.listdir(self.corpus_path) if f.endswith(".pdf")]
            if not files:
                return "HATA: Prospektüs veri havuzu boş. Lütfen PDF dosyalarını data/corpus/ klasörüne ekleyin."
            
            all_docs = []
            for file in files:
                loader = PyPDFLoader(os.path.join(self.corpus_path, file))
                all_docs.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings, 
                persist_directory=self.vector_store_path
            )
        else:
            vectorstore = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
        
        if not query or len(query.strip()) < 2:
            return "Sorgu çok kısa veya boş. Prospektüs araması yapılamadı."
        
        # Search
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([os.path.basename(doc.metadata['source']) for doc in results]))
        
        return f"CONTEXT:\n{context}\n\nSOURCES: {', '.join(sources)}"

    def _mock_orchestration(self, drug_name: str) -> str:
        """
        Simulated high-quality report for Demo Mode.
        """
        import time
        from datetime import datetime
        time.sleep(1.5)
        
        name = drug_name if drug_name and drug_name != "NOT_READABLE" else "Örnek İlaç (Parol 500mg)"
        
        return f"""
# Pharma-Guard AI: Tibbi Analiz Raporu (DEMO)

**Analiz Edilen Ilac:** {name}
**Analiz Tarihi:** {datetime.now().strftime("%d/%m/%Y %H:%M")}

---

### [1] Gorsel Tarama Bulgulari (Vision-Scanner)
*   **Marka Adi:** {name}
*   **Etkin Madde:** Parasetamol (Simule Edildi)
*   **Dozaj:** 500 mg
*   **Form:** Tablet
*   **Barkod:** 8699514010103 (Ornek)

### [2] Prospektus Dogrulamasi (RAG-Specialist)
*Sistemimiz simulasyon veri tabanindan ilgili prospektusu basariyla cekti.*
*   **Endikasyon:** Hafif ve orta siddetli agrilar ile atesin semptomatik tedavisinde kullanilir.
*   **Yan Etkiler:** Cok seyrek olarak alerjik reaksiyonlar rapor edilmistir.

### [3] Guvenlik ve Uyumluluk Kontrolu (Safety-Auditor)
*   [OK] Gorseldeki dozaj ve form prospektus ile uyumludur.
*   [!] Onemli: Karaciger hastalari icin dozaj ayarlamasi gerekebilir.

---

### [4] Uzman Ozeti ve Tavsiye
Bu calisma bir Simulasyon (Demo) raporudur. Gercek API anahtarlari girildiginde, bu raporlar canli tibbi veritabani (RAG) ve gercek zamanli gorsel analiz (LLaVA/Gemini) ile üretilecektir.

**Sistem Durumu:** OK - DEMO MODU CALISIYOR
"""

    def orchestrate(self, input_data: Any, is_image: bool = True) -> str:
        """
        PG-MO: Master Orchestrator logic.
        """
        # Check if we should run in Demo Mode
        is_demo = not (self.google_api_key and "your_" not in self.google_api_key)
        
        if is_demo:
            return self._mock_orchestration("Tespit Edilen İlaç")

        vision_info = {}
        if is_image:
            vision_info = self.vision_scanner(input_data)
            if "error" in vision_info:
                # Fallback to Gemini Vision if LLaVA fails (for robustness in development)
                # But we should inform the orchestrator.
                pass
        
        # System Master Prompt
        system_prompt = """
        ### ROLE: PHARMA-GUARD MASTER ORCHESTRATOR (PG-MO) ###
        Sen, Gemini 2.0 tabanlı, multimodal yeteneklere sahip ve çoklu ajan ekosistemini yöneten baş mimarsın. 
        Görevin; görsel veya metinsel girişi alınan bir ilacı, sıfır hata toleransı ile analiz etmektir.

        KURAL 1: Yazı okunmuyorsa asla tahmin etme! Kullanıcıyı "Fotoğrafı daha ışıklı bir yerde çek" diye uyar.
        KURAL 2: Bilgi kaynağın %100 tıbbi prospektüsler olmalı.
        KURAL 3: Bilgiler arasında 1 mg fark olsa bile raporu blokla ve 'VERİ UYUŞMAZLIĞI' alarmı ver.

        ### AJANLARDAN GELEN VERİLERİ SENTEZLE VE RAPORLA ###
        """

        # Build context for the orchestrator
        input_context = f"Vision Scanner Results: {json.dumps(vision_info)}\n"
        
        # Trigger RAG based on extracted drug name
        drug_name = vision_info.get("brand_name", "")
        rag_context = ""
        if drug_name and drug_name != "NOT_READABLE":
            rag_context = self.rag_specialist(drug_name)
        
        full_prompt = f"{system_prompt}\n\nVision Info: {input_context}\n\nRAG Context: {rag_context}\n\nLütfen detaylı raporu oluştur."
        
        content_parts = [{"type": "text", "text": full_prompt}]
        
        if is_image and "error" in vision_info:
            encoded_image = base64.b64encode(input_data).decode('utf-8')
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_parts)
        ]
        
        response = self.orchestrator.invoke(messages)
        return response.content
