import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import base64
from PIL import Image
import io

load_dotenv()

class PharmaGuardAgents:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llava_url = os.getenv("LLAVA_API_URL", "http://localhost:11434/api/generate")

        self._orchestrator = None
        self._embeddings = None

        self.vector_store_path = "data/chroma_db"
        self.corpus_path = "data/corpus/"

        os.makedirs(self.corpus_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)

    def _is_demo_mode(self) -> bool:
        key = self.groq_api_key or ""
        return not key or "your_" in key.lower()

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from langchain_groq import ChatGroq
            self._orchestrator = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=self.groq_api_key,
                temperature=0.1
            )
        return self._orchestrator

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def vision_scanner(self, image_bytes: bytes) -> Dict[str, Any]:
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
        """
        try:
            response = requests.post(
                self.llava_url,
                json={"model": "llava:v1.6", "prompt": prompt, "stream": False, "images": [encoded_image], "format": "json"},
                timeout=30
            )
            if response.status_code == 200:
                return json.loads(response.json().get('response', '{}'))
            return {"error": "LLaVA not responding"}
        except:
            return {"error": "LLaVA connection failed"}

    def rag_specialist(self, query: str) -> str:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        embeddings = self._get_embeddings()
        if not os.path.exists(self.vector_store_path):
            files = [f for f in os.listdir(self.corpus_path) if f.endswith(".pdf")]
            if not files: return "HATA: Veri havuzu bos."
            all_docs = []
            for file in files:
                loader = PyPDFLoader(os.path.join(self.corpus_path, file))
                all_docs.extend(loader.load())
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=self.vector_store_path)
        else:
            vectorstore = Chroma(persist_directory=self.vector_store_path, embedding_function=embeddings)
        
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        return f"CONTEXT:\n{context}"

    def _mock_orchestration(self, drug_name: str) -> str:
        from datetime import datetime
        name = drug_name if drug_name and drug_name != "NOT_READABLE" else "Parol 500mg"
        return f"""
# Pharma-Guard AI: Rapor (DEMO)
**İlaç:** {name} | **Mod:** Groq Simulation
Sistem şu an demo modunda çalışmaktadır.
"""

    def orchestrate(self, input_data: Any, is_image: bool = True) -> str:
        try:
            if self._is_demo_mode(): return self._mock_orchestration("İlaç")
            vision_info = self.vision_scanner(input_data) if is_image else {}
            
            # More robust and professional prompt
            system_prompt = """
            Sen Pharma-Guard AI asistanısın, uzman bir tıbbi farmakolog ve görsel analiz uzmanı olarak görev yapıyorsun.
            Llama-3.3 altyapısını kullanarak, kullanıcıya yüklediği ilaç hakkında profesyonel, doğru ve anlaşılır bir analiz raporu sunmalısın.
            
            RAPOR FORMATI:
            1. [İlaç Bilgileri]: Marka adı, etken madde ve dozaj.
            2. [Kullanım Amacı]: Bu ilaç ne için kullanılır? (RAG verisine ve genel tıbbi bilgine dayan)
            3. [Prospektüs Özet]: Kullanım talimatları ve önemli uyarılar.
            4. [Güvenlik Notu]: Yan etkiler ve dikkat edilmesi gerekenler.
            
            Kural: Eğer RAG verisi yetersizse, kendi tıbbi bilgi birikimini kullanarak ilacı açıkla ancak bunun genel bilgi olduğunu belirt.
            Analizlerini mutlaka TÜRKÇE yap.
            """
            
            drug_name = vision_info.get("brand_name", "")
            rag_context = self.rag_specialist(drug_name) if drug_name and drug_name != "NOT_READABLE" else "RAG verisi bulunamadı."
            
            full_prompt = f"""
            Görsel Analiz Sonuçları: {json.dumps(vision_info)}
            RAG (Prospektüs) Verisi: {rag_context}
            
            Lütfen yukarıdaki verilere dayanarak detaylı ve profesyonel bir ilaç analiz raporu oluştur. 
            Eğer görselden ilaç adı tam okunmuyorsa, kullanıcıdan daha net bir fotoğraf iste.
            """
            
            if is_image and ("error" in vision_info or not vision_info or vision_info.get("brand_name") == "NOT_READABLE"):
                from langchain_groq import ChatGroq
                vision_model = ChatGroq(model="llama-3.2-11b-vision-preview", groq_api_key=self.groq_api_key)
                image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(input_data).decode('utf-8')}"}}
                response = vision_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=[{"type": "text", "text": full_prompt}, image_content])])
            else:
                response = self._get_orchestrator().invoke([SystemMessage(content=system_prompt), HumanMessage(content=full_prompt)])
            return response.content
        except Exception as e:
            if any(x in str(e).upper() for x in ["QUOTA", "429", "LIMIT"]):
                return self._mock_orchestration("KOTA DOLDU - DEMO")
            return self._mock_orchestration(f"Hata: {str(e)[:50]}")
