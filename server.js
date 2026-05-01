require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname)); // Statik dosyalari (HTML, JS, CSS) sunmak icin

// API Key kontrolü
if (!process.env.GROQ_API_KEY) {
    console.log("UYARI: .env dosyasinda GROQ_API_KEY eksik! Gemini denenecek.");
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.post('/api/claude', async (req, res) => {
    console.log("İstek geldi, işleniyor...");
    const { messages, system } = req.body;

    // ÖNCE GROQ DENE (Daha hızlı ve kota sorunu az)
    if (process.env.GROQ_API_KEY) {
        try {
            console.log("Groq API kullaniliyor...");
            const groqMsgs = [];
            if (system) groqMsgs.push({ role: 'system', content: system });
            
            if (Array.isArray(messages)) {
                messages.forEach(m => groqMsgs.push({ role: m.role || 'user', content: m.content }));
            } else {
                groqMsgs.push({ role: 'user', content: messages });
            }

            const groqRes = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
                model: 'llama-3.3-70b-versatile',
                messages: groqMsgs,
                temperature: 0.7
            }, {
                headers: {
                    'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 15000
            });

            const text = groqRes.data.choices[0].message.content;
            console.log("Groq basarili.");
            return res.json({ content: [{ text: text }] });
        } catch (error) {
            console.error("Groq Hatasi:", error.message);
            // Groq fail olursa Gemini'ye gec
        }
    }

    // YEDEK OLARAK GEMINI DENE
    try {
        console.log("Gemini API deneniyor...");
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
        
        let fullPrompt = "";
        if (system) fullPrompt += `System: ${system}\n\n`;
        
        if (Array.isArray(messages)) {
            fullPrompt += messages.map(m => `${m.role === 'user' ? 'Kullanıcı' : 'Asistan'}: ${m.content}`).join("\n");
        } else {
            fullPrompt += messages;
        }

        const result = await model.generateContent(fullPrompt);
        const response = await result.response;
        const text = response.text();

        console.log("Gemini basarili.");
        res.json({ content: [{ text: text }] });
    } catch (error) {
        console.error("Gemini Hatası:", error.message);
        
        // Eğer her iki API de başarısız olursa
        res.status(500).json({ 
            error: "Tüm API servisleri şu an meşgul.",
            message: error.message 
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`SUNUCU ${PORT} PORTUNDA CALISIYOR`);
});