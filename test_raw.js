require('dotenv').config();
const axios = require('axios');

async function testRaw() {
    const key = process.env.GEMINI_API_KEY;
    const url = `https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=${key}`;
    
    try {
        console.log("Testing raw fetch to v1...");
        const res = await axios.post(url, {
            contents: [{ parts: [{ text: "Merhaba" }] }]
        });
        console.log("Success v1:", res.data.candidates[0].content.parts[0].text);
    } catch (e) {
        console.error("Error raw v1:", e.response ? e.response.data : e.message);
        
        const urlBeta = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`;
        try {
            console.log("Testing raw fetch to v1beta...");
            const res = await axios.post(urlBeta, {
                contents: [{ parts: [{ text: "Merhaba" }] }]
            });
            console.log("Success v1beta:", res.data.candidates[0].content.parts[0].text);
        } catch (e2) {
            console.error("Error raw v1beta:", e2.response ? e2.response.data : e2.message);
        }
    }
}

testRaw();
