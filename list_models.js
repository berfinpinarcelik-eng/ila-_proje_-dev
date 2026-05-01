require('dotenv').config();
const axios = require('axios');

async function listModels() {
    const key = process.env.GEMINI_API_KEY;
    const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${key}`;
    
    try {
        console.log("Listing available models...");
        const res = await axios.get(url);
        console.log("Available models:", JSON.stringify(res.data, null, 2));
    } catch (e) {
        console.error("Error listing models:", e.response ? e.response.data : e.message);
    }
}

listModels();
