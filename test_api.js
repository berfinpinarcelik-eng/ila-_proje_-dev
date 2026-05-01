require('dotenv').config();
const { GoogleGenerativeAI } = require("@google/generative-ai");

async function test() {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-2.0-flash-exp"];
    
    for (const modelName of models) {
        try {
            console.log(`Testing ${modelName}...`);
            const model = genAI.getGenerativeModel({ model: modelName });
            const result = await model.generateContent("Merhaba");
            console.log(`Success with ${modelName}:`, result.response.text().substring(0, 50) + "...");
            return modelName; // Found a working one!
        } catch (e) {
            console.error(`Error with ${modelName}:`, e.message);
        }
    }
}

test();
