const axios = require('axios');

async function testGroq() {
    try {
        console.log("Testing server with Groq...");
        const res = await axios.post('http://localhost:3000/api/claude', {
            messages: [{ role: 'user', content: 'Merhaba, nasılsın?' }]
        });
        console.log("Response:", res.data.content[0].text);
    } catch (e) {
        console.error("Error:", e.response ? e.response.data : e.message);
    }
}

testGroq();
