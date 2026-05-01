async function callClaude(messages, systemPrompt = null, maxTokens = 1000) {
  try {
    const body = {
      messages: messages,
      system: systemPrompt,
      max_tokens: maxTokens
    };

    // Sunucun 3000 portunda bekliyor
    const res = await fetch('/api/claude', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || 'Sunucu hatası: ' + res.status);
    }

    const data = await res.json();

    // Sunucudan gelen formatı doğru okuyalım
    if (data && data.content && data.content[0]) {
      return data.content[0].text;
    } else {
      throw new Error("Geçersiz veri formatı");
    }

  } catch (error) {
    console.error("Bağlantı Hatası:", error);
    throw error;
  }
}
