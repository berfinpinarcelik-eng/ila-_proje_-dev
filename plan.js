// ── Generate Plan ──
async function generatePlan() {
  state.goal = document.getElementById('goalInput').value.trim() || 'Derslerimi geliştirmek istiyorum';
  showScreen('screenLoading');

  const msgs = ['Müfredatın analiz ediliyor...','Haftalık plan oluşturuluyor...','Konular sıralanıyor...','Son dokunuşlar yapılıyor...'];
  let mi = 0;
  const lt = setInterval(() => { document.getElementById('loadingText').textContent = msgs[mi++ % msgs.length]; }, 1800);

  const prompt = `Sen bir Türkçe AI eğitim koçusun. Kullanıcı bilgileri:
- Sınıf: ${state.grade}
- Seçilen dersler: ${state.subjects.join(', ')}
- Hedef: ${state.goal}
- Haftada çalışma günü: ${state.days}

SADECE aşağıdaki JSON formatında yanıt ver, başka hiçbir şey yazma, açıklama yapma:
{
  "headline": "kısa motivasyon cümlesi max 8 kelime",
  "subline": "planı açıklayan 1 cümle",
  "weeklyPlan": {
    "Pazartesi": [],
    "Salı": [],
    "Çarşamba": [],
    "Perşembe": [],
    "Cuma": [],
    "Cumartesi": [],
    "Pazar": []
  },
  "tips": ["ipucu1","ipucu2","ipucu3"]
}

Her günün dizisindeki ders objesi formatı:
{ "subject":"DersAdı", "topic":"KonuBaşlığı", "duration":"X dk", "icon":"iconAdı", "color":"#hexrenk" }

Kurallar:
- Sadece kullanıcının seçtiği dersleri ekle: ${state.subjects.join(', ')}
- Tam olarak ${state.days} güne dersleri yay, diğer günler boş dizi []
- Konular ${state.grade} seviyesine uygun Türkçe olsun
- icon için şunlardan birini kullan: calculator, book, flask, globe, code-2, brain, pencil, bar-chart-2, atom, languages
- Her ders için farklı hex renk kullan`;

  try {
    const raw = await callClaude([{ role: 'user', content: prompt }]);
    state.plan = JSON.parse(raw.replace(/```json|```/g, '').trim());
  } catch (e) {
    console.warn('API plan hatası, fallback kullanılıyor:', e.message);
    state.plan = fallbackPlan();
  }

  clearInterval(lt);
  buildDashboard();
  showScreen('screenDash');
}

function fallbackPlan() {
  const DAYS = ['Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar'];
  const colors = ['#818cf8','#f59e0b','#34d399','#f472b6','#60a5fa','#fb923c','#a78bfa'];
  const icons = ['book','calculator','flask','globe','code-2','brain'];
  const wp = {}; let d = 0;
  DAYS.forEach((day, i) => {
    if (d < parseInt(state.days) && i < 5) {
      wp[day] = state.subjects.map((s, j) => ({
        subject: s, topic: s + ' - Temel Konular', duration: '45 dk',
        icon: icons[j % icons.length], color: colors[j % colors.length]
      }));
      d++;
    } else { wp[day] = []; }
  });
  return {
    headline: 'Hedefe giden yol açık! 🚀',
    subline: 'Seçtiğin derslerle haftalık planın hazır.',
    weeklyPlan: wp,
    tips: ['Her gün düzenli çalış.', 'Kısa molalar ver.', 'Notlarını tekrar et.']
  };
}
