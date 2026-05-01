// ── Detailed Daily Plan Modal ──
async function showDetailPlan(lesson, dayName) {
  // Show modal with loading state
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.id = 'detailModal';
  overlay.innerHTML = `
    <div class="modal-box" style="padding:32px;display:flex;flex-direction:column;align-items:center;gap:16px;">
      <div class="spinner" style="width:40px;height:40px;"></div>
      <div style="font-size:14px;color:var(--muted);font-weight:600;">Detaylı plan hazırlanıyor...</div>
      <div style="font-size:12px;color:var(--muted);">AI ${lesson.subject} - ${lesson.topic} için çalışma planın oluşturuluyor ✨</div>
    </div>`;
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);

  const prompt = `Sen uzman bir Türkçe eğitim koçusun. Aşağıdaki ders için SADECE JSON formatında çok detaylı günlük çalışma planı oluştur:

Öğrenci Bilgileri:
- Sınıf: ${state.grade}
- Ders: ${lesson.subject}
- Konu: ${lesson.topic}
- Toplam Süre: ${lesson.duration}
- Gün: ${dayName}
- Genel Hedef: ${state.goal}

JSON formatı (başka hiçbir şey yazma):
{
  "title": "Konu başlığı",
  "summary": "Konunun 2-3 cümlelik özeti ve neden önemli olduğu",
  "totalDuration": "toplam dakika",
  "difficulty": "Kolay|Orta|Zor",
  "prerequisites": ["ön bilgi 1", "ön bilgi 2"],
  "timeline": [
    {
      "time": "09:00 - 09:15",
      "phase": "Isınma",
      "activity": "Ne yapılacağı detaylı açıklaması",
      "method": "Çalışma yöntemi (örn: zihin haritası, okuma, soru çözme)",
      "tip": "Bu aşama için özel ipucu"
    }
  ],
  "keyPoints": ["öğrenilecek temel nokta 1", "temel nokta 2", "temel nokta 3", "temel nokta 4", "temel nokta 5"],
  "resources": [
    {"type": "Kitap|Video|Alıştırma|Web", "name": "kaynak adı", "detail": "nasıl kullanılacağı"}
  ],
  "practiceQuestions": ["soru 1", "soru 2", "soru 3"],
  "commonMistakes": ["yaygın hata 1", "yaygın hata 2"],
  "successCriteria": "Bu konuyu öğrendiğini nasıl anlarsın - 1-2 cümle",
  "nextTopic": "Bir sonraki çalışılması önerilen konu"
}

Kurallar:
- timeline'da en az 5-7 aşama olsun (ısınma, ana çalışma blokları, mola, tekrar, değerlendirme)
- Her aşama gerçekçi zaman dilimleri içersin
- ${state.grade} seviyesine uygun, Türkçe içerik
- Çok somut ve uygulanabilir tavsiyeler ver
- practiceQuestions gerçek sınav sorusu formatında olsun`;

  try {
    const raw = await callClaude([{ role: 'user', content: prompt }], null, 2000);
    const detail = JSON.parse(raw.replace(/```json|```/g, '').trim());
    renderDetailModal(detail, lesson);
  } catch (e) {
    console.warn('Detay planı hatası:', e.message);
    renderDetailModalError(lesson);
  }
}

function renderDetailModal(d, lesson) {
  const overlay = document.getElementById('detailModal');
  if (!overlay) return;

  const diffColor = d.difficulty === 'Kolay' ? '#4ade80' : d.difficulty === 'Orta' ? '#f59e0b' : '#f472b6';

  overlay.innerHTML = `
    <div class="modal-box">
      <!-- Header -->
      <div style="padding:24px 28px;border-bottom:1px solid var(--border);background:rgba(91,127,255,.04);border-radius:24px 24px 0 0;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;">
          <div style="flex:1;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
              <span class="tag">${lesson.subject}</span>
              <span class="tag" style="background:${diffColor}18;color:${diffColor};border-color:${diffColor}33;">${d.difficulty || 'Orta'}</span>
              <span class="tag purple">⏱ ${d.totalDuration}</span>
            </div>
            <h2 style="font-size:20px;font-weight:900;line-height:1.3;">${d.title}</h2>
            <p style="font-size:13px;color:var(--muted);margin-top:8px;line-height:1.6;">${d.summary}</p>
          </div>
          <button onclick="document.getElementById('detailModal').remove()" style="background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:10px;padding:7px 12px;color:var(--muted);cursor:pointer;font-family:inherit;font-size:13px;font-weight:600;white-space:nowrap;">✕ Kapat</button>
        </div>
      </div>

      <div style="padding:24px 28px;display:flex;flex-direction:column;gap:20px;">

        <!-- Prerequisites -->
        ${d.prerequisites && d.prerequisites.length ? `
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📌 Ön Koşul Bilgiler</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">
            ${d.prerequisites.map(p => `<span class="tag amber">✓ ${p}</span>`).join('')}
          </div>
        </div>` : ''}

        <!-- Timeline -->
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;">⏰ Saatlik Çalışma Planı</div>
          <div style="position:relative;">
            ${(d.timeline||[]).map((t, i) => `
            <div class="timeline-item">
              <div style="display:flex;flex-direction:column;align-items:center;gap:0;padding-top:3px;">
                <div class="timeline-dot" style="color:var(--accent);background:var(--accent);"></div>
                ${i < (d.timeline.length-1) ? `<div style="width:1px;flex:1;background:rgba(91,127,255,.15);margin-top:4px;min-height:30px;"></div>` : ''}
              </div>
              <div style="flex:1;padding-bottom:${i < (d.timeline.length-1) ? '8px' : '0'};">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:5px;">
                  <span style="font-size:11px;font-weight:800;color:var(--accent);font-family:'JetBrains Mono',monospace;">${t.time}</span>
                  <span style="font-size:12px;font-weight:700;color:var(--text);">${t.phase}</span>
                  ${t.method ? `<span class="tag" style="font-size:10px;">${t.method}</span>` : ''}
                </div>
                <div style="font-size:13.5px;color:var(--text);line-height:1.6;margin-bottom:${t.tip?'6px':'0'};">${t.activity}</div>
                ${t.tip ? `<div style="font-size:12px;color:#818cf8;background:rgba(91,127,255,.08);border-left:2px solid var(--accent);padding:6px 10px;border-radius:0 8px 8px 0;margin-top:6px;">💡 ${t.tip}</div>` : ''}
              </div>
            </div>`).join('')}
          </div>
        </div>

        <!-- Key Points -->
        ${d.keyPoints && d.keyPoints.length ? `
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">🎯 Temel Öğrenme Noktaları</div>
          <div style="display:flex;flex-direction:column;gap:7px;">
            ${d.keyPoints.map((k,i) => `
            <div class="detail-section" style="display:flex;gap:10px;align-items:flex-start;">
              <div style="width:22px;height:22px;border-radius:7px;background:rgba(91,127,255,.15);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;color:#818cf8;flex-shrink:0;">${i+1}</div>
              <div style="font-size:13.5px;padding-top:2px;">${k}</div>
            </div>`).join('')}
          </div>
        </div>` : ''}

        <!-- Practice Questions -->
        ${d.practiceQuestions && d.practiceQuestions.length ? `
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📝 Pratik Sorular</div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${d.practiceQuestions.map((q,i) => `
            <div class="detail-section" style="display:flex;gap:10px;align-items:flex-start;">
              <span style="font-size:11px;font-weight:800;color:#f59e0b;margin-top:2px;flex-shrink:0;">S${i+1}</span>
              <span style="font-size:13.5px;">${q}</span>
            </div>`).join('')}
          </div>
        </div>` : ''}

        <!-- Common Mistakes -->
        ${d.commonMistakes && d.commonMistakes.length ? `
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">⚠️ Dikkat Edilecek Hatalar</div>
          <div style="display:flex;flex-direction:column;gap:7px;">
            ${d.commonMistakes.map(m => `
            <div style="display:flex;gap:8px;background:rgba(244,114,182,.06);border:1px solid rgba(244,114,182,.15);border-radius:12px;padding:10px 14px;">
              <span style="color:#f472b6;flex-shrink:0;">✗</span>
              <span style="font-size:13.5px;">${m}</span>
            </div>`).join('')}
          </div>
        </div>` : ''}

        <!-- Resources -->
        ${d.resources && d.resources.length ? `
        <div>
          <div style="font-size:13px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">📚 Önerilen Kaynaklar</div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${d.resources.map(r => `
            <div class="detail-section" style="display:flex;gap:12px;align-items:flex-start;">
              <span class="tag ${r.type==='Video'?'purple':r.type==='Alıştırma'?'green':r.type==='Web'?'amber':''}" style="font-size:10px;flex-shrink:0;">${r.type}</span>
              <div>
                <div style="font-size:13.5px;font-weight:700;margin-bottom:2px;">${r.name}</div>
                <div style="font-size:12px;color:var(--muted);">${r.detail}</div>
              </div>
            </div>`).join('')}
          </div>
        </div>` : ''}

        <!-- Success Criteria & Next Topic -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;flex-wrap:wrap;">
          ${d.successCriteria ? `
          <div style="background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.18);border-radius:14px;padding:14px 16px;">
            <div style="font-size:12px;font-weight:700;color:#4ade80;margin-bottom:6px;">✅ Başarı Kriteri</div>
            <div style="font-size:13px;line-height:1.5;">${d.successCriteria}</div>
          </div>` : ''}
          ${d.nextTopic ? `
          <div style="background:rgba(129,140,248,.07);border:1px solid rgba(129,140,248,.18);border-radius:14px;padding:14px 16px;">
            <div style="font-size:12px;font-weight:700;color:#818cf8;margin-bottom:6px;">➡️ Sonraki Konu</div>
            <div style="font-size:13px;line-height:1.5;">${d.nextTopic}</div>
          </div>` : ''}
        </div>

        <!-- Start Chat CTA -->
        <button class="btn-primary" style="width:100%;" onclick="document.getElementById('detailModal').remove(); startLesson('${lesson.subject}','${lesson.topic}')">
          <i data-lucide="sparkles" size="16"></i> AI Koç ile Bu Konuyu Çalış
        </button>

      </div>
    </div>`;

  lucide.createIcons();
}

function renderDetailModalError(lesson) {
  const overlay = document.getElementById('detailModal');
  if (!overlay) return;
  overlay.innerHTML = `
    <div class="modal-box" style="padding:40px;text-align:center;">
      <div style="font-size:40px;margin-bottom:16px;">🔌</div>
      <div style="font-size:18px;font-weight:800;margin-bottom:10px;">Detay planı yüklenemedi</div>
      <div style="font-size:14px;color:var(--muted);margin-bottom:24px;line-height:1.6;">Bağlantı sorunu yaşandı. Yine de AI Koç ile bu konuyu doğrudan çalışabilirsin.</div>
      <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
        <button class="btn-primary" onclick="document.getElementById('detailModal').remove(); startLesson('${lesson.subject}','${lesson.topic}')">
          <i data-lucide="sparkles" size="16"></i> AI Koç ile Çalış
        </button>
        <button onclick="document.getElementById('detailModal').remove()" style="background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:14px;padding:14px 24px;font-size:15px;font-weight:700;cursor:pointer;font-family:inherit;color:var(--muted);">Kapat</button>
      </div>
    </div>`;
  lucide.createIcons();
}
