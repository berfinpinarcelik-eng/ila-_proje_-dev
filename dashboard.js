// ── Dashboard ──
const DAYS = ['Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar'];
const todayIdx = (new Date().getDay() + 6) % 7;
let selDay = todayIdx;
const chatHistory = [];

function buildDashboard() {
  document.getElementById('sidebarGrade').textContent = state.grade;
  const now = new Date();
  const mon = new Date(now); mon.setDate(now.getDate() - todayIdx);
  const sun = new Date(mon); sun.setDate(mon.getDate() + 6);
  const fmt = d => d.toLocaleDateString('tr-TR', { day:'numeric', month:'long' });
  const todayLessons = state.plan.weeklyPlan[DAYS[todayIdx]] || [];
  const first = todayLessons[0];

  document.getElementById('mainContent').innerHTML = `
  <div style="display:flex;flex-direction:column;gap:22px;max-width:860px;margin:0 auto;">

    <div class="fade-up" style="animation-delay:.0s">
      <div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">${state.grade}</div>
      <h1 style="font-size:26px;font-weight:900;letter-spacing:-.5px;line-height:1.2;">${state.plan.headline}</h1>
      <p style="color:var(--muted);margin-top:8px;font-size:14px;">${state.plan.subline}</p>
    </div>

    ${first ? `
    <div class="glass-card fade-up" style="padding:18px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;animation-delay:.06s;">
      <div style="width:46px;height:46px;border-radius:14px;background:${first.color}18;border:1px solid ${first.color}44;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <i data-lucide="${first.icon}" size="20" style="color:${first.color};"></i>
      </div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;">Bugünkü İlk Ders</div>
        <div style="font-size:15px;font-weight:800;">${first.topic}</div>
        <div style="font-size:12px;color:${first.color};margin-top:3px;">${first.subject} · ${first.duration}</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button class="btn-primary" style="padding:10px 16px;font-size:13px;" onclick="showDetailPlan(${JSON.stringify(first).replace(/"/g,'&quot;')},'${DAYS[todayIdx]}')">
          📋 Detaylı Plan
        </button>
        <button class="btn-primary" style="padding:10px 20px;font-size:13px;background:linear-gradient(135deg,#1e3a5f,#2d5016);" onclick="startLesson('${first.subject}','${first.topic}')">Derse Başla →</button>
      </div>
    </div>` : `
    <div class="glass-card fade-up" style="padding:18px 22px;font-size:14px;color:var(--muted);animation-delay:.06s;">Bugün dinlenme günün 🌙 Yarın hazır ol!</div>`}

    <div class="glass-card fade-up" style="padding:22px;animation-delay:.12s;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:18px;">
        <div>
          <div style="font-size:16px;font-weight:800;">Haftalık Plan</div>
          <div style="font-size:13px;color:var(--muted);margin-top:3px;">${fmt(mon)} – ${fmt(sun)}</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;" id="dayTabs"></div>
      </div>
      <div id="dayLessons" style="display:flex;flex-direction:column;gap:9px;"></div>
    </div>

    <div class="glass-card fade-up" style="padding:22px;animation-delay:.18s;">
      <div style="font-size:16px;font-weight:800;margin-bottom:14px;">💡 AI Koçundan İpuçları</div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        ${(state.plan.tips||[]).map((t,i) => `
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="width:24px;height:24px;border-radius:8px;background:rgba(91,127,255,.12);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:12px;font-weight:800;color:#818cf8;">${i+1}</div>
          <div style="font-size:14px;color:var(--muted);padding-top:3px;">${t}</div>
        </div>`).join('')}
      </div>
    </div>

    <div class="glass-card fade-up" id="chatPanel" style="overflow:hidden;animation-delay:.22s;">
      <div style="padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.02);">
        <div style="width:32px;height:32px;border-radius:10px;background:linear-gradient(135deg,#5b7fff,#c084fc);display:flex;align-items:center;justify-content:center;">
          <i data-lucide="sparkles" size="14"></i>
        </div>
        <div>
          <div style="font-size:14px;font-weight:700;">AI Koçluk Asistanı</div>
          <div style="font-size:11px;color:var(--muted);">Sorularını yanıtlıyorum</div>
        </div>
        <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
          <div style="width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);"></div>
          <span style="font-size:11px;color:var(--green);font-weight:600;">Çevrimiçi</span>
        </div>
      </div>
      <div style="height:300px;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px;" id="chatMsgs" class="main-scroll"></div>
      <div style="padding:12px 18px;border-top:1px solid var(--border);display:flex;gap:8px;">
        <input class="chat-input" id="chatInput" placeholder="Herhangi bir konuyu sor..." autocomplete="off">
        <button class="send-btn" id="sendBtn">Gönder</button>
      </div>
    </div>

  </div>`;

  lucide.createIcons();
  renderDayTabs();
  renderLessons();

  const tl = (state.plan.weeklyPlan[DAYS[todayIdx]]||[]).map(l => `"${l.topic}"`).join(', ');
  addChatMsg(`Merhaba! 👋 ${state.grade} planın hazır. Bugün ${tl||'dinlenme günün'} var. Herhangi bir konuda soru sorabilir ya da ders satırlarına tıklayarak detaylı günlük plan alabilirsin!`, false);

  document.getElementById('sendBtn').addEventListener('click', sendChat);
  document.getElementById('chatInput').addEventListener('keydown', e => { if (e.key==='Enter') sendChat(); });
}

function renderDayTabs() {
  const cont = document.getElementById('dayTabs'); if (!cont) return;
  cont.innerHTML = '';
  DAYS.forEach((d, i) => {
    const btn = document.createElement('button');
    btn.className = 'day-tab' + (i===selDay?' sel':'') + (i===todayIdx&&i!==selDay?' tod':'');
    btn.textContent = d.slice(0, 3);
    btn.onclick = () => { selDay=i; renderDayTabs(); renderLessons(); };
    cont.appendChild(btn);
  });
}

function renderLessons() {
  const cont = document.getElementById('dayLessons'); if (!cont) return;
  cont.innerHTML = '';
  const lessons = state.plan.weeklyPlan[DAYS[selDay]] || [];
  const isToday = selDay === todayIdx;
  const dayName = DAYS[selDay];

  if (isToday) {
    const b = document.createElement('div');
    b.style.cssText = 'font-size:12px;color:#818cf8;font-weight:700;margin-bottom:2px;';
    b.textContent = '📅 Bugün · ' + DAYS[selDay];
    cont.appendChild(b);
  }

  if (!lessons.length) {
    cont.innerHTML = '<div style="font-size:14px;color:var(--muted);padding:12px 0;">Bu gün için ders planlanmamış 🌙</div>';
    return;
  }

  lessons.forEach(l => {
    const row = document.createElement('div');
    row.className = 'lesson-row';
    row.title = 'Detaylı planı gör';
    row.onmouseenter = () => row.style.borderColor = l.color + '55';
    row.onmouseleave = () => row.style.borderColor = 'var(--border)';
    row.onclick = () => showDetailPlan(l, dayName);
    row.innerHTML = `
      <div class="lesson-icon" style="background:${l.color}14;border:1px solid ${l.color}33;">
        <i data-lucide="${l.icon||'book'}" size="16" style="color:${l.color};"></i>
      </div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:11px;color:${l.color};font-weight:700;text-transform:uppercase;letter-spacing:.06em;">${l.subject}</div>
        <div style="font-size:14px;font-weight:700;margin-top:2px;">${l.topic}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:3px;">📋 Detaylı plan için tıkla</div>
      </div>
      <div style="text-align:right;flex-shrink:0;display:flex;flex-direction:column;align-items:flex-end;gap:6px;">
        <div style="font-size:12px;color:var(--muted);">${l.duration}</div>
        ${isToday ? `<button onclick="event.stopPropagation();startLesson('${l.subject}','${l.topic}')" style="font-size:11px;font-weight:700;background:rgba(91,127,255,.14);color:#818cf8;border:1px solid rgba(91,127,255,.3);border-radius:8px;padding:3px 10px;cursor:pointer;font-family:inherit;">Başla</button>` : ''}
      </div>`;
    cont.appendChild(row);
  });
  lucide.createIcons();
}

function startLesson(subject, topic) {
  scrollToChat();
  setTimeout(() => {
    addChatMsg(`${subject} dersinden <strong>"${topic}"</strong> konusunu öğrenmeye başlayalım! 🚀 Önce temel kavramları açıklayayım...`, false);
    chatHistory.push({ role: 'assistant', content: `${subject} - ${topic} konusunu çalışmaya başlıyoruz.` });
  }, 300);
}

function scrollToChat() {
  document.getElementById('chatPanel')?.scrollIntoView({ behavior: 'smooth' });
}

function addChatMsg(html, isUser) {
  const msgs = document.getElementById('chatMsgs'); if (!msgs) return;
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;justify-content:' + (isUser?'flex-end':'flex-start') + ';';
  const bubble = document.createElement('div');
  bubble.className = isUser ? 'chat-user' : 'chat-ai';
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const text = input?.value?.trim(); if (!text) return;
  input.value = '';
  sendBtn.disabled = true;
  addChatMsg(text, true);
  chatHistory.push({ role: 'user', content: text });

  const msgs = document.getElementById('chatMsgs');
  const tw = document.createElement('div');
  tw.id = 'typing'; tw.style.cssText = 'display:flex;justify-content:flex-start;';
  const tb = document.createElement('div'); tb.className = 'chat-ai';
  tb.innerHTML = '<span class="blink">●</span> <span class="blink" style="animation-delay:.25s">●</span> <span class="blink" style="animation-delay:.5s">●</span>';
  tw.appendChild(tb); msgs.appendChild(tw); msgs.scrollTop = msgs.scrollHeight;

  const system = `Sen bir Türkçe AI eğitim koçusun. Öğrenci: ${state.grade}, dersler: ${state.subjects.join(', ')}, hedef: ${state.goal}. Kısa, net, teşvik edici yanıtlar ver. Türkçe konuş. Öğrenciye adını bilmiyorsan "sen" diye hitap et.`;

  try {
    const reply = await callClaude(chatHistory, system);
    document.getElementById('typing')?.remove();
    chatHistory.push({ role: 'assistant', content: reply });
    addChatMsg(reply, false);
  } catch (e) {
    document.getElementById('typing')?.remove();
    addChatMsg(`⚠️ Şu an yanıt alamadım. Lütfen birkaç saniye bekleyip tekrar dene.`, false);
    console.warn('Chat hatası:', e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}
