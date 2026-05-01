// ── Global State ──
const state = { grade:'', subjects:[], goal:'', days:'5', plan:null };

// ── Chip setup ──
function setupChips(id, multi, key, btnId) {
  document.getElementById(id).querySelectorAll('.chip').forEach(c => {
    c.addEventListener('click', () => {
      if (multi) {
        c.classList.toggle('selected');
        state[key] = [...document.getElementById(id).querySelectorAll('.chip.selected')].map(x=>x.dataset.val);
      } else {
        document.getElementById(id).querySelectorAll('.chip').forEach(x=>x.classList.remove('selected'));
        c.classList.add('selected');
        state[key] = c.dataset.val;
      }
      if (btnId) document.getElementById(btnId).disabled = multi ? state[key].length===0 : !state[key];
    });
  });
}

function goStep(n) {
  document.querySelectorAll('.step').forEach(s=>s.classList.remove('active'));
  document.getElementById('step'+n).classList.add('active');
  lucide.createIcons();
}

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  lucide.createIcons();
}

// Init chips on load
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  setupChips('gradeChips', false, 'grade', 'btn1');
  setupChips('subjectChips', true, 'subjects', 'btn2');
  setupChips('daysChips', false, 'days', null);
});
