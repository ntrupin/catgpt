const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const resetBtn = document.getElementById("reset");
const moodPill = document.getElementById("mood-pill");
const catFace = document.getElementById("cat-face");
const reasoningChip = document.getElementById("reasoning-chip");
const intentChip = document.getElementById("intent-chip");
const ttcChip = document.getElementById("ttc-chip");
const sendBtn = form.querySelector('button[type="submit"]');
const MODE = window.CATGPT_MODE || "reasoning";
const MODE_LABEL = MODE === "reasoning" ? "REASONING" : "INSTANT";

const FACES = {
  PLAYFUL: "=^.^=",
  HUNGRY: "=^o^=",
  SLEEPY: "=-.-= z",
  GRUMPY: "=^>.<^=",
};
const THINKING_ACTIONS_BY_MOOD = {
  PLAYFUL: [
    "plotting a pounce...",
    "running zoomies simulation...",
    "calculating leap trajectory...",
    "comparing cardboard box options...",
    "reviewing bird-watching notes...",
    "deciding between cuddle and chaos...",
    "performing paw-based analysis...",
  ],
  HUNGRY: [
    "evaluating snack routes...",
    "listening for can-opener frequencies...",
    "detecting treat rustle patterns...",
    "tracking kitchen activity...",
    "optimizing meow timing for food...",
    "mapping the shortest path to kibble...",
    "staring directly at the food cabinet...",
  ],
  SLEEPY: [
    "checking nap probability...",
    "triangulating blanket warmth...",
    "testing loaf posture hypotheses...",
    "slow-blinking strategy...",
    "calculating ideal curl radius...",
    "kneading through options...",
    "searching for premium sunbeam...",
  ],
  GRUMPY: [
    "judging from afar...",
    "preparing stealth mode...",
    "monitoring hallway activity...",
    "tracking suspicious dust particles...",
    "consulting whisker telemetry...",
    "planning dramatic tail flick...",
    "pretending not to care...",
  ],
  DEFAULT: [
    "scheming by window...",
    "investigating invisible things...",
    "staring into the distance...",
    "aligning ears with cosmic signals...",
  ],
};

function addBubble(role, text, mood = null, action = null, consensus = null) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;
  wrap.textContent = text;

  if (role === "cat") {
    const bits = [];
    if (mood) bits.push(`mood: ${mood.toLowerCase()}`);
    if (MODE === "reasoning" && action) bits.push(`intent: ${action}`);
    if (MODE === "reasoning" && consensus) bits.push(`ttc: ${consensus}`);
    if (bits.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = bits.join(" · ");
      wrap.appendChild(meta);
    }
  }

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function setMood(mood) {
  moodPill.dataset.mood = mood;
  moodPill.textContent = mood;
  moodPill.classList.remove("bump");
  requestAnimationFrame(() => moodPill.classList.add("bump"));
  catFace.textContent = FACES[mood] || FACES.PLAYFUL;
}

function setIntent(action) {
  if (!intentChip) return;
  intentChip.textContent = (action || "observing").toLowerCase();
}

function randomThinkingAction(mood = "PLAYFUL") {
  const pool = THINKING_ACTIONS_BY_MOOD[mood] || THINKING_ACTIONS_BY_MOOD.DEFAULT;
  return pool[Math.floor(Math.random() * pool.length)];
}

function setReasoning(isThinking) {
  if (!reasoningChip) {
    input.disabled = false;
    sendBtn.disabled = false;
    resetBtn.disabled = false;
    return;
  }

  if (MODE !== "reasoning") {
    reasoningChip.textContent = "instant";
    reasoningChip.classList.remove("thinking");
    reasoningChip.classList.add("idle");
    input.disabled = false;
    sendBtn.disabled = false;
    resetBtn.disabled = false;
    return;
  }

  if (isThinking) {
    reasoningChip.textContent = "reasoning";
  } else {
    reasoningChip.textContent = "idle";
  }

  reasoningChip.classList.toggle("thinking", isThinking);
  reasoningChip.classList.toggle("idle", !isThinking);
  input.disabled = isThinking;
  sendBtn.disabled = isThinking;
  resetBtn.disabled = isThinking;
}

function startThinkingBubble(bubble, mood) {
  bubble.textContent = randomThinkingAction(mood);
}

function setTTC() {
  ttcChip.textContent = MODE_LABEL;
}

async function sendMessage(message) {
  let typing = null;
  if (MODE === "reasoning") {
    typing = addBubble("cat", "...");
    startThinkingBubble(typing, moodPill.dataset.mood || "PLAYFUL");
    setReasoning(true);
  }

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    if (typing) typing.remove();
    if (!res.ok) {
      setIntent("spooked");
      addBubble("cat", "hiss?", "GRUMPY", "spooked");
      return;
    }

    setMood(data.mood);
    setIntent(data.action);
    setTTC();
    const ttcMeta =
      MODE === "reasoning" && data.consensus != null && data.samples != null
        ? `${data.consensus}/${data.samples}`
        : null;
    addBubble("cat", data.reply, data.mood, data.action, ttcMeta);
  } catch {
    if (typing) typing.remove();
    setIntent("confused");
    addBubble("cat", "mrrp... (network hiccup)", "PLAYFUL", "confused");
  } finally {
    if (MODE === "reasoning") setReasoning(false);
    input.focus();
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addBubble("user", message);
  input.value = "";
  input.focus();
  await sendMessage(message);
});

resetBtn.addEventListener("click", async () => {
  const res = await fetch("/api/reset", { method: "POST" });
  const data = await res.json();
  setMood(data.mood || "PLAYFUL");
  setIntent("observing");
  setTTC();
  addBubble("cat", "*tail flick* reset.", data.mood || "PLAYFUL", "observing");
});

setMood(window.CATGPT_INIT_MOOD || "PLAYFUL");
setIntent("observing");
setReasoning(false);
setTTC();
addBubble("cat", "mrrp! say hi.", moodPill.dataset.mood, "observing");
