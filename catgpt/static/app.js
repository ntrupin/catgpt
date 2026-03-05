const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const resetBtn = document.getElementById("reset");
const moodPill = document.getElementById("mood-pill");
const catFace = document.getElementById("cat-face");

const FACES = {
  PLAYFUL: "=^.^=",
  HUNGRY: "=^o^=",
  SLEEPY: "=-.-= z",
  GRUMPY: "=^>.<^=",
};

function addBubble(role, text, mood = null, typing = false) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}` + (typing ? " typing" : "");
  wrap.textContent = text;

  if (role === "cat" && mood) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `mood: ${mood.toLowerCase()}`;
    wrap.appendChild(meta);
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

async function sendMessage(message) {
  const typing = addBubble("cat", "...", null, true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    typing.remove();
    if (!res.ok) {
      addBubble("cat", "hiss?", "GRUMPY");
      return;
    }

    setMood(data.mood);
    addBubble("cat", data.reply, data.mood);
  } catch {
    typing.remove();
    addBubble("cat", "mrrp... (network hiccup)", "PLAYFUL");
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
  setMood(data.mood);
  addBubble("cat", "*ear flick*", data.mood);
});

setMood(window.CATGPT_INIT_MOOD || "PLAYFUL");
addBubble("cat", "mrrp! say hi.", moodPill.dataset.mood);
