const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const resetBtn = document.getElementById("reset");
const moodPill = document.getElementById("mood-pill");
const catFace = document.getElementById("cat-face");
const reasoningChip = document.getElementById("reasoning-chip");
const intentChip = document.getElementById("intent-chip");
const ttcChip = document.getElementById("ttc-chip");
const workspace = document.querySelector(".workspace");
const rolloutGallery = document.getElementById("rollout-gallery");
const rolloutSummary = document.getElementById("rollout-summary");
const rolloutToggle = document.getElementById("rollout-toggle");
const rolloutPanel = document.getElementById("rollout-panel");
const sendBtn = form.querySelector('button[type="submit"]');
const MODE = window.CATGPT_MODE || "reasoning";
const MODE_LABEL = MODE === "reasoning" ? "reasoning" : "instant";

const FACES = {
  PLAYFUL: "=^.^=",
  HUNGRY: "=^o^=",
  SLEEPY: "=-.-= z",
  DREAMING: "=-.-= zz",
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
  DREAMING: [
    "chasing spectral moths...",
    "floating through blanket fog...",
    "dreaming of impossible sunbeams...",
    "padding softly through moonlight...",
    "hearing a distant can opener...",
    "napping inside another nap...",
    "tracking phantom whisker signals...",
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

function formatToken(value) {
  return String(value || "").replaceAll("_", " ");
}

function planSummary(plan) {
  if (!Array.isArray(plan) || !plan.length) return null;
  return plan.map(formatToken).join(" > ");
}

function driveSummary(state = {}) {
  const order = [
    ["h", "hunger"],
    ["e", "energy"],
    ["t", "trust"],
    ["m", "mischief"],
  ];
  const bits = order.filter(([, key]) => state[key]).map(([short, key]) => `${short}=${formatToken(state[key])}`);
  return bits.length ? `drives: ${bits.join(" ")}` : null;
}

function worldSummary(state = {}) {
  const order = ["bowl", "toy", "vacuum", "sunbeam", "box"];
  const bits = order.filter((key) => state[key]).slice(0, 2).map((key) => `${key}=${formatToken(state[key])}`);
  return bits.length ? `world: ${bits.join(" ")}` : null;
}

function dreamSummary(state = {}) {
  if (!state?.dream || state.dream === "awake") return null;
  return `dream: ${formatToken(state.dream)}`;
}

function truncate(text, max = 96) {
  if (!text || text.length <= max) return text;
  return `${text.slice(0, max - 1)}...`;
}

function setRolloutSummary(text) {
  if (!rolloutSummary) return;
  rolloutSummary.textContent = text;
}

function clearRolloutGallery(message = "The gallery fills in after a reasoning turn.", summary = "waiting") {
  if (!rolloutGallery) return;
  rolloutGallery.replaceChildren();
  const empty = document.createElement("p");
  empty.className = "rollout-empty";
  empty.textContent = message;
  rolloutGallery.appendChild(empty);
  setRolloutSummary(summary);
}

function renderRolloutGallery(gallery = [], consensus = null, samples = null) {
  if (!rolloutGallery) return;
  rolloutGallery.replaceChildren();

  if (!Array.isArray(gallery) || !gallery.length) {
    clearRolloutGallery();
    return;
  }

  const summary =
    consensus != null && samples != null ? `winner ${consensus}/${samples}` : `${gallery.length} samples`;
  setRolloutSummary(summary);

  gallery.forEach((item) => {
    const card = document.createElement("article");
    card.className = `rollout-card${item.winner ? " winner" : ""}`;

    const top = document.createElement("div");
    top.className = "rollout-topline";

    const badges = document.createElement("div");
    badges.className = "rollout-badges";

    if (item.winner) {
      const badge = document.createElement("span");
      badge.className = "rollout-badge winner";
      badge.textContent = "winner";
      badges.appendChild(badge);
    }

    const moodBadge = document.createElement("span");
    moodBadge.className = "rollout-badge";
    moodBadge.textContent = formatToken(item.mood).toLowerCase();
    badges.appendChild(moodBadge);

    if (item.state?.dream && item.state.dream !== "awake") {
      const dreamBadge = document.createElement("span");
      dreamBadge.className = "rollout-badge dream";
      dreamBadge.textContent = `dream ${formatToken(item.state.dream)}`;
      badges.appendChild(dreamBadge);
    }

    const agreeBadge = document.createElement("span");
    agreeBadge.className = "rollout-badge";
    agreeBadge.textContent = `agree ${item.agreement ?? 1}`;
    badges.appendChild(agreeBadge);

    const score = document.createElement("div");
    score.className = "rollout-score";
    score.textContent = `score ${Number(item.score || 0).toFixed(2)}`;

    top.appendChild(badges);
    top.appendChild(score);
    card.appendChild(top);

    const reply = document.createElement("div");
    reply.className = "rollout-reply";
    reply.textContent = item.reply || "mrrp";
    card.appendChild(reply);

    const meta = document.createElement("div");
    meta.className = "rollout-meta";
    const metaBits = [];
    if (item.action) metaBits.push(`intent: ${formatToken(item.action)}`);
    if (item.state?.room) metaBits.push(`room: ${formatToken(item.state.room)}`);
    if (item.state?.focus) metaBits.push(`focus: ${formatToken(item.state.focus)}`);
    const dream = dreamSummary(item.state || {});
    if (dream) metaBits.push(dream);
    meta.textContent = metaBits.join(" · ");
    card.appendChild(meta);

    const plan = document.createElement("div");
    plan.className = "rollout-plan";
    plan.textContent = `plan: ${planSummary(item.plan) || "watch > blink > improvise"}`;
    card.appendChild(plan);

    const stateLine = document.createElement("div");
    stateLine.className = "rollout-think";
    const drives = driveSummary(item.state || {});
    const world = worldSummary(item.state || {});
    const stateBits = [drives, world].filter(Boolean);
    stateLine.textContent = truncate(stateBits.join(" · ") || "state: unreadable", 110);
    card.appendChild(stateLine);

    rolloutGallery.appendChild(card);
  });
}

function addBubble(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;
  wrap.textContent = text;

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function setMood(mood, state = {}) {
  const isDreaming = state?.dream && state.dream !== "awake";
  const displayMood = isDreaming ? "DREAMING" : mood;
  moodPill.dataset.mood = displayMood;
  moodPill.textContent = formatToken(displayMood).toLowerCase();
  moodPill.classList.remove("bump");
  requestAnimationFrame(() => moodPill.classList.add("bump"));
  catFace.textContent = FACES[displayMood] || FACES[mood] || FACES.PLAYFUL;
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

function setTTC(consensus = null, samples = null) {
  if (!ttcChip) return;
  ttcChip.textContent =
    MODE === "reasoning" && consensus != null && samples != null ? `winner ${consensus}/${samples}` : MODE_LABEL;
}

function setRolloutCollapsed(collapsed) {
  if (!workspace || !rolloutPanel || !rolloutToggle) return;
  workspace.classList.toggle("gallery-collapsed", collapsed);
  rolloutPanel.setAttribute("aria-hidden", String(collapsed));
  rolloutToggle.textContent = collapsed ? "show rollouts" : "hide rollouts";
  rolloutToggle.setAttribute("aria-expanded", String(!collapsed));
}

async function sendMessage(message) {
  let typing = null;
  if (MODE === "reasoning") {
    typing = addBubble("cat", "...");
    startThinkingBubble(typing, moodPill.dataset.mood || "PLAYFUL");
    setReasoning(true);
    clearRolloutGallery("Sampling possible cat brains...", "sampling...");
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
      addBubble("cat", "hiss?");
      return;
    }

    setMood(data.mood || "PLAYFUL", data.state);
    setIntent(data.action);
    setTTC(data.consensus, data.samples);
    addBubble("cat", data.reply);
    renderRolloutGallery(data.gallery, data.consensus, data.samples);
  } catch {
    if (typing) typing.remove();
    setIntent("confused");
    addBubble("cat", "mrrp... (network hiccup)");
    clearRolloutGallery("The rollout gallery is empty after that hiccup.", "idle");
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
  addBubble("cat", "*tail flick* reset.");
  clearRolloutGallery();
});

if (rolloutToggle) {
  rolloutToggle.addEventListener("click", () => {
    const collapsed = !workspace?.classList.contains("gallery-collapsed");
    setRolloutCollapsed(collapsed);
  });
}

setMood(window.CATGPT_INIT_MOOD || "PLAYFUL");
setIntent("observing");
setReasoning(false);
setTTC();
setRolloutCollapsed(true);
addBubble("cat", "mrrp! say hi.");
clearRolloutGallery();
