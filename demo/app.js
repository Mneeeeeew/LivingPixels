const canvas = document.querySelector("#canvas"), context = canvas.getContext("2d");
context.imageSmoothingEnabled = false;
const status = document.querySelector("#status"), stepLabel = document.querySelector("#step"), fpsLabel = document.querySelector("#fps");
let model, state, running = true, stepCount = 0, frames = 0, fpsStamp = performance.now();
const index = (channel, y, x) => channel * model.size * model.size + y * model.size + x;
function value(channel, y, x) { return x < 0 || x >= model.size || y < 0 || y >= model.size ? 0 : state[index(channel, y, x)]; }
function reset() { state = new Float32Array(model.channels * model.size * model.size); const c = Math.floor(model.size / 2); state[index(3, c, c)] = 1; stepCount = 0; }
function step() {
  const s = model.size, c = model.channels, features = new Float32Array(c * 3), hidden = new Float32Array(model.hiddenChannels), next = state.slice();
  for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
    let alive = false; for (let oy = -1; oy <= 1 && !alive; oy++) for (let ox = -1; ox <= 1; ox++) if (value(3, y + oy, x + ox) > model.aliveThreshold) alive = true;
    if (!alive || Math.random() >= model.updateProbability) continue;
    for (let ch = 0; ch < c; ch++) {
      features[ch] = value(ch, y, x);
      features[c + ch] = (-value(ch, y - 1, x - 1) + value(ch, y - 1, x + 1) - 2 * value(ch, y, x - 1) + 2 * value(ch, y, x + 1) - value(ch, y + 1, x - 1) + value(ch, y + 1, x + 1)) / 8;
      features[2 * c + ch] = (-value(ch, y - 1, x - 1) - 2 * value(ch, y - 1, x) - value(ch, y - 1, x + 1) + value(ch, y + 1, x - 1) + 2 * value(ch, y + 1, x) + value(ch, y + 1, x + 1)) / 8;
    }
    for (let h = 0; h < model.hiddenChannels; h++) { let sum = model.b1[h]; for (let f = 0; f < features.length; f++) sum += features[f] * model.w1[f * model.hiddenChannels + h]; hidden[h] = Math.max(0, sum); }
    for (let ch = 0; ch < c; ch++) { let delta = model.b2[ch]; for (let h = 0; h < model.hiddenChannels; h++) delta += hidden[h] * model.w2[h * c + ch]; next[index(ch, y, x)] += delta; }
  }
  state = next; stepCount++;
}
function draw() {
  const pixels = context.createImageData(model.size, model.size), plane = model.size * model.size;
  for (let y = 0; y < model.size; y++) for (let x = 0; x < model.size; x++) { const source = index(0, y, x), target = (y * model.size + x) * 4; for (let ch = 0; ch < 4; ch++) pixels.data[target + ch] = Math.max(0, Math.min(255, state[source + ch * plane] * 255)); }
  context.putImageData(pixels, 0, 0); stepLabel.textContent = `step=${stepCount}`;
}
function eraseAt(event) { const rect = canvas.getBoundingClientRect(), x = Math.floor((event.clientX - rect.left) / rect.width * model.size), y = Math.floor((event.clientY - rect.top) / rect.height * model.size); for (let oy = -2; oy <= 2; oy++) for (let ox = -2; ox <= 2; ox++) if (ox * ox + oy * oy <= 4 && x + ox >= 0 && x + ox < model.size && y + oy >= 0 && y + oy < model.size) for (let ch = 0; ch < model.channels; ch++) state[index(ch, y + oy, x + ox)] = 0; }
function damage(fraction, random = false) { const s = model.size, radius = Math.max(1, Math.floor(Math.sqrt(fraction * s * s / Math.PI))), cx = random ? Math.floor(Math.random() * s) : Math.floor(s / 2), cy = random ? Math.floor(Math.random() * s) : Math.floor(s / 2); for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) if (random ? Math.random() < fraction : (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) for (let ch = 0; ch < model.channels; ch++) state[index(ch, y, x)] = 0; }
function frame(now) { if (running) { step(); step(); } draw(); frames++; if (now - fpsStamp > 1000) { fpsLabel.textContent = `fps=${frames}`; frames = 0; fpsStamp = now; } requestAnimationFrame(frame); }
document.querySelector("#reset").addEventListener("click", () => { reset(); draw(); });
document.querySelector("#pause").addEventListener("click", e => { running = !running; e.currentTarget.textContent = running ? "Pause" : "Play"; });
document.querySelectorAll("[data-damage]").forEach(button => button.addEventListener("click", () => damage(Number(button.dataset.damage))));
document.querySelector("#random").addEventListener("click", () => damage(0.3, true));
let painting = false; canvas.addEventListener("pointerdown", e => { painting = true; canvas.setPointerCapture(e.pointerId); eraseAt(e); }); canvas.addEventListener("pointermove", e => { if (painting) eraseAt(e); }); canvas.addEventListener("pointerup", () => { painting = false; }); canvas.addEventListener("pointercancel", () => { painting = false; });
fetch("model.json").then(response => { if (!response.ok) throw new Error("model.json not found — train and export a checkpoint first"); return response.json(); }).then(loaded => { model = loaded; reset(); status.textContent = `loaded ${model.channels} channels · ${model.hiddenChannels} hidden units`; requestAnimationFrame(frame); }).catch(error => { status.textContent = error.message; });

