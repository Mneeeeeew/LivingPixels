const canvas = document.querySelector("#canvas");
const context = canvas.getContext("2d");
const targetCanvas = document.querySelector("#targetCanvas");
const targetContext = targetCanvas.getContext("2d");
context.imageSmoothingEnabled = false;
targetContext.imageSmoothingEnabled = false;

const status = document.querySelector("#status");
const stepLabel = document.querySelector("#step");
const fpsLabel = document.querySelector("#fps");
const trainButton = document.querySelector("#train");
const pauseButton = document.querySelector("#pause");
const targetSize = 32;
let model;
let state;
let targetPixels = new Uint8ClampedArray(targetSize * targetSize * 4);
let running = true;
let stepCount = 0;
let frameCount = 0;
let fpsStamp = performance.now();

const index = (channel, y, x) => channel * model.size * model.size + y * model.size + x;

function value(channel, y, x) {
  return x < 0 || x >= model.size || y < 0 || y >= model.size ? 0 : state[index(channel, y, x)];
}

function reset() {
  state = new Float32Array(model.channels * model.size * model.size);
  const center = Math.floor(model.size / 2);
  state[index(3, center, center)] = 1;
  stepCount = 0;
}

function step() {
  const size = model.size;
  const channels = model.channels;
  const features = new Float32Array(channels * 3);
  const hidden = new Float32Array(model.hiddenChannels);
  const next = state.slice();

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let alive = false;
      for (let oy = -1; oy <= 1 && !alive; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (value(3, y + oy, x + ox) > model.aliveThreshold) alive = true;
        }
      }
      if (!alive || Math.random() >= model.updateProbability) continue;

      for (let channel = 0; channel < channels; channel += 1) {
        features[channel] = value(channel, y, x);
        features[channels + channel] = (-value(channel, y - 1, x - 1) + value(channel, y - 1, x + 1) - 2 * value(channel, y, x - 1) + 2 * value(channel, y, x + 1) - value(channel, y + 1, x - 1) + value(channel, y + 1, x + 1)) / 8;
        features[2 * channels + channel] = (-value(channel, y - 1, x - 1) - 2 * value(channel, y - 1, x) - value(channel, y - 1, x + 1) + value(channel, y + 1, x - 1) + 2 * value(channel, y + 1, x) + value(channel, y + 1, x + 1)) / 8;
      }
      for (let hiddenIndex = 0; hiddenIndex < model.hiddenChannels; hiddenIndex += 1) {
        let sum = model.b1[hiddenIndex];
        for (let featureIndex = 0; featureIndex < features.length; featureIndex += 1) sum += features[featureIndex] * model.w1[featureIndex * model.hiddenChannels + hiddenIndex];
        hidden[hiddenIndex] = Math.max(0, sum);
      }
      for (let channel = 0; channel < channels; channel += 1) {
        let delta = model.b2[channel];
        for (let hiddenIndex = 0; hiddenIndex < model.hiddenChannels; hiddenIndex += 1) delta += hidden[hiddenIndex] * model.w2[hiddenIndex * channels + channel];
        next[index(channel, y, x)] += delta;
      }
    }
  }
  state = next;
  stepCount += 1;
}

function draw() {
  const pixels = context.createImageData(model.size, model.size);
  const plane = model.size * model.size;
  for (let y = 0; y < model.size; y += 1) {
    for (let x = 0; x < model.size; x += 1) {
      const source = index(0, y, x);
      const destination = (y * model.size + x) * 4;
      for (let channel = 0; channel < 4; channel += 1) pixels.data[destination + channel] = Math.max(0, Math.min(255, state[source + channel * plane] * 255));
    }
  }
  context.putImageData(pixels, 0, 0);
  stepLabel.textContent = `step=${stepCount}`;
}

function drawTarget() {
  targetContext.putImageData(new ImageData(targetPixels, targetSize, targetSize), 0, 0);
}

function targetPoint(event) {
  const rect = targetCanvas.getBoundingClientRect();
  return {
    x: Math.floor((event.clientX - rect.left) / rect.width * targetSize),
    y: Math.floor((event.clientY - rect.top) / rect.height * targetSize),
  };
}

function paintTarget(event) {
  const point = targetPoint(event);
  const radius = Math.max(1, Math.min(4, Number(document.querySelector("#brushSize").value) || 2));
  const eraser = document.querySelector("#eraser").checked;
  const color = document.querySelector("#brushColor").value;
  const rgb = eraser ? [0, 0, 0] : [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
  for (let oy = -radius; oy <= radius; oy += 1) {
    for (let ox = -radius; ox <= radius; ox += 1) {
      const x = point.x + ox;
      const y = point.y + oy;
      if (ox * ox + oy * oy > radius * radius || x < 0 || x >= targetSize || y < 0 || y >= targetSize) continue;
      const offset = (y * targetSize + x) * 4;
      targetPixels[offset] = rgb[0];
      targetPixels[offset + 1] = rgb[1];
      targetPixels[offset + 2] = rgb[2];
      targetPixels[offset + 3] = eraser ? 0 : 255;
    }
  }
  drawTarget();
}

function clearTarget() {
  targetPixels = new Uint8ClampedArray(targetSize * targetSize * 4);
  drawTarget();
}

function loadPreset() {
  const image = new Image();
  image.onload = () => {
    targetContext.clearRect(0, 0, targetSize, targetSize);
    targetContext.drawImage(image, 0, 0, targetSize, targetSize);
    targetPixels = targetContext.getImageData(0, 0, targetSize, targetSize).data;
  };
  image.src = "/assets/target.png";
}

function damage(fraction, random = false) {
  const size = model.size;
  const radius = Math.max(1, Math.floor(Math.sqrt(fraction * size * size / Math.PI)));
  const centerX = random ? Math.floor(Math.random() * size) : Math.floor(size / 2);
  const centerY = random ? Math.floor(Math.random() * size) : Math.floor(size / 2);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (random ? Math.random() < fraction : (x - centerX) ** 2 + (y - centerY) ** 2 <= radius ** 2) {
        for (let channel = 0; channel < model.channels; channel += 1) state[index(channel, y, x)] = 0;
      }
    }
  }
}

function eraseOrganism(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * model.size);
  const y = Math.floor((event.clientY - rect.top) / rect.height * model.size);
  for (let oy = -2; oy <= 2; oy += 1) for (let ox = -2; ox <= 2; ox += 1) {
    if (ox * ox + oy * oy > 4 || x + ox < 0 || x + ox >= model.size || y + oy < 0 || y + oy >= model.size) continue;
    for (let channel = 0; channel < model.channels; channel += 1) state[index(channel, y + oy, x + ox)] = 0;
  }
}

async function trainTarget() {
  const alpha = targetPixels.filter((_, index) => index % 4 === 3).some(value => value > 0);
  if (!alpha) { status.textContent = "Draw at least one target pixel first."; return; }
  trainButton.disabled = true;
  running = false;
  status.textContent = "Training the drawn target locally…";
  try {
    const response = await fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ size: targetSize, target: Array.from(targetPixels, value => value / 255), iterations: Number(document.querySelector("#iterations").value) || 300, steps: 32, poolSize: 64, batchSize: 4 }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Training failed");
    model = result.model;
    reset();
    running = true;
    status.textContent = `Trained custom target · loss=${result.loss.toFixed(5)} · iterations=${result.iterations}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    trainButton.disabled = false;
  }
}

function frame(now) {
  if (running) { step(); step(); }
  draw();
  frameCount += 1;
  if (now - fpsStamp > 1000) { fpsLabel.textContent = `fps=${frameCount}`; frameCount = 0; fpsStamp = now; }
  requestAnimationFrame(frame);
}

document.querySelector("#reset").addEventListener("click", () => { reset(); draw(); });
pauseButton.addEventListener("click", () => { running = !running; pauseButton.textContent = running ? "Pause" : "Play"; });
document.querySelectorAll("[data-damage]").forEach(button => button.addEventListener("click", () => damage(Number(button.dataset.damage))));
document.querySelector("#random").addEventListener("click", () => damage(0.3, true));
document.querySelector("#clearTarget").addEventListener("click", clearTarget);
document.querySelector("#presetTarget").addEventListener("click", loadPreset);
trainButton.addEventListener("click", trainTarget);

let drawingTarget = false;
targetCanvas.addEventListener("pointerdown", event => { drawingTarget = true; targetCanvas.setPointerCapture(event.pointerId); paintTarget(event); });
targetCanvas.addEventListener("pointermove", event => { if (drawingTarget) paintTarget(event); });
targetCanvas.addEventListener("pointerup", () => { drawingTarget = false; });
targetCanvas.addEventListener("pointercancel", () => { drawingTarget = false; });
let erasingOrganism = false;
canvas.addEventListener("pointerdown", event => { erasingOrganism = true; canvas.setPointerCapture(event.pointerId); eraseOrganism(event); });
canvas.addEventListener("pointermove", event => { if (erasingOrganism) eraseOrganism(event); });
canvas.addEventListener("pointerup", () => { erasingOrganism = false; });
canvas.addEventListener("pointercancel", () => { erasingOrganism = false; });

targetPixels = new Uint8ClampedArray(targetSize * targetSize * 4);
drawTarget();
Promise.all([fetch("/api/model").then(response => response.json()), new Promise(resolve => setTimeout(resolve, 100))]).then(([loaded]) => {
  if (loaded.error) throw new Error(loaded.error);
  model = loaded;
  reset();
  trainButton.disabled = false;
  status.textContent = `Loaded ${model.channels} channels · draw a target and train it`;
  loadPreset();
  requestAnimationFrame(frame);
}).catch(error => { status.textContent = error.message; });

