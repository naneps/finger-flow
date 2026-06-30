import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './styles.css';

const video = document.querySelector('#camera');
const canvas = document.querySelector('#scene');
const ctx = canvas.getContext('2d');
const startButton = document.querySelector('#startButton');
const startPanel = document.querySelector('#startPanel');
const handStatus = document.querySelector('#handStatus');
const particleCount = document.querySelector('#particleCount');

const CONFIG = {
  maxParticles: 950,
  trailSpawn: 5,
  burstSpawn: 90,
  pinchThreshold: 0.055,
  pinchCooldownMs: 320,
  handSmoothness: 0.28,
};

let handLandmarker;
let cameraReady = false;
let tracking = false;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let lastPinchTime = 0;
let particles = [];
let handPoint = null;
let smoothPoint = null;
let previousPoint = null;
let pinchActive = false;
let hueBase = 190;

function setStatus(message) {
  handStatus.textContent = message;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function mirroredX(normalizedX) {
  return (1 - normalizedX) * canvas.clientWidth;
}

function toCanvasPoint(landmark) {
  return {
    x: mirroredX(landmark.x),
    y: landmark.y * canvas.clientHeight,
    z: landmark.z ?? 0,
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpPoint(a, b, t) {
  if (!a) return { ...b };
  return {
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
    z: lerp(a.z ?? 0, b.z ?? 0, t),
  };
}

function spawnTrail(point, speed) {
  const intensity = Math.min(2.8, 0.7 + speed * 0.018);
  const amount = Math.floor(CONFIG.trailSpawn * intensity);
  hueBase = (hueBase + 0.7 + speed * 0.02) % 360;

  for (let i = 0; i < amount; i += 1) {
    const angle = randomBetween(0, Math.PI * 2);
    const drift = randomBetween(0.2, 2.8 + speed * 0.018);
    const size = randomBetween(4, 14) * Math.min(1.8, intensity);

    particles.push({
      x: point.x + randomBetween(-8, 8),
      y: point.y + randomBetween(-8, 8),
      vx: Math.cos(angle) * drift + (previousPoint ? (point.x - previousPoint.x) * 0.018 : 0),
      vy: Math.sin(angle) * drift + (previousPoint ? (point.y - previousPoint.y) * 0.018 : 0),
      size,
      life: randomBetween(0.58, 1),
      decay: randomBetween(0.008, 0.018),
      hue: (hueBase + randomBetween(-34, 54)) % 360,
      spin: randomBetween(-0.035, 0.035),
      angle,
    });
  }
}

function spawnBurst(point) {
  hueBase = (hueBase + 46) % 360;

  for (let i = 0; i < CONFIG.burstSpawn; i += 1) {
    const angle = (i / CONFIG.burstSpawn) * Math.PI * 2 + randomBetween(-0.17, 0.17);
    const velocity = randomBetween(2.4, 11.5);
    const warmHue = i % 3 === 0 ? 318 : i % 3 === 1 ? 190 : 52;

    particles.push({
      x: point.x,
      y: point.y,
      vx: Math.cos(angle) * velocity,
      vy: Math.sin(angle) * velocity,
      size: randomBetween(5, 22),
      life: randomBetween(0.72, 1),
      decay: randomBetween(0.01, 0.022),
      hue: warmHue + randomBetween(-18, 18),
      spin: randomBetween(-0.06, 0.06),
      angle,
    });
  }
}

function drawGlowCircle(x, y, radius, hue, alpha) {
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, `hsla(${hue}, 100%, 72%, ${alpha})`);
  gradient.addColorStop(0.4, `hsla(${hue + 20}, 100%, 58%, ${alpha * 0.38})`);
  gradient.addColorStop(1, `hsla(${hue}, 100%, 48%, 0)`);

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function updateAndDrawParticles(delta) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  const target = smoothPoint;

  for (const particle of particles) {
    if (target) {
      const dx = target.x - particle.x;
      const dy = target.y - particle.y;
      const pull = pinchActive ? 0.0018 : 0.00055;
      particle.vx += dx * pull;
      particle.vy += dy * pull;
    }

    particle.angle += particle.spin * delta;
    particle.vx *= 0.986;
    particle.vy *= 0.986;
    particle.vy += 0.008 * delta;
    particle.x += particle.vx * delta;
    particle.y += particle.vy * delta;
    particle.life -= particle.decay * delta;
    particle.size *= 0.993;

    const alpha = Math.max(0, particle.life);
    const radius = particle.size * (1.6 + alpha * 1.7);

    drawGlowCircle(particle.x, particle.y, radius, particle.hue, alpha * 0.42);

    ctx.fillStyle = `hsla(${particle.hue}, 100%, 82%, ${alpha})`;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, Math.max(0.7, particle.size * 0.24), 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  particles = particles.filter((particle) => particle.life > 0.02 && particle.size > 0.6);

  if (particles.length > CONFIG.maxParticles) {
    particles.splice(0, particles.length - CONFIG.maxParticles);
  }

  particleCount.textContent = `${particles.length} particles`;
}

function drawFingerCursor(point) {
  if (!point) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  drawGlowCircle(point.x, point.y, pinchActive ? 62 : 42, pinchActive ? 315 : hueBase, 0.54);

  ctx.strokeStyle = pinchActive ? 'rgba(255, 160, 232, 0.92)' : 'rgba(127, 240, 255, 0.86)';
  ctx.lineWidth = pinchActive ? 3 : 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, pinchActive ? 20 : 13, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = 'rgba(255, 255, 255, 0.96)';
  ctx.beginPath();
  ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawIdleMist(delta) {
  const time = performance.now() * 0.0001;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  for (let i = 0; i < 4; i += 1) {
    const x = canvas.clientWidth * (0.18 + i * 0.22 + Math.sin(time * 8 + i) * 0.04);
    const y = canvas.clientHeight * (0.28 + Math.cos(time * 7 + i * 1.8) * 0.08);
    drawGlowCircle(x, y, 90 + Math.sin(time * 16 + i) * 22, 188 + i * 38, 0.024 * delta);
  }

  ctx.restore();
}

function readGesture(results) {
  const firstHand = results.landmarks?.[0];

  if (!firstHand) {
    handPoint = null;
    pinchActive = false;
    setStatus(cameraReady ? 'Show your hand' : 'Camera idle');
    return;
  }

  const indexTip = firstHand[8];
  const thumbTip = firstHand[4];
  const indexPoint = toCanvasPoint(indexTip);
  const thumbPoint = toCanvasPoint(thumbTip);
  const normalizedPinchDistance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);

  handPoint = indexPoint;
  smoothPoint = lerpPoint(smoothPoint, handPoint, CONFIG.handSmoothness);
  pinchActive = normalizedPinchDistance < CONFIG.pinchThreshold;

  const now = performance.now();
  if (pinchActive && now - lastPinchTime > CONFIG.pinchCooldownMs) {
    const centerPoint = {
      x: (indexPoint.x + thumbPoint.x) / 2,
      y: (indexPoint.y + thumbPoint.y) / 2,
      z: 0,
    };
    spawnBurst(centerPoint);
    lastPinchTime = now;
  }

  setStatus(pinchActive ? 'Pinch burst!' : 'Tracking finger');
}

async function setupHandLandmarker() {
  setStatus('Loading model...');

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
}

async function setupCamera() {
  setStatus('Opening camera...');

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });

  await video.play();
  cameraReady = true;
}

function predictHands() {
  if (!tracking || !handLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, performance.now());
    readGesture(results);
  }
}

function render() {
  const now = performance.now();
  const delta = Math.min(2.2, (now - lastFrameTime) / 16.67);
  lastFrameTime = now;

  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawIdleMist(delta);

  predictHands();

  if (smoothPoint) {
    const speed = previousPoint ? distance(smoothPoint, previousPoint) : 0;
    spawnTrail(smoothPoint, speed);
    previousPoint = { ...smoothPoint };
  } else {
    previousPoint = null;
  }

  updateAndDrawParticles(delta);
  drawFingerCursor(smoothPoint);

  requestAnimationFrame(render);
}

async function start() {
  try {
    startButton.disabled = true;
    startButton.textContent = 'Starting...';

    await setupHandLandmarker();
    await setupCamera();

    tracking = true;
    startPanel.classList.add('is-hidden');
    setStatus('Show your hand');
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = 'Try again';
    setStatus('Camera/model failed');
    alert('Failed to start camera tracking. Make sure camera permission is allowed and run this on localhost or HTTPS.');
  }
}

window.addEventListener('resize', resizeCanvas);
startButton.addEventListener('click', start);

resizeCanvas();
render();
