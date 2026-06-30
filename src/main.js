import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './styles.css';

const video = document.querySelector('#camera');
const canvas = document.querySelector('#scene');
const ctx = canvas.getContext('2d');

const startButton = document.querySelector('#startButton');
const startPanel = document.querySelector('#startPanel');
const handStatus = document.querySelector('#handStatus');
const particleCount = document.querySelector('#particleCount');
const modeTitle = document.querySelector('#modeTitle');
const modeDescription = document.querySelector('#modeDescription');
const modeButtons = [...document.querySelectorAll('[data-mode]')];

const cameraOpacity = document.querySelector('#cameraOpacity');
const particleIntensity = document.querySelector('#particleIntensity');
const glowStrength = document.querySelector('#glowStrength');
const trailLife = document.querySelector('#trailLife');
const smoothing = document.querySelector('#smoothing');
const showSkeleton = document.querySelector('#showSkeleton');
const mirrorCamera = document.querySelector('#mirrorCamera');
const clearButton = document.querySelector('#clearButton');

const cameraOpacityValue = document.querySelector('#cameraOpacityValue');
const particleIntensityValue = document.querySelector('#particleIntensityValue');
const glowStrengthValue = document.querySelector('#glowStrengthValue');
const trailLifeValue = document.querySelector('#trailLifeValue');
const smoothingValue = document.querySelector('#smoothingValue');

const FINGER_TIPS = [4, 8, 12, 16, 20];
const INDEX_TIP = 8;
const THUMB_TIP = 4;
const PALM = 0;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const MODE_META = {
  liquid: {
    title: 'Liquid Flow',
    description: 'Index finger paints soft liquid trails. Pinch to burst.',
  },
  allFingers: {
    title: 'All Fingers',
    description: 'Every fingertip becomes a brush, so both hands can paint together.',
  },
  skeleton3d: {
    title: '3D Hand',
    description: 'Landmarks react to depth, making the hand feel like a glowing 3D rig.',
  },
  ribbons: {
    title: 'Liquid Ribbons',
    description: 'Finger movement creates smooth ribbon strokes that melt into particles.',
  },
  constellation: {
    title: 'Constellation',
    description: 'All hand landmarks connect into a living star map.',
  },
  gravity: {
    title: 'Gravity Orbs',
    description: 'Particles orbit and get pulled by your fingertips like little planets.',
  },
};

const settings = {
  mode: 'liquid',
  cameraOpacity: 0.42,
  intensity: 1.25,
  glow: 1.2,
  trailLife: 1.1,
  smoothing: 0.28,
  showSkeleton: true,
  mirror: true,
};

const CONFIG = {
  maxParticles: 1500,
  baseTrailSpawn: 5,
  burstSpawn: 105,
  pinchThreshold: 0.055,
  pinchCooldownMs: 280,
};

let handLandmarker;
let cameraReady = false;
let tracking = false;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let lastPinchTime = 0;
let particles = [];
let ribbons = [];
let detectedHands = [];
let smoothLandmarkMap = new Map();
let previousLandmarkMap = new Map();
let pinchActive = false;
let hueBase = 188;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

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
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function mirroredX(normalizedX) {
  return settings.mirror ? (1 - normalizedX) * canvas.clientWidth : normalizedX * canvas.clientWidth;
}

function toCanvasPoint(landmark) {
  const z = landmark.z ?? 0;
  return {
    x: mirroredX(landmark.x),
    y: landmark.y * canvas.clientHeight,
    z,
    depth: clamp(1 - z * 4.2, 0.48, 2.35),
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
    depth: lerp(a.depth ?? 1, b.depth ?? 1, t),
  };
}

function getPreviousPoint(key) {
  return previousLandmarkMap.get(key) ?? null;
}

function savePreviousPoint(key, point) {
  previousLandmarkMap.set(key, { ...point });
}

function getPrimaryIndexPoint() {
  return detectedHands[0]?.points?.[INDEX_TIP] ?? null;
}

function getAttractors() {
  if (!detectedHands.length) return [];

  if (settings.mode === 'gravity') {
    return detectedHands.flatMap((hand) => FINGER_TIPS.map((tip) => hand.points[tip]).filter(Boolean));
  }

  if (settings.mode === 'allFingers' || settings.mode === 'ribbons') {
    return detectedHands.flatMap((hand) => FINGER_TIPS.map((tip) => hand.points[tip]).filter(Boolean));
  }

  return [getPrimaryIndexPoint()].filter(Boolean);
}

function spawnParticle(point, options = {}) {
  const {
    amount = 1,
    speed = 0,
    spread = 9,
    size = [4, 15],
    velocity = [0.2, 3.2],
    hue = hueBase,
    directional = null,
    life = [0.58, 1],
  } = options;

  const scaledAmount = Math.max(1, Math.floor(amount * settings.intensity));

  for (let i = 0; i < scaledAmount; i += 1) {
    const angle = directional ?? randomBetween(0, Math.PI * 2);
    const drift = randomBetween(velocity[0], velocity[1] + speed * 0.018);
    const radius = randomBetween(size[0], size[1]) * clamp(settings.glow, 0.5, 1.8);

    particles.push({
      x: point.x + randomBetween(-spread, spread),
      y: point.y + randomBetween(-spread, spread),
      vx: Math.cos(angle) * drift + randomBetween(-0.4, 0.4),
      vy: Math.sin(angle) * drift + randomBetween(-0.4, 0.4),
      size: radius,
      life: randomBetween(life[0], life[1]),
      decay: randomBetween(0.008, 0.018) / settings.trailLife,
      hue: (hue + randomBetween(-34, 56)) % 360,
      spin: randomBetween(-0.035, 0.035),
      angle,
    });
  }
}

function spawnTrail(point, speed, hue = hueBase) {
  const intensity = Math.min(3.3, 0.7 + speed * 0.018);
  const amount = CONFIG.baseTrailSpawn * intensity;
  hueBase = (hueBase + 0.7 + speed * 0.018) % 360;

  spawnParticle(point, {
    amount,
    speed,
    hue,
    spread: 7 + speed * 0.015,
    size: [4, 14],
    velocity: [0.1, 2.7],
  });
}

function spawnBurst(point) {
  hueBase = (hueBase + 46) % 360;
  const amount = Math.floor(CONFIG.burstSpawn * settings.intensity);

  for (let i = 0; i < amount; i += 1) {
    const angle = (i / amount) * Math.PI * 2 + randomBetween(-0.18, 0.18);
    const velocity = randomBetween(2.6, 12.5);
    const warmHue = i % 3 === 0 ? 318 : i % 3 === 1 ? 190 : 52;

    particles.push({
      x: point.x,
      y: point.y,
      vx: Math.cos(angle) * velocity,
      vy: Math.sin(angle) * velocity,
      size: randomBetween(5, 24) * settings.glow,
      life: randomBetween(0.72, 1),
      decay: randomBetween(0.01, 0.022) / settings.trailLife,
      hue: warmHue + randomBetween(-18, 18),
      spin: randomBetween(-0.06, 0.06),
      angle,
    });
  }
}

function addRibbonSegment(from, to, hue, width = 18) {
  if (!from || !to) return;
  const speed = distance(from, to);
  if (speed < 0.5) return;

  ribbons.push({
    x1: from.x,
    y1: from.y,
    x2: to.x,
    y2: to.y,
    cx: (from.x + to.x) / 2 + randomBetween(-18, 18),
    cy: (from.y + to.y) / 2 + randomBetween(-18, 18),
    hue,
    width: clamp(width + speed * 0.045, 8, 34),
    life: 1,
    decay: randomBetween(0.018, 0.032) / settings.trailLife,
  });

  if (ribbons.length > 280) {
    ribbons.splice(0, ribbons.length - 280);
  }
}

function drawGlowCircle(x, y, radius, hue, alpha) {
  const finalRadius = radius * settings.glow;
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, finalRadius);
  gradient.addColorStop(0, `hsla(${hue}, 100%, 75%, ${alpha})`);
  gradient.addColorStop(0.38, `hsla(${hue + 20}, 100%, 58%, ${alpha * 0.38})`);
  gradient.addColorStop(1, `hsla(${hue}, 100%, 48%, 0)`);

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, finalRadius, 0, Math.PI * 2);
  ctx.fill();
}

function drawIdleMist(delta) {
  const time = performance.now() * 0.0001;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  for (let i = 0; i < 5; i += 1) {
    const x = canvas.clientWidth * (0.14 + i * 0.19 + Math.sin(time * 8 + i) * 0.035);
    const y = canvas.clientHeight * (0.25 + Math.cos(time * 7 + i * 1.8) * 0.075);
    drawGlowCircle(x, y, 88 + Math.sin(time * 16 + i) * 24, 188 + i * 38, 0.022 * delta);
  }

  ctx.restore();
}

function drawRibbons(delta) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';

  for (const ribbon of ribbons) {
    ribbon.life -= ribbon.decay * delta;
    const alpha = clamp(ribbon.life, 0, 1);

    ctx.lineWidth = ribbon.width * alpha;
    ctx.strokeStyle = `hsla(${ribbon.hue}, 100%, 70%, ${alpha * 0.34})`;
    ctx.shadowBlur = 22 * settings.glow;
    ctx.shadowColor = `hsla(${ribbon.hue}, 100%, 65%, ${alpha})`;
    ctx.beginPath();
    ctx.moveTo(ribbon.x1, ribbon.y1);
    ctx.quadraticCurveTo(ribbon.cx, ribbon.cy, ribbon.x2, ribbon.y2);
    ctx.stroke();
  }

  ctx.restore();
  ribbons = ribbons.filter((ribbon) => ribbon.life > 0.025);
}

function updateAndDrawParticles(delta) {
  const attractors = getAttractors();

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';

  for (const particle of particles) {
    if (attractors.length) {
      let nearest = attractors[0];
      let nearestDistance = distance(particle, nearest);

      for (let i = 1; i < attractors.length; i += 1) {
        const candidateDistance = distance(particle, attractors[i]);
        if (candidateDistance < nearestDistance) {
          nearest = attractors[i];
          nearestDistance = candidateDistance;
        }
      }

      const dx = nearest.x - particle.x;
      const dy = nearest.y - particle.y;
      const pull = settings.mode === 'gravity' ? 0.0024 : pinchActive ? 0.0018 : 0.00055;
      particle.vx += dx * pull;
      particle.vy += dy * pull;

      if (settings.mode === 'gravity') {
        particle.vx += -dy * 0.0007;
        particle.vy += dx * 0.0007;
      }
    }

    particle.angle += particle.spin * delta;
    particle.vx *= settings.mode === 'gravity' ? 0.992 : 0.986;
    particle.vy *= settings.mode === 'gravity' ? 0.992 : 0.986;
    particle.vy += (settings.mode === 'gravity' ? -0.001 : 0.008) * delta;
    particle.x += particle.vx * delta;
    particle.y += particle.vy * delta;
    particle.life -= particle.decay * delta;
    particle.size *= 0.993;

    const alpha = Math.max(0, particle.life);
    const radius = particle.size * (1.6 + alpha * 1.6);

    drawGlowCircle(particle.x, particle.y, radius, particle.hue, alpha * 0.42);

    ctx.shadowBlur = 12 * settings.glow;
    ctx.shadowColor = `hsla(${particle.hue}, 100%, 74%, ${alpha})`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 84%, ${alpha})`;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, Math.max(0.7, particle.size * 0.23), 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  particles = particles.filter((particle) => particle.life > 0.02 && particle.size > 0.55);

  const maxParticles = Math.floor(CONFIG.maxParticles * clamp(settings.intensity, 0.5, 1.8));
  if (particles.length > maxParticles) {
    particles.splice(0, particles.length - maxParticles);
  }

  particleCount.textContent = `${particles.length}`;
}

function drawFingerCursor(point) {
  if (!point) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  drawGlowCircle(point.x, point.y, pinchActive ? 66 : 44, pinchActive ? 315 : hueBase, 0.56);

  ctx.strokeStyle = pinchActive ? 'rgba(255, 160, 232, 0.95)' : 'rgba(127, 240, 255, 0.9)';
  ctx.lineWidth = pinchActive ? 3 : 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, pinchActive ? 21 : 13, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = 'rgba(255, 255, 255, 0.96)';
  ctx.beginPath();
  ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawHandSkeleton(hand, options = {}) {
  const { strong = false, constellation = false } = options;
  const points = hand.points;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [start, end] of HAND_CONNECTIONS) {
    const a = points[start];
    const b = points[end];
    const avgDepth = ((a.depth ?? 1) + (b.depth ?? 1)) / 2;
    ctx.lineWidth = (strong ? 3.4 : 1.6) * avgDepth;
    ctx.strokeStyle = `hsla(${hueBase + start * 5}, 100%, 72%, ${strong ? 0.42 : 0.24})`;
    ctx.shadowBlur = strong ? 18 : 10;
    ctx.shadowColor = `hsla(${hueBase + start * 5}, 100%, 65%, 0.72)`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  if (constellation) {
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const a = points[i];
        const b = points[j];
        const d = distance(a, b);
        if (d > 78) continue;
        const alpha = (1 - d / 78) * 0.17;
        ctx.lineWidth = 1;
        ctx.strokeStyle = `hsla(${hueBase + i * 7}, 100%, 76%, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  points.forEach((point, index) => {
    const isTip = FINGER_TIPS.includes(index);
    const radius = (isTip ? 5.8 : 3.2) * point.depth * (strong ? 1.28 : 1);
    const hue = hueBase + index * 9;
    drawGlowCircle(point.x, point.y, radius * 4.1, hue, isTip ? 0.18 : 0.09);
    ctx.fillStyle = `hsla(${hue}, 100%, ${isTip ? 82 : 74}%, ${isTip ? 0.92 : 0.62})`;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function drawLiquidHand(hand) {
  const tips = FINGER_TIPS.map((tip) => hand.points[tip]);
  const palm = hand.points[PALM];

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 7 * settings.glow;
  ctx.strokeStyle = `hsla(${hueBase}, 100%, 70%, 0.16)`;
  ctx.shadowBlur = 22 * settings.glow;
  ctx.shadowColor = `hsla(${hueBase}, 100%, 65%, 0.9)`;

  for (const tip of tips) {
    ctx.beginPath();
    ctx.moveTo(palm.x, palm.y);
    ctx.quadraticCurveTo((palm.x + tip.x) / 2, palm.y - 28, tip.x, tip.y);
    ctx.stroke();
  }

  ctx.restore();
}

function updateModeEffects() {
  for (const hand of detectedHands) {
    const indexPoint = hand.points[INDEX_TIP];
    const indexKey = `${hand.index}-${INDEX_TIP}`;
    const previousIndex = getPreviousPoint(indexKey);
    const indexSpeed = distance(indexPoint, previousIndex);

    if (settings.mode === 'liquid') {
      spawnTrail(indexPoint, indexSpeed, hueBase);
      addRibbonSegment(previousIndex, indexPoint, hueBase, 20);
      drawLiquidHand(hand);
    }

    if (settings.mode === 'allFingers') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const key = `${hand.index}-${tip}`;
        const point = hand.points[tip];
        const prev = getPreviousPoint(key);
        const speed = distance(point, prev);
        spawnTrail(point, speed, hueBase + tipIndex * 32);
      });
    }

    if (settings.mode === 'skeleton3d') {
      drawHandSkeleton(hand, { strong: true });
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const point = hand.points[tip];
        const key = `${hand.index}-${tip}`;
        const speed = distance(point, getPreviousPoint(key));
        spawnParticle(point, {
          amount: 1.6,
          speed,
          hue: 190 + tipIndex * 38,
          spread: 4,
          size: [2.5, 8],
          velocity: [0.1, 1.6],
        });
      });
    }

    if (settings.mode === 'ribbons') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const key = `${hand.index}-${tip}`;
        const point = hand.points[tip];
        const prev = getPreviousPoint(key);
        const speed = distance(point, prev);
        addRibbonSegment(prev, point, hueBase + tipIndex * 34, 22);
        spawnParticle(point, {
          amount: 2.5,
          speed,
          hue: hueBase + tipIndex * 34,
          spread: 5,
          size: [3, 11],
          velocity: [0.1, 2.1],
        });
      });
    }

    if (settings.mode === 'constellation') {
      drawHandSkeleton(hand, { constellation: true });
      hand.points.forEach((point, landmarkIndex) => {
        if (landmarkIndex % 2 !== 0 && !FINGER_TIPS.includes(landmarkIndex)) return;
        spawnParticle(point, {
          amount: 0.9,
          hue: 185 + landmarkIndex * 8,
          spread: 2,
          size: [1.8, 6.4],
          velocity: [0.05, 0.8],
          life: [0.38, 0.76],
        });
      });
    }

    if (settings.mode === 'gravity') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const point = hand.points[tip];
        const key = `${hand.index}-${tip}`;
        const speed = distance(point, getPreviousPoint(key));
        spawnParticle(point, {
          amount: 2.2,
          speed,
          hue: 210 + tipIndex * 28,
          spread: 12,
          size: [3, 13],
          velocity: [0.3, 2.6],
          life: [0.7, 1],
        });
      });
    }
  }
}

function saveCurrentLandmarksAsPrevious() {
  for (const hand of detectedHands) {
    hand.points.forEach((point, landmarkIndex) => {
      savePreviousPoint(`${hand.index}-${landmarkIndex}`, point);
    });
  }
}

function readGesture(results) {
  const rawHands = results.landmarks ?? [];

  if (!rawHands.length) {
    detectedHands = [];
    pinchActive = false;
    setStatus(cameraReady ? 'Show your hand' : 'Camera idle');
    return;
  }

  const nextSmoothMap = new Map();
  detectedHands = rawHands.map((landmarks, handIndex) => {
    const points = landmarks.map((landmark, landmarkIndex) => {
      const key = `${handIndex}-${landmarkIndex}`;
      const rawPoint = toCanvasPoint(landmark);
      const smoothed = lerpPoint(smoothLandmarkMap.get(key), rawPoint, settings.smoothing);
      nextSmoothMap.set(key, smoothed);
      return smoothed;
    });

    return {
      index: handIndex,
      raw: landmarks,
      points,
    };
  });

  smoothLandmarkMap = nextSmoothMap;

  const firstRawHand = rawHands[0];
  const indexTip = firstRawHand[INDEX_TIP];
  const thumbTip = firstRawHand[THUMB_TIP];
  const normalizedPinchDistance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
  pinchActive = normalizedPinchDistance < CONFIG.pinchThreshold;

  const now = performance.now();
  if (pinchActive && now - lastPinchTime > CONFIG.pinchCooldownMs) {
    const indexPoint = detectedHands[0].points[INDEX_TIP];
    const thumbPoint = detectedHands[0].points[THUMB_TIP];
    spawnBurst({
      x: (indexPoint.x + thumbPoint.x) / 2,
      y: (indexPoint.y + thumbPoint.y) / 2,
      z: 0,
      depth: 1,
    });
    lastPinchTime = now;
  }

  const handLabel = rawHands.length > 1 ? `${rawHands.length} hands` : '1 hand';
  setStatus(pinchActive ? 'Pinch burst!' : `${handLabel} tracking`);
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

  if (settings.showSkeleton && settings.mode !== 'skeleton3d' && settings.mode !== 'constellation') {
    detectedHands.forEach((hand) => drawHandSkeleton(hand));
  }

  updateModeEffects();
  drawRibbons(delta);
  updateAndDrawParticles(delta);
  drawFingerCursor(getPrimaryIndexPoint());
  saveCurrentLandmarksAsPrevious();

  requestAnimationFrame(render);
}

function percentage(value) {
  return `${Math.round(value)}%`;
}

function applyCameraStyle() {
  video.style.opacity = `${settings.cameraOpacity}`;
  video.style.transform = settings.mirror ? 'scaleX(-1)' : 'scaleX(1)';
}

function updateSettingLabels() {
  cameraOpacityValue.textContent = percentage(settings.cameraOpacity * 100);
  particleIntensityValue.textContent = percentage(settings.intensity * 100);
  glowStrengthValue.textContent = percentage(settings.glow * 100);
  trailLifeValue.textContent = percentage(settings.trailLife * 100);
  smoothingValue.textContent = percentage(settings.smoothing * 100);
}

function setMode(mode) {
  settings.mode = mode;
  const meta = MODE_META[mode];
  modeTitle.textContent = meta.title;
  modeDescription.textContent = meta.description;

  modeButtons.forEach((button) => {
    button.classList.toggle('is-active', button.dataset.mode === mode);
  });
}

function bindControls() {
  modeButtons.forEach((button) => {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  });

  cameraOpacity.addEventListener('input', () => {
    settings.cameraOpacity = Number(cameraOpacity.value) / 100;
    applyCameraStyle();
    updateSettingLabels();
  });

  particleIntensity.addEventListener('input', () => {
    settings.intensity = Number(particleIntensity.value) / 100;
    updateSettingLabels();
  });

  glowStrength.addEventListener('input', () => {
    settings.glow = Number(glowStrength.value) / 100;
    updateSettingLabels();
  });

  trailLife.addEventListener('input', () => {
    settings.trailLife = Number(trailLife.value) / 100;
    updateSettingLabels();
  });

  smoothing.addEventListener('input', () => {
    settings.smoothing = Number(smoothing.value) / 100;
    updateSettingLabels();
  });

  showSkeleton.addEventListener('change', () => {
    settings.showSkeleton = showSkeleton.checked;
  });

  mirrorCamera.addEventListener('change', () => {
    settings.mirror = mirrorCamera.checked;
    smoothLandmarkMap = new Map();
    previousLandmarkMap = new Map();
    applyCameraStyle();
  });

  clearButton.addEventListener('click', () => {
    particles = [];
    ribbons = [];
    particleCount.textContent = '0';
  });
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

bindControls();
setMode(settings.mode);
applyCameraStyle();
updateSettingLabels();
resizeCanvas();
render();
