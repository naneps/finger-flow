import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './styles.css';

const video = document.querySelector('#camera');
const canvas = document.querySelector('#scene');
const ctx = canvas.getContext('2d', { alpha: true });

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
const lowPower = document.querySelector('#lowPower');
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
    description: 'Index finger paints visible liquid trails. Pinch to burst.',
  },
  allFingers: {
    title: 'All Fingers',
    description: 'Two-hand tracking is enabled. Every fingertip becomes a brush.',
  },
  skeleton3d: {
    title: '3D Hand',
    description: 'Depth-reactive hand rig with optimized glow.',
  },
  ribbons: {
    title: 'Liquid Ribbons',
    description: 'Fingers draw smooth ribbon strokes that fade softly.',
  },
  constellation: {
    title: 'Constellation',
    description: 'Hand landmarks connect into a light star map.',
  },
  gravity: {
    title: 'Gravity Orbs',
    description: 'Particles orbit around your fingertips like small planets.',
  },
};

const settings = {
  mode: 'allFingers',
  cameraOpacity: 0.36,
  intensity: 1,
  glow: 1.05,
  trailLife: 1.05,
  smoothing: 0.24,
  showSkeleton: true,
  mirror: true,
  lowPower: true,
};

const CONFIG = {
  baseTrailSpawn: 3.2,
  burstSpawn: 48,
  pinchThreshold: 0.055,
  pinchCooldownMs: 320,
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
let frameCostEma = 16;
let autoLowPowerNoticeShown = false;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setStatus(message) {
  handStatus.textContent = message;
}

function maxParticles() {
  const base = settings.lowPower ? 520 : 900;
  return Math.floor(base * clamp(settings.intensity, 0.4, 1.45));
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const qualityDpr = settings.lowPower ? 1 : 1.35;
  const dpr = Math.min(window.devicePixelRatio || 1, qualityDpr);

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
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
    depth: clamp(1 - z * 4, 0.6, 1.85),
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

  if (settings.mode === 'gravity' || settings.mode === 'allFingers' || settings.mode === 'ribbons') {
    return detectedHands.flatMap((hand) => FINGER_TIPS.map((tip) => hand.points[tip]).filter(Boolean));
  }

  return [getPrimaryIndexPoint()].filter(Boolean);
}

function spawnParticle(point, options = {}) {
  const {
    amount = 1,
    speed = 0,
    spread = 8,
    size = [3, 10],
    velocity = [0.15, 2.2],
    hue = hueBase,
    life = [0.45, 0.82],
  } = options;

  const perfScale = settings.lowPower ? 0.82 : 1;
  const scaledAmount = Math.max(1, Math.floor(amount * settings.intensity * perfScale));

  for (let i = 0; i < scaledAmount; i += 1) {
    if (particles.length >= maxParticles()) break;

    const angle = randomBetween(0, Math.PI * 2);
    const drift = randomBetween(velocity[0], velocity[1] + speed * 0.012);
    const particleSize = randomBetween(size[0], size[1]) * clamp(settings.glow, 0.45, 1.25);

    particles.push({
      x: point.x + randomBetween(-spread, spread),
      y: point.y + randomBetween(-spread, spread),
      vx: Math.cos(angle) * drift + randomBetween(-0.28, 0.28),
      vy: Math.sin(angle) * drift + randomBetween(-0.28, 0.28),
      size: particleSize,
      life: randomBetween(life[0], life[1]),
      decay: randomBetween(0.01, 0.021) / settings.trailLife,
      hue: (hue + randomBetween(-28, 42)) % 360,
    });
  }
}

function spawnTrail(point, speed, hue = hueBase) {
  const intensity = Math.min(settings.lowPower ? 2.4 : 3.1, 0.75 + speed * 0.016);
  hueBase = (hueBase + 0.55 + speed * 0.012) % 360;

  spawnParticle(point, {
    amount: CONFIG.baseTrailSpawn * intensity,
    speed,
    hue,
    spread: 5 + speed * 0.01,
    size: settings.lowPower ? [3.4, 12] : [4, 15],
    velocity: [0.08, settings.lowPower ? 2.1 : 2.8],
  });
}

function spawnBurst(point) {
  hueBase = (hueBase + 46) % 360;
  const amount = Math.floor(CONFIG.burstSpawn * settings.intensity * (settings.lowPower ? 0.9 : 1.2));

  for (let i = 0; i < amount; i += 1) {
    if (particles.length >= maxParticles()) break;

    const angle = (i / amount) * Math.PI * 2 + randomBetween(-0.18, 0.18);
    const velocity = randomBetween(2.2, settings.lowPower ? 9 : 12);
    const warmHue = i % 3 === 0 ? 318 : i % 3 === 1 ? 190 : 52;

    particles.push({
      x: point.x,
      y: point.y,
      vx: Math.cos(angle) * velocity,
      vy: Math.sin(angle) * velocity,
      size: randomBetween(5, settings.lowPower ? 16 : 24) * settings.glow,
      life: randomBetween(0.62, 0.98),
      decay: randomBetween(0.012, 0.026) / settings.trailLife,
      hue: warmHue + randomBetween(-18, 18),
    });
  }
}

function addRibbonSegment(from, to, hue, width = 16) {
  if (!from || !to) return;
  const speed = distance(from, to);
  if (speed < 0.55) return;

  ribbons.push({
    x1: from.x,
    y1: from.y,
    x2: to.x,
    y2: to.y,
    cx: (from.x + to.x) / 2 + randomBetween(-14, 14),
    cy: (from.y + to.y) / 2 + randomBetween(-14, 14),
    hue,
    width: clamp(width + speed * 0.04, 8, settings.lowPower ? 28 : 34),
    life: 1,
    decay: randomBetween(0.018, 0.032) / settings.trailLife,
  });

  const maxRibbons = settings.lowPower ? 130 : 220;
  if (ribbons.length > maxRibbons) {
    ribbons.splice(0, ribbons.length - maxRibbons);
  }
}

function drawAura(x, y, radius, hue, alpha) {
  if (settings.lowPower) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowBlur = radius * 0.55 * settings.glow;
    ctx.shadowColor = `hsla(${hue}, 100%, 70%, ${alpha})`;
    ctx.fillStyle = `hsla(${hue}, 100%, 70%, ${alpha * 0.55})`;
    ctx.beginPath();
    ctx.arc(x, y, radius * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  const finalRadius = radius * settings.glow;
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, finalRadius);
  gradient.addColorStop(0, `hsla(${hue}, 100%, 75%, ${alpha})`);
  gradient.addColorStop(0.45, `hsla(${hue + 20}, 100%, 58%, ${alpha * 0.32})`);
  gradient.addColorStop(1, `hsla(${hue}, 100%, 48%, 0)`);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, finalRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawIdleMist(delta) {
  if (settings.lowPower) return;

  const time = performance.now() * 0.0001;
  for (let i = 0; i < 3; i += 1) {
    const x = canvas.clientWidth * (0.18 + i * 0.28 + Math.sin(time * 8 + i) * 0.03);
    const y = canvas.clientHeight * (0.28 + Math.cos(time * 7 + i * 1.8) * 0.06);
    drawAura(x, y, 76 + Math.sin(time * 16 + i) * 18, 188 + i * 44, 0.024 * delta);
  }
}

function drawRibbons(delta) {
  if (!ribbons.length) return;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';

  for (const ribbon of ribbons) {
    ribbon.life -= ribbon.decay * delta;
    const alpha = clamp(ribbon.life, 0, 1);

    ctx.lineWidth = ribbon.width * alpha;
    ctx.strokeStyle = `hsla(${ribbon.hue}, 100%, 72%, ${alpha * (settings.lowPower ? 0.48 : 0.42)})`;
    ctx.shadowBlur = settings.lowPower ? 8 * settings.glow : 18 * settings.glow;
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
  ctx.shadowBlur = settings.lowPower ? 7 * settings.glow : 13 * settings.glow;

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
      const pull = settings.mode === 'gravity' ? 0.0018 : pinchActive ? 0.0012 : 0.00042;
      particle.vx += dx * pull;
      particle.vy += dy * pull;

      if (settings.mode === 'gravity') {
        particle.vx += -dy * 0.00055;
        particle.vy += dx * 0.00055;
      }
    }

    particle.vx *= settings.mode === 'gravity' ? 0.992 : 0.984;
    particle.vy *= settings.mode === 'gravity' ? 0.992 : 0.984;
    particle.vy += (settings.mode === 'gravity' ? -0.001 : 0.006) * delta;
    particle.x += particle.vx * delta;
    particle.y += particle.vy * delta;
    particle.life -= particle.decay * delta;
    particle.size *= 0.992;

    const alpha = Math.max(0, particle.life);
    const radius = Math.max(1.2, particle.size * (settings.lowPower ? 0.72 : 0.78));

    ctx.shadowColor = `hsla(${particle.hue}, 100%, 70%, ${alpha})`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 76%, ${alpha * 0.95})`;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  particles = particles.filter((particle) => particle.life > 0.02 && particle.size > 0.55);

  const limit = maxParticles();
  if (particles.length > limit) {
    particles.splice(0, particles.length - limit);
  }

  particleCount.textContent = `${particles.length}`;
}

function drawFingerCursor(point) {
  if (!point) return;

  drawAura(point.x, point.y, pinchActive ? 58 : 42, pinchActive ? 315 : hueBase, 0.55);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.strokeStyle = pinchActive ? 'rgba(255, 160, 232, 0.95)' : 'rgba(127, 240, 255, 0.92)';
  ctx.lineWidth = pinchActive ? 3 : 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, pinchActive ? 18 : 12, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
  ctx.beginPath();
  ctx.arc(point.x, point.y, 3.2, 0, Math.PI * 2);
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
    ctx.lineWidth = (strong ? 2.8 : 1.45) * avgDepth;
    ctx.strokeStyle = `hsla(${hueBase + start * 5}, 100%, 72%, ${strong ? 0.42 : 0.26})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  if (constellation && !settings.lowPower) {
    for (let i = 0; i < points.length; i += 2) {
      for (let j = i + 2; j < points.length; j += 3) {
        const a = points[i];
        const b = points[j];
        const d = distance(a, b);
        if (d > 76) continue;
        const alpha = (1 - d / 76) * 0.13;
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
    const radius = (isTip ? 4.8 : 2.8) * point.depth * (strong ? 1.16 : 1);
    const hue = hueBase + index * 9;
    ctx.fillStyle = `hsla(${hue}, 100%, ${isTip ? 84 : 76}%, ${isTip ? 0.9 : 0.6})`;
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
  ctx.lineWidth = settings.lowPower ? 2.5 : 5 * settings.glow;
  ctx.strokeStyle = `hsla(${hueBase}, 100%, 70%, ${settings.lowPower ? 0.12 : 0.16})`;

  for (const tip of tips) {
    ctx.beginPath();
    ctx.moveTo(palm.x, palm.y);
    ctx.quadraticCurveTo((palm.x + tip.x) / 2, palm.y - 22, tip.x, tip.y);
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
      addRibbonSegment(previousIndex, indexPoint, hueBase, 18);
      drawLiquidHand(hand);
    }

    if (settings.mode === 'allFingers') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const key = `${hand.index}-${tip}`;
        const point = hand.points[tip];
        const prev = getPreviousPoint(key);
        const speed = distance(point, prev);
        const hue = hueBase + tipIndex * 32 + hand.index * 18;
        spawnParticle(point, {
          amount: 2.6,
          speed,
          hue,
          spread: 5,
          size: [3.2, 11],
          velocity: [0.08, 2.1],
        });
        addRibbonSegment(prev, point, hue, 13);
      });
      drawLiquidHand(hand);
    }

    if (settings.mode === 'skeleton3d') {
      drawHandSkeleton(hand, { strong: true });
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const point = hand.points[tip];
        const key = `${hand.index}-${tip}`;
        const speed = distance(point, getPreviousPoint(key));
        spawnParticle(point, {
          amount: 1.2,
          speed,
          hue: 190 + tipIndex * 38,
          spread: 3,
          size: [2.5, 8],
          velocity: [0.05, 1.3],
        });
      });
    }

    if (settings.mode === 'ribbons') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const key = `${hand.index}-${tip}`;
        const point = hand.points[tip];
        const prev = getPreviousPoint(key);
        const speed = distance(point, prev);
        const hue = hueBase + tipIndex * 34 + hand.index * 18;
        addRibbonSegment(prev, point, hue, 18);
        spawnParticle(point, {
          amount: settings.lowPower ? 1.1 : 1.9,
          speed,
          hue,
          spread: 3,
          size: [2.4, 7.8],
          velocity: [0.06, 1.5],
        });
      });
    }

    if (settings.mode === 'constellation') {
      drawHandSkeleton(hand, { constellation: true });
      hand.points.forEach((point, landmarkIndex) => {
        if (settings.lowPower && landmarkIndex % 4 !== 0 && !FINGER_TIPS.includes(landmarkIndex)) return;
        if (!settings.lowPower && landmarkIndex % 2 !== 0 && !FINGER_TIPS.includes(landmarkIndex)) return;
        spawnParticle(point, {
          amount: 0.65,
          hue: 185 + landmarkIndex * 8,
          spread: 1.5,
          size: [1.8, 5.4],
          velocity: [0.04, 0.55],
          life: [0.34, 0.68],
        });
      });
    }

    if (settings.mode === 'gravity') {
      FINGER_TIPS.forEach((tip, tipIndex) => {
        const point = hand.points[tip];
        const key = `${hand.index}-${tip}`;
        const speed = distance(point, getPreviousPoint(key));
        spawnParticle(point, {
          amount: 1.6,
          speed,
          hue: 210 + tipIndex * 28 + hand.index * 18,
          spread: 8,
          size: [2.8, 10],
          velocity: [0.18, 2],
          life: [0.52, 0.9],
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

    return { index: handIndex, raw: landmarks, points };
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
      width: { ideal: settings.lowPower ? 960 : 1280 },
      height: { ideal: settings.lowPower ? 540 : 720 },
      frameRate: { ideal: settings.lowPower ? 24 : 30, max: settings.lowPower ? 30 : 60 },
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

function maybeEnableAutoLowPower(deltaMs) {
  frameCostEma = frameCostEma * 0.94 + deltaMs * 0.06;

  if (!settings.lowPower && frameCostEma > 34 && !autoLowPowerNoticeShown) {
    settings.lowPower = true;
    lowPower.checked = true;
    particles.splice(0, Math.max(0, particles.length - 420));
    ribbons.splice(0, Math.max(0, ribbons.length - 100));
    resizeCanvas();
    setStatus('Performance mode on');
    autoLowPowerNoticeShown = true;
  }
}

function render() {
  const now = performance.now();
  const frameMs = now - lastFrameTime;
  const delta = Math.min(2, frameMs / 16.67);
  lastFrameTime = now;
  maybeEnableAutoLowPower(frameMs);

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

function syncControlValues() {
  cameraOpacity.value = Math.round(settings.cameraOpacity * 100);
  particleIntensity.value = Math.round(settings.intensity * 100);
  glowStrength.value = Math.round(settings.glow * 100);
  trailLife.value = Math.round(settings.trailLife * 100);
  smoothing.value = Math.round(settings.smoothing * 100);
  showSkeleton.checked = settings.showSkeleton;
  mirrorCamera.checked = settings.mirror;
  lowPower.checked = settings.lowPower;
  updateSettingLabels();
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

  lowPower.addEventListener('change', () => {
    settings.lowPower = lowPower.checked;
    resizeCanvas();
    if (settings.lowPower) {
      particles.splice(0, Math.max(0, particles.length - 420));
      ribbons.splice(0, Math.max(0, ribbons.length - 100));
    }
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
syncControlValues();
applyCameraStyle();
resizeCanvas();
render();
