import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("webcam");
const hudCanvas = document.getElementById("hud");
const videoFxCanvas = document.getElementById("videoFx");

const startBtn = document.getElementById("startBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const calibrateBtn = document.getElementById("calibrateBtn");
const debugBtn = document.getElementById("debugBtn");

const statusPill = document.getElementById("statusPill");
const trackingText = document.getElementById("trackingText");
const cameraText = document.getElementById("cameraText");
const lightingText = document.getElementById("lightingText");
const poseText = document.getElementById("poseText");
const voiceText = document.getElementById("voiceText");
const warningBanner = document.getElementById("warningBanner");
const logsEl = document.getElementById("logs");

const hudCtx = hudCanvas.getContext("2d");
const videoFxCtx = videoFxCanvas.getContext("2d", { willReadFrequently: true });

let faceLandmarker = null;
let drawingUtils = null;
let stream = null;
let running = false;
let debugMode = false;
let lastVideoTime = -1;
let animationId = 0;
let qualitySampleTimer = 0;
let lastVoiceAt = 0;
let lastLogAt = 0;

const state = {
  cameraWidth: 0,
  cameraHeight: 0,
  brightness: 0,
  lightState: "Unknown",
  trackingState: "OFFLINE",
  lastSeenAt: 0,
  facePresent: false,
  faceCenterX: 0.5,
  faceCenterY: 0.5,
  faceScale: 0.2,
  leftEyeX: 0.46,
  leftEyeY: 0.45,
  rightEyeX: 0.54,
  rightEyeY: 0.45,
  rawYaw: 0,
  rawPitch: 0,
  rawRoll: 0,
  yaw: 0,
  pitch: 0,
  roll: 0,
  calibrated: false,
  zeroYaw: 0,
  zeroPitch: 0,
  zeroRoll: 0,
  smoothTargetX: 0.5,
  smoothTargetY: 0.5,
  ringPhase: 0,
  sweepPhase: 0,
  pulse: 0,
  fpsTime: performance.now(),
  fpsFrames: 0,
  fps: 0,
  warning: "",
  lastBlendshapeCategory: "",
  headConfidence: 0
};

const voiceMessages = [
  "SYSTEM STANDBY",
  "VISOR LINK STABLE",
  "FACE TRACK LOCKED",
  "HEAD POSE MATRIX STABLE",
  "TARGETING ARRAY NOMINAL",
  "LOW LATENCY TRACKING ACTIVE",
  "ENVIRONMENTAL ANALYSIS RUNNING",
  "DIAGNOSTICS ONLINE",
  "MOTION COMPENSATION ENGAGED",
  "CALIBRATION PROFILE READY"
];

function setStatus(text, mode = "ok") {
  statusPill.textContent = text;
  statusPill.className = mode;
}

function setVoice(text) {
  voiceText.textContent = text;
}

function log(message) {
  const now = new Date();
  const stamp = now.toLocaleTimeString();
  logsEl.value = `[${stamp}] ${message}\n` + logsEl.value.slice(0, 14000);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smooth(current, target, factor) {
  return current + (target - current) * factor;
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  const t = clamp((value - inMin) / (inMax - inMin), 0, 1);
  return outMin + (outMax - outMin) * t;
}

function applyDeadZone(value, zone) {
  if (Math.abs(value) < zone) return 0;
  return value;
}

function resizeCanvases() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = window.innerWidth;
  const height = window.innerHeight;

  hudCanvas.width = Math.round(width * dpr);
  hudCanvas.height = Math.round(height * dpr);
  hudCanvas.style.width = `${width}px`;
  hudCanvas.style.height = `${height}px`;

  videoFxCanvas.width = Math.round(width * dpr);
  videoFxCanvas.height = Math.round(height * dpr);
  videoFxCanvas.style.width = `${width}px`;
  videoFxCanvas.style.height = `${height}px`;

  hudCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  videoFxCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function updateCameraText() {
  if (!state.cameraWidth || !state.cameraHeight) return;
  cameraText.textContent = `${state.cameraWidth}x${state.cameraHeight} @ ${state.fps.toFixed(0)}fps`;
}

async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 }
    }
  });

  video.srcObject = stream;
  await video.play();

  state.cameraWidth = video.videoWidth || 1280;
  state.cameraHeight = video.videoHeight || 720;
  updateCameraText();
  log(`Camera started: ${state.cameraWidth}x${state.cameraHeight}`);
}

async function setupFaceTracking() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    numFaces: 1
  });

  drawingUtils = new DrawingUtils(hudCtx);
  log("Face landmarker ready");
}

function extractFaceMetrics(result) {
  if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
    state.facePresent = false;
    return;
  }

  const landmarks = result.faceLandmarks[0];
  state.facePresent = true;
  state.lastSeenAt = performance.now();

  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const faceCenterX = (minX + maxX) * 0.5;
  const faceCenterY = (minY + maxY) * 0.5;
  const faceScale = Math.max(maxX - minX, maxY - minY);

  state.faceCenterX = faceCenterX;
  state.faceCenterY = faceCenterY;
  state.faceScale = faceScale;

  const leftEye = landmarks[468] || landmarks[159] || landmarks[33];
  const rightEye = landmarks[473] || landmarks[386] || landmarks[263];

  if (leftEye && rightEye) {
    state.leftEyeX = leftEye.x;
    state.leftEyeY = leftEye.y;
    state.rightEyeX = rightEye.x;
    state.rightEyeY = rightEye.y;
  }

  const nose = landmarks[1];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  const forehead = landmarks[10];
  const chin = landmarks[152];

  if (nose && leftCheek && rightCheek && forehead && chin) {
    const horizMidX = (leftCheek.x + rightCheek.x) * 0.5;
    const vertMidY = (forehead.y + chin.y) * 0.5;

    let yaw = (nose.x - horizMidX) / Math.max(0.001, (rightCheek.x - leftCheek.x));
    let pitch = (nose.y - vertMidY) / Math.max(0.001, (chin.y - forehead.y));
    const roll = Math.atan2(
      state.rightEyeY - state.leftEyeY,
      state.rightEyeX - state.leftEyeX
    );

    yaw = clamp(yaw * 2.4, -1.2, 1.2);
    pitch = clamp(pitch * 2.8, -1.2, 1.2);

    state.rawYaw = yaw;
    state.rawPitch = pitch;
    state.rawRoll = roll;

    let relYaw = yaw - state.zeroYaw;
    let relPitch = pitch - state.zeroPitch;
    let relRoll = roll - state.zeroRoll;

    relYaw = applyDeadZone(relYaw, 0.02);
    relPitch = applyDeadZone(relPitch, 0.02);
    relRoll = applyDeadZone(relRoll, 0.01);

    state.yaw = smooth(state.yaw, clamp(relYaw, -1, 1), 0.14);
    state.pitch = smooth(state.pitch, clamp(relPitch, -1, 1), 0.14);
    state.roll = smooth(state.roll, clamp(relRoll, -0.8, 0.8), 0.12);
  }

  if (result.faceBlendshapes?.[0]?.categories?.length) {
    const strongest = [...result.faceBlendshapes[0].categories]
      .sort((a, b) => b.score - a.score)[0];
    state.lastBlendshapeCategory = `${strongest.categoryName}:${strongest.score.toFixed(2)}`;
    state.headConfidence = strongest.score;
  }

  state.smoothTargetX = smooth(state.smoothTargetX, state.faceCenterX, 0.12);
  state.smoothTargetY = smooth(state.smoothTargetY, state.faceCenterY, 0.12);
}

function analyzeFrameLighting() {
  if (!video.videoWidth || !video.videoHeight) return;

  const sampleW = 160;
  const sampleH = 90;

  videoFxCtx.clearRect(0, 0, sampleW, sampleH);
  videoFxCtx.save();
  videoFxCtx.scale(-1, 1);
  videoFxCtx.drawImage(video, -sampleW, 0, sampleW, sampleH);
  videoFxCtx.restore();

  const image = videoFxCtx.getImageData(0, 0, sampleW, sampleH).data;
  let total = 0;

  for (let i = 0; i < image.length; i += 4) {
    total += (image[i] + image[i + 1] + image[i + 2]) / 3;
  }

  const brightness = total / (image.length / 4);
  state.brightness = brightness;

  let lightState = "Good";
  if (brightness < 45) lightState = "Very Dark";
  else if (brightness < 70) lightState = "Low";
  else if (brightness > 205) lightState = "Overexposed";

  state.lightState = lightState;
  lightingText.textContent = `${lightState} (${brightness.toFixed(0)})`;
}

function refreshWarnings() {
  const now = performance.now();
  const lostFor = now - state.lastSeenAt;

  let warning = "";

  if (!state.facePresent || lostFor > 500) {
    warning = "NO FACE LOCK";
    setStatus("SEARCHING", "warn");
    state.trackingState = "SEARCHING";
  } else if (state.lightState === "Very Dark" || state.lightState === "Overexposed") {
    warning = "BAD LIGHTING";
    setStatus("DEGRADED", "warn");
    state.trackingState = "DEGRADED";
  } else if (state.faceScale < 0.13) {
    warning = "MOVE CLOSER";
    setStatus("DEGRADED", "warn");
    state.trackingState = "DEGRADED";
  } else {
    setStatus("LOCKED", "ok");
    state.trackingState = "LOCKED";
  }

  state.warning = warning;
  trackingText.textContent = state.trackingState;

  if (warning) {
    warningBanner.classList.add("show");
    warningBanner.textContent = warning;
  } else {
    warningBanner.classList.remove("show");
  }
}

function updatePoseText() {
  poseText.textContent =
    `yaw ${state.yaw.toFixed(2)} / pitch ${state.pitch.toFixed(2)} / roll ${state.roll.toFixed(2)}`;
}

function maybeUpdateVoice() {
  const now = performance.now();
  if (now - lastVoiceAt < 2200) return;

  lastVoiceAt = now;

  if (state.warning === "NO FACE LOCK") {
    setVoice("FACE LOCK LOST");
    return;
  }

  if (state.warning === "BAD LIGHTING") {
    setVoice("VISUAL CONDITIONS DEGRADED");
    return;
  }

  if (state.warning === "MOVE CLOSER") {
    setVoice("PILOT TOO FAR FROM OPTICS");
    return;
  }

  const idx = Math.floor(Math.random() * voiceMessages.length);
  setVoice(voiceMessages[idx]);
}

function maybeLog() {
  const now = performance.now();
  if (now - lastLogAt < 900) return;
  lastLogAt = now;

  log(
    [
      `tracking=${state.trackingState}`,
      `light=${state.lightState}`,
      `bright=${state.brightness.toFixed(0)}`,
      `faceScale=${state.faceScale.toFixed(3)}`,
      `yaw=${state.yaw.toFixed(3)}`,
      `pitch=${state.pitch.toFixed(3)}`,
      `roll=${state.roll.toFixed(3)}`,
      `center=(${state.faceCenterX.toFixed(3)}, ${state.faceCenterY.toFixed(3)})`
    ].join(" | ")
  );
}

function calibrate() {
  state.zeroYaw = state.rawYaw;
  state.zeroPitch = state.rawPitch;
  state.zeroRoll = state.rawRoll;
  state.calibrated = true;
  setVoice("CALIBRATION CAPTURED");
  log(
    `Calibration set | yaw=${state.zeroYaw.toFixed(3)} pitch=${state.zeroPitch.toFixed(3)} roll=${state.zeroRoll.toFixed(3)}`
  );
}

function drawHelmetVignette(ctx, width, height, t) {
  const cx = width / 2;
  const cy = height / 2;
  const vignette = ctx.createRadialGradient(cx, cy, Math.min(width, height) * 0.22, cx, cy, Math.max(width, height) * 0.76);
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(0.65, "rgba(0,0,0,0.08)");
  vignette.addColorStop(0.82, "rgba(0,0,0,0.42)");
  vignette.addColorStop(1, "rgba(0,0,0,0.85)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.strokeStyle = "rgba(90,180,255,0.22)";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(90,180,255,0.3)";
  ctx.shadowBlur = 20;

  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.47, height * 0.45, 0, 0.12 * Math.PI, 0.88 * Math.PI);
  ctx.stroke();

  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.47, height * 0.45, 0, 1.12 * Math.PI, 1.88 * Math.PI);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(width * 0.12, cy);
  ctx.quadraticCurveTo(width * 0.2, cy - 40, width * 0.28, cy - 12);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(width * 0.88, cy);
  ctx.quadraticCurveTo(width * 0.8, cy - 40, width * 0.72, cy - 12);
  ctx.stroke();

  ctx.restore();

  const scanGlow = 0.07 + 0.03 * Math.sin(t * 1.5);
  ctx.fillStyle = `rgba(80,170,255,${scanGlow})`;
  ctx.fillRect(0, 0, width, height);
}

function drawGridArc(ctx, width, height, yawPx, pitchPx) {
  ctx.save();
  ctx.translate(yawPx * 0.15, pitchPx * 0.15);
  ctx.strokeStyle = "rgba(90,180,255,0.08)";
  ctx.lineWidth = 1;

  const baseY = height * 0.74;
  const amplitude = height * 0.08;

  for (let i = -6; i <= 6; i++) {
    const y = baseY + i * 16;
    ctx.beginPath();
    for (let x = 0; x <= width; x += 10) {
      const nx = x / width;
      const curve = Math.sin((nx * Math.PI) + i * 0.18) * amplitude;
      const py = y + curve * 0.22;
      if (x === 0) ctx.moveTo(x, py);
      else ctx.lineTo(x, py);
    }
    ctx.stroke();
  }

  for (let i = 0; i <= 14; i++) {
    const x = width * (i / 14);
    ctx.beginPath();
    ctx.moveTo(x, baseY - 60);
    ctx.lineTo(x + (x - width / 2) * 0.08, height);
    ctx.stroke();
  }

  ctx.restore();
}

function drawCenterReticle(ctx, width, height, t) {
  const cx = lerp(width * 0.5, width * state.smoothTargetX, 0.45);
  const cy = lerp(height * 0.5, height * state.smoothTargetY, 0.45);

  const yawPx = state.yaw * 140;
  const pitchPx = state.pitch * 110;
  const roll = state.roll * 0.8;

  ctx.save();
  ctx.translate(cx + yawPx * 0.14, cy + pitchPx * 0.1);
  ctx.rotate(roll);

  const r = mapRange(state.faceScale, 0.12, 0.45, 72, 118);

  ctx.shadowColor = "rgba(90,180,255,0.45)";
  ctx.shadowBlur = 24;

  ctx.strokeStyle = "rgba(90,180,255,0.88)";
  ctx.lineWidth = 2.2;

  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(90,180,255,0.28)";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(0, 0, r + 12, t * 0.9, t * 0.9 + Math.PI * 1.45);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.arc(0, 0, r - 16, -t * 1.2, -t * 1.2 + Math.PI * 1.18);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(90,180,255,0.75)";
  ctx.lineWidth = 1.2;
  for (let i = 0; i < 8; i++) {
    const a = (Math.PI * 2 * i) / 8 + t * 0.22;
    const x1 = Math.cos(a) * (r - 32);
    const y1 = Math.sin(a) * (r - 32);
    const x2 = Math.cos(a) * (r + 22);
    const y2 = Math.sin(a) * (r + 22);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(180,230,255,0.95)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(-24, 0);
  ctx.lineTo(24, 0);
  ctx.moveTo(0, -24);
  ctx.lineTo(0, 24);
  ctx.stroke();

  ctx.fillStyle = "rgba(190,235,255,0.95)";
  ctx.beginPath();
  ctx.arc(0, 0, 3.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawEyeTargets(ctx, width, height, t) {
  const leftX = width * state.leftEyeX;
  const leftY = height * state.leftEyeY;
  const rightX = width * state.rightEyeX;
  const rightY = height * state.rightEyeY;

  ctx.save();
  ctx.strokeStyle = "rgba(90,180,255,0.82)";
  ctx.fillStyle = "rgba(90,180,255,0.18)";
  ctx.lineWidth = 1.4;
  ctx.shadowColor = "rgba(90,180,255,0.35)";
  ctx.shadowBlur = 14;

  for (const [x, y] of [[leftX, leftY], [rightX, rightY]]) {
    const r = 18 + Math.sin(t * 3.6) * 2;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, r + 10, t * 1.6, t * 1.6 + Math.PI * 0.9);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x - 26, y);
    ctx.lineTo(x - 10, y);
    ctx.moveTo(x + 10, y);
    ctx.lineTo(x + 26, y);
    ctx.moveTo(x, y - 26);
    ctx.lineTo(x, y - 10);
    ctx.moveTo(x, y + 10);
    ctx.lineTo(x, y + 26);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawSweep(ctx, width, height, t) {
  const sweepX = (t * 220) % (width + 320) - 160;
  const grad = ctx.createLinearGradient(sweepX - 70, 0, sweepX + 70, 0);
  grad.addColorStop(0, "rgba(90,180,255,0)");
  grad.addColorStop(0.5, "rgba(90,180,255,0.12)");
  grad.addColorStop(1, "rgba(90,180,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(sweepX - 90, 0, 180, height);
}

function drawPanels(ctx, width, height, t) {
  const yawPx = state.yaw * 100;
  const pitchPx = state.pitch * 80;

  const panels = [
    {
      x: width * 0.08 + yawPx * 0.45,
      y: height * 0.18 + pitchPx * 0.28,
      w: 260,
      h: 150,
      title: "TARGET ANALYSIS"
    },
    {
      x: width * 0.72 + yawPx * 0.62,
      y: height * 0.14 + pitchPx * 0.24,
      w: 220,
      h: 170,
      title: "ENERGY PROFILE"
    },
    {
      x: width * 0.77 + yawPx * 0.58,
      y: height * 0.47 + pitchPx * 0.34,
      w: 200,
      h: 180,
      title: "SYSTEM HEALTH"
    },
    {
      x: width * 0.1 + yawPx * 0.34,
      y: height * 0.63 + pitchPx * 0.38,
      w: 280,
      h: 140,
      title: "TRAJECTORY GRID"
    }
  ];

  panels.forEach((p, idx) => {
    drawPanel(ctx, p.x, p.y, p.w, p.h, p.title, t, idx);
  });
}

function drawPanel(ctx, x, y, w, h, title, t, variant) {
  ctx.save();

  ctx.strokeStyle = "rgba(90,180,255,0.24)";
  ctx.fillStyle = "rgba(4,14,24,0.22)";
  ctx.lineWidth = 1.2;
  ctx.shadowColor = "rgba(90,180,255,0.24)";
  ctx.shadowBlur = 12;

  roundRect(ctx, x, y, w, h, 16);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(180,230,255,0.9)";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText(title, x + 14, y + 20);

  ctx.strokeStyle = "rgba(90,180,255,0.16)";
  ctx.beginPath();
  ctx.moveTo(x + 14, y + 30);
  ctx.lineTo(x + w - 14, y + 30);
  ctx.stroke();

  if (variant === 0) {
    const bars = 18;
    for (let i = 0; i < bars; i++) {
      const bx = x + 16 + i * 12;
      const bh = 12 + Math.abs(Math.sin(t * 2 + i * 0.4)) * 62;
      ctx.fillStyle = i % 3 === 0 ? "rgba(220,245,255,0.88)" : "rgba(90,180,255,0.65)";
      ctx.fillRect(bx, y + h - 18 - bh, 8, bh);
    }
  } else if (variant === 1) {
    const cx = x + w * 0.56;
    const cy = y + h * 0.58;
    ctx.strokeStyle = "rgba(90,180,255,0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 44, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, 62, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * (0.55 + 0.2 * Math.sin(t))));
    ctx.stroke();

    ctx.fillStyle = "rgba(200,240,255,0.92)";
    ctx.font = "28px Inter, system-ui, sans-serif";
    ctx.fillText(`${(76 + 10 * Math.sin(t * 0.8)).toFixed(0)}%`, cx - 28, cy + 10);
  } else if (variant === 2) {
    const lines = [
      ["TRACK", state.trackingState],
      ["LIGHT", state.lightState],
      ["FPS", state.fps.toFixed(0)],
      ["MODE", state.calibrated ? "CALIBRATED" : "UNCALIBRATED"]
    ];
    ctx.fillStyle = "rgba(190,235,255,0.86)";
    ctx.font = "12px ui-monospace, monospace";
    lines.forEach((line, i) => {
      ctx.fillText(`${line[0]} : ${line[1]}`, x + 14, y + 58 + i * 26);
    });
  } else if (variant === 3) {
    ctx.strokeStyle = "rgba(90,180,255,0.24)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      const py = y + 42 + i * 10;
      ctx.beginPath();
      for (let j = 0; j < 26; j++) {
        const px = x + 14 + j * 10;
        const wave = Math.sin((j * 0.45) + t * 1.8 + i * 0.6) * 5;
        if (j === 0) ctx.moveTo(px, py + wave);
        else ctx.lineTo(px, py + wave);
      }
      ctx.stroke();
    }
  }

  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawSideWidgets(ctx, width, height, t) {
  const yawPx = state.yaw * 120;
  const pitchPx = state.pitch * 90;

  ctx.save();
  ctx.translate(yawPx * 0.18, pitchPx * 0.16);
  ctx.strokeStyle = "rgba(90,180,255,0.28)";
  ctx.lineWidth = 1.2;

  for (let i = 0; i < 5; i++) {
    const x = width - 180;
    const y = 150 + i * 88;
    ctx.beginPath();
    ctx.roundRect?.(x, y, 110, 44, 12);
    if (!ctx.roundRect) roundRect(ctx, x, y, 110, 44, 12);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x + 84, y + 22, 13 + Math.sin(t * 1.4 + i) * 1.4, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x + 14, y + 22);
    ctx.lineTo(x + 44, y + 22);
    ctx.stroke();
  }

  ctx.restore();
}

function drawDataTicks(ctx, width, height, t) {
  ctx.save();
  ctx.strokeStyle = "rgba(90,180,255,0.18)";
  ctx.lineWidth = 1;

  for (let i = 0; i < 22; i++) {
    const x = 40 + i * 42;
    const len = i % 4 === 0 ? 18 : 8;
    ctx.beginPath();
    ctx.moveTo(x, 34);
    ctx.lineTo(x, 34 + len);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, height - 34);
    ctx.lineTo(x, height - 34 - len);
    ctx.stroke();
  }

  for (let i = 0; i < 14; i++) {
    const y = 80 + i * 38;
    const len = i % 3 === 0 ? 18 : 9;
    ctx.beginPath();
    ctx.moveTo(26, y);
    ctx.lineTo(26 + len, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(width - 26, y);
    ctx.lineTo(width - 26 - len, y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawDebug(ctx, width, height, result) {
  if (!debugMode || !result?.faceLandmarks?.length) return;

  const landmarks = result.faceLandmarks[0];
  ctx.save();
  ctx.globalAlpha = 0.85;

  if (drawingUtils) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "rgba(80,180,255,0.28)", lineWidth: 0.5 }
    );
    drawingUtils.drawLandmarks(landmarks, {
      color: "rgba(255,255,255,0.9)",
      radius: 1
    });
  }

  const xs = landmarks.map((p) => p.x * width);
  const ys = landmarks.map((p) => p.y * height);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

  ctx.fillStyle = "rgba(200,240,255,0.95)";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`yaw ${state.yaw.toFixed(2)}`, minX, minY - 24);
  ctx.fillText(`pitch ${state.pitch.toFixed(2)}`, minX, minY - 10);
  ctx.fillText(`roll ${state.roll.toFixed(2)}`, minX, minY + (maxY - minY) + 18);

  ctx.restore();
}

function drawHud(result) {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const t = performance.now() * 0.001;

  hudCtx.clearRect(0, 0, width, height);

  hudCtx.save();

  const globalRoll = state.roll * 0.18;
  const parallaxX = state.yaw * 36;
  const parallaxY = state.pitch * 28;

  hudCtx.translate(width / 2 + parallaxX, height / 2 + parallaxY);
  hudCtx.rotate(globalRoll);
  hudCtx.translate(-width / 2, -height / 2);

  drawHelmetVignette(hudCtx, width, height, t);
  drawGridArc(hudCtx, width, height, parallaxX, parallaxY);
  drawPanels(hudCtx, width, height, t);
  drawSideWidgets(hudCtx, width, height, t);
  drawDataTicks(hudCtx, width, height, t);
  drawSweep(hudCtx, width, height, t);
  drawCenterReticle(hudCtx, width, height, t);
  drawEyeTargets(hudCtx, width, height, t);

  hudCtx.restore();

  drawDebug(hudCtx, width, height, result);
}

function renderLoop() {
  if (!running) return;

  const now = performance.now();

  if (video.readyState >= 2 && faceLandmarker) {
    state.fpsFrames++;
    if (now - state.fpsTime >= 1000) {
      state.fps = (state.fpsFrames * 1000) / (now - state.fpsTime);
      state.fpsFrames = 0;
      state.fpsTime = now;
      updateCameraText();
    }

    let result = null;

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      result = faceLandmarker.detectForVideo(video, now);
      extractFaceMetrics(result);
    }

    if (now - qualitySampleTimer > 500) {
      qualitySampleTimer = now;
      analyzeFrameLighting();
    }

    refreshWarnings();
    updatePoseText();
    maybeUpdateVoice();
    maybeLog();
    drawHud(result);
  } else {
    drawHud(null);
  }

  animationId = requestAnimationFrame(renderLoop);
}

async function start() {
  if (running) return;

  try {
    setStatus("BOOTING", "warn");
    setVoice("INITIALIZING HELMET INTERFACE");
    await setupCamera();
    await setupFaceTracking();
    resizeCanvases();

    running = true;
    state.lastSeenAt = performance.now();
    renderLoop();

    setStatus("ONLINE", "ok");
    setVoice("HELMET INTERFACE ONLINE");
    log("System online");
  } catch (error) {
    console.error(error);
    setStatus("ERROR", "bad");
    setVoice("INITIALIZATION FAILURE");
    log(`Error: ${error.message}`);
    trackingText.textContent = "ERROR";
  }
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}

startBtn.addEventListener("click", start);
fullscreenBtn.addEventListener("click", toggleFullscreen);
calibrateBtn.addEventListener("click", calibrate);

debugBtn.addEventListener("click", () => {
  debugMode = !debugMode;
  debugBtn.textContent = `Debug: ${debugMode ? "On" : "Off"}`;
  setVoice(debugMode ? "DEBUG VISUALS ENABLED" : "DEBUG VISUALS DISABLED");
  log(`Debug mode ${debugMode ? "enabled" : "disabled"}`);
});

window.addEventListener("resize", resizeCanvases);

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    setVoice("STANDBY");
  }
});

log("Ready. Click Start HUD.");
setVoice("SYSTEM STANDBY");
resizeCanvases();
