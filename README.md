# FingerPlay

A liquid glass webcam playground powered by **MediaPipe Hand Landmarker**, **Vite**, and **HTML Canvas**.

Move your fingers to bend glowing liquid particles. Pinch your thumb and index finger to trigger a burst.

## Demo

After GitHub Pages finishes deploying, the app is available at:

```txt
https://naneps.github.io/finger-flow/
```

> Camera access works best on `https://` or `localhost`.

## Features

- Liquid glass UI theme
- Webcam camera preview
- Adjustable camera opacity
- Mirrored front-camera interaction
- Real-time hand tracking with MediaPipe
- Pinch gesture particle burst
- Particle intensity, glow, trail life, and smoothing controls
- Skeleton and mirror toggles
- Clear particles button
- Responsive desktop/mobile layout
- GitHub Pages deployment workflow

## Visual modes

- **Liquid Flow** — index finger paints liquid trails
- **All Fingers** — every fingertip becomes a brush
- **3D Hand** — depth-reactive glowing hand rig
- **Liquid Ribbons** — fingers draw soft ribbon strokes
- **Constellation** — hand landmarks become a star map
- **Gravity Orbs** — particles orbit around fingertips

## Run locally

```bash
npm install
npm run dev
```

Open the local URL shown by Vite, usually:

```txt
http://localhost:5173
```

## Build

```bash
npm run build
npm run preview
```

## GitHub Pages deployment

This repo includes a workflow at:

```txt
.github/workflows/deploy.yml
```

It builds the Vite app and deploys `dist` to GitHub Pages on every push to `main`.

If the deployment does not appear automatically:

1. Open the repository on GitHub.
2. Go to **Settings** → **Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Go to the **Actions** tab and rerun **Deploy to GitHub Pages**.

## Tech stack

- Vite
- JavaScript
- MediaPipe Tasks Vision
- HTML Canvas
- GitHub Actions
