# Finger Flow

A satisfying webcam hand-tracking particle toy powered by **MediaPipe Hand Landmarker**, **Vite**, and **HTML Canvas**.

Move your index finger to draw glowing particle trails. Pinch your thumb and index finger to trigger a particle burst.

## Demo

After GitHub Pages finishes deploying, the app should be available at:

```txt
https://naneps.github.io/finger-flow/
```

> Camera access works best on `https://` or `localhost`.

## Features

- Webcam camera preview
- Mirrored front-camera interaction
- Real-time hand tracking with MediaPipe
- Index-finger particle brush
- Pinch gesture burst effect
- Glowing particles with satisfying fading trails
- Responsive desktop/mobile layout
- GitHub Pages deployment workflow

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
