/**
 * useHandLandmarks — MediaPipe Hands in-browser landmark extraction.
 * Runs entirely client-side. Raw video never leaves the device.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

export function useHandLandmarks() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [landmarks, setLandmarks] = useState(null);
  const [hasHand, setHasHand] = useState(false);
  const [fps, setFps] = useState(0);
  const streamRef = useRef(null);
  const handsRef = useRef(null);
  const animFrameRef = useRef(null);
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: 'user'
        }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsActive(true);
    } catch (err) {
      console.error('Camera error:', err);
      alert('Camera access denied. Please allow camera permissions.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    setIsActive(false);
    setHasHand(false);
    setLandmarks(null);
  }, []);

  useEffect(() => {
    // Load MediaPipe Hands
    const loadMediaPipe = async () => {
      if (window.Hands) return;
      
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js');
    };
    
    loadMediaPipe().catch(console.error);
    
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  useEffect(() => {
    if (!isActive || !videoRef.current) return;
    
    const initHands = async () => {
      if (!window.Hands) {
        console.warn('MediaPipe Hands not loaded yet');
        return;
      }
      
      const hands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
      });
      
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });
      
      hands.onResults((results) => {
        const canvas = canvasRef.current;
        if (!canvas || !videoRef.current) return;
        
        const ctx = canvas.getContext('2d');
        const vw = videoRef.current.videoWidth || 1280;
        const vh = videoRef.current.videoHeight || 720;
        canvas.width = vw;
        canvas.height = vh;
        ctx.clearRect(0, 0, vw, vh);
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
          const hand = results.multiHandLandmarks[0];
          setHasHand(true);
          
          // Draw high-visibility glowing skeleton
          if (window.drawConnectors && window.drawLandmarks) {
            // Shadow glow
            ctx.save();
            ctx.shadowColor = '#00f0ff';
            ctx.shadowBlur = 10;
            window.drawConnectors(ctx, hand, window.HAND_CONNECTIONS, {
              color: '#00e5ff', lineWidth: 4
            });
            ctx.restore();

            // Inner bones
            window.drawConnectors(ctx, hand, window.HAND_CONNECTIONS, {
              color: '#ffffff', lineWidth: 2
            });

            // Joints
            window.drawLandmarks(ctx, hand, {
              color: '#fbbf24', fillColor: '#f59e0b', lineWidth: 2, radius: 4
            });

            // Highlight fingertips (4, 8, 12, 16, 20)
            const fingertips = [4, 8, 12, 16, 20];
            fingertips.forEach(idx => {
              const pt = hand[idx];
              if (pt) {
                const x = pt.x * vw;
                const y = pt.y * vh;
                ctx.beginPath();
                ctx.arc(x, y, 7, 0, 2 * Math.PI);
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 3;
                ctx.fillStyle = '#ffffff';
                ctx.fill();
                ctx.stroke();
              }
            });
          }
          
          // Compute bounding box around hand
          let minX = 1, minY = 1, maxX = 0, maxY = 0;
          hand.forEach(lm => {
            if (lm.x < minX) minX = lm.x;
            if (lm.x > maxX) maxX = lm.x;
            if (lm.y < minY) minY = lm.y;
            if (lm.y > maxY) maxY = lm.y;
          });
          const bx = Math.max(0, minX * vw - 15);
          const by = Math.max(0, minY * vh - 15);
          const bw = Math.min(vw - bx, (maxX - minX) * vw + 30);
          const bh = Math.min(vh - by, (maxY - minY) * vh + 30);

          ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 6]);
          ctx.strokeRect(bx, by, bw, bh);
          ctx.setLineDash([]);

          ctx.fillStyle = 'rgba(0, 240, 255, 0.85)';
          ctx.fillRect(bx, by - 22, 100, 22);
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 11px sans-serif';
          ctx.fillText('HAND TRACKED', bx + 6, by - 7);
          
          // Normalize landmarks
          const rawLandmarks = [];
          for (const lm of hand) {
            rawLandmarks.push(lm.x, lm.y, lm.z);
          }
          
          // Normalize: center on wrist, scale by max distance
          const coords = [];
          for (let i = 0; i < 21; i++) {
            coords.push([rawLandmarks[i*3], rawLandmarks[i*3+1], rawLandmarks[i*3+2]]);
          }
          
          const wrist = [...coords[0]];
          const centered = coords.map(c => [c[0]-wrist[0], c[1]-wrist[1], c[2]-wrist[2]]);
          
          let maxDist = 0;
          for (const c of centered) {
            const d = Math.sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
            if (d > maxDist) maxDist = d;
          }
          
          const normalized = [];
          for (const c of centered) {
            if (maxDist > 1e-6) {
              normalized.push(c[0]/maxDist, c[1]/maxDist, c[2]/maxDist);
            } else {
              normalized.push(c[0], c[1], c[2]);
            }
          }
          
          setLandmarks(normalized);
        } else {
          setHasHand(false);
          setLandmarks(null);
        }
        
        // FPS counter
        fpsCounterRef.current.frames++;
        const now = Date.now();
        if (now - fpsCounterRef.current.lastTime >= 1000) {
          setFps(fpsCounterRef.current.frames);
          fpsCounterRef.current.frames = 0;
          fpsCounterRef.current.lastTime = now;
        }
      });
      
      handsRef.current = hands;
      
      // Process frames
      const processFrame = async () => {
        if (videoRef.current && videoRef.current.readyState >= 2 && handsRef.current) {
          await handsRef.current.send({ image: videoRef.current });
        }
        animFrameRef.current = requestAnimationFrame(processFrame);
      };
      
      processFrame();
    };
    
    // Small delay to ensure MediaPipe is loaded
    const timer = setTimeout(initHands, 500);
    return () => {
      clearTimeout(timer);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isActive]);

  return {
    videoRef,
    canvasRef,
    isActive,
    startCamera,
    stopCamera,
    landmarks,
    hasHand,
    fps,
  };
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}
