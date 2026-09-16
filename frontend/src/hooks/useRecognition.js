/**
 * useRecognition — Recognition hook supporting both Static Digits and Temporal Words.
 * Handles rolling 32-frame trajectory buffering for temporal (video dataset) recognition
 * and throttled single-frame requests for static digit recognition.
 */
import { useState, useRef, useCallback } from 'react';
import { recognize } from '../lib/api';

const SEQ_LEN = 32;
const FEATURE_DIM = 63;
const MIN_STATIC_INTERVAL = 100;    // ~10 req/sec for digits
const MIN_TEMPORAL_INTERVAL = 140;  // ~7 req/sec for words

export function useRecognition(sessionId) {
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('idle');
  const [latency, setLatency] = useState(0);
  const [bufferFill, setBufferFill] = useState(0); // 0 to 100%

  const lastRequestTime = useRef(0);
  const lastTemporalSampleTime = useRef(0);
  const pendingRequest = useRef(false);
  const sequenceBuffer = useRef([]);
  const consecutiveNoHand = useRef(0);

  const pushFrame = useCallback(async (landmarks, mode = 'static') => {
    const now = Date.now();

    // If hand is missing, increment counter and reset sequence if missing for a while
    if (!landmarks || landmarks.length !== FEATURE_DIM) {
      consecutiveNoHand.current += 1;
      if (consecutiveNoHand.current > 12) {
        sequenceBuffer.current = [];
        setBufferFill(0);
      }
      return;
    }

    consecutiveNoHand.current = 0;

    let payloadLandmarks = null;

    const isTemporal = mode === 'temporal' || mode === 'gesture_asl';

    if (isTemporal) {
      // Sample frames at ~75ms interval so 32 frames covers ~2.4s of human sign motion
      if (now - lastTemporalSampleTime.current < 75) {
        return;
      }
      lastTemporalSampleTime.current = now;

      // Add current frame to sliding trajectory buffer
      sequenceBuffer.current.push(landmarks);
      if (sequenceBuffer.current.length > SEQ_LEN) {
        sequenceBuffer.current = sequenceBuffer.current.slice(-SEQ_LEN);
      }

      const count = sequenceBuffer.current.length;
      setBufferFill(Math.round((count / SEQ_LEN) * 100));

      // We need at least 10 frames to start making meaningful temporal predictions
      if (count < 8) return;

      // Throttle temporal requests
      if (now - lastRequestTime.current < MIN_TEMPORAL_INTERVAL) return;
      if (pendingRequest.current) return;

      // Build 32-frame flat array (pad beginning with zeros if under 32)
      const frames = [];
      const padCount = SEQ_LEN - count;
      for (let i = 0; i < padCount; i++) {
        frames.push(new Array(FEATURE_DIM).fill(0.0));
      }
      for (let i = 0; i < count; i++) {
        frames.push(sequenceBuffer.current[i]);
      }

      // Flatten to 2016 numbers
      payloadLandmarks = frames.flat();

    } else {
      // Static mode: single 63-dim frame
      if (now - lastRequestTime.current < MIN_STATIC_INTERVAL) return;
      if (pendingRequest.current) return;

      payloadLandmarks = landmarks;
      setBufferFill(100);
    }

    lastRequestTime.current = now;
    pendingRequest.current = true;

    try {
      const start = performance.now();
      const result = await recognize(sessionId, payloadLandmarks, mode, now);
      const end = performance.now();

      setLatency(Math.round(end - start));
      setPrediction(result);
      setStatus(result.status || 'idle');
    } catch (err) {
      setStatus('error');
      console.error('Recognition error:', err);
    } finally {
      pendingRequest.current = false;
    }
  }, [sessionId]);

  const reset = useCallback(() => {
    sequenceBuffer.current = [];
    setBufferFill(0);
    setPrediction(null);
    setStatus('idle');
    setLatency(0);
  }, []);

  const clearBuffer = useCallback(() => {
    sequenceBuffer.current = [];
    setBufferFill(0);
  }, []);

  return {
    prediction,
    status,
    latency,
    bufferFill,
    sendLandmarks: pushFrame,
    reset,
    clearBuffer,
  };
}
