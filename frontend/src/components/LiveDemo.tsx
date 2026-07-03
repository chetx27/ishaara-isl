import React, { useEffect, useRef, useState } from 'react';
import './LiveDemo.css';

interface PredictionMessage {
  status: 'success' | 'error';
  sign?: string;
  confidence?: number;
  message?: string;
}

export const LiveDemo: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [currentSign, setCurrentSign] = useState<string>('LISTENING...');
  const [confidence, setConfidence] = useState<number>(0);
  const [captionLog, setCaptionLog] = useState<string[]>([]);
  const [errorState, setErrorState] = useState<string | null>(null);

  useEffect(() => {
    // Initialize WebSocket
    wsRef.current = new WebSocket('ws://localhost:8000/ws/inference');
    
    wsRef.current.onmessage = (event) => {
      const data: PredictionMessage = JSON.parse(event.data);
      
      if (data.status === 'success' && data.sign) {
        if (data.sign === 'NOT_CONFIDENT') {
          setCurrentSign('LOW CONFIDENCE');
          setConfidence(data.confidence || 0);
          setErrorState('MODEL UNCERTAIN');
        } else {
          setCurrentSign(data.sign.toUpperCase());
          setConfidence(data.confidence || 0);
          setErrorState(null);
          setCaptionLog(prev => [...prev.slice(-4), data.sign!.toUpperCase()]);
        }
      } else if (data.status === 'error') {
        setErrorState(data.message || 'UNKNOWN ERROR');
      }
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setHasPermission(true);
        setErrorState(null);
      } catch (err) {
        setHasPermission(false);
        setErrorState('CAMERA PERMISSION DENIED');
      }
    };

    startCamera();

    const captureFrame = () => {
      if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        const context = canvasRef.current.getContext('2d');
        if (context && videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
          context.drawImage(videoRef.current, 0, 0, 640, 480);
          const frameBase64 = canvasRef.current.toDataURL('image/jpeg', 0.5);
          wsRef.current.send(JSON.stringify({ frame: frameBase64 }));
        }
      }
      // Process at ~15 fps to reduce load
      setTimeout(() => {
        animationFrameId = requestAnimationFrame(captureFrame);
      }, 1000 / 15);
    };

    if (hasPermission) {
      captureFrame();
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [hasPermission]);

  if (hasPermission === false) {
    return (
      <div className="demo-container error-container">
        <h2>ERROR STATE</h2>
        <p>CAMERA PERMISSION DENIED. PLEASE ENABLE ACCESS.</p>
      </div>
    );
  }

  return (
    <div className="demo-container">
      <div className="video-wrapper">
        <video ref={videoRef} autoPlay playsInline muted className="live-video" />
        <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
        
        {/* Overlay UI */}
        <div className="overlay-header">
          <div className="status-indicator">
            <span className="dot animate-pulse"></span> LIVE
          </div>
          {errorState && (
            <div className="error-badge">
              {errorState}
            </div>
          )}
        </div>

        <div className="overlay-footer">
          <div className="current-sign">
            <h3>{currentSign}</h3>
            {currentSign !== 'LISTENING...' && currentSign !== 'LOW CONFIDENCE' && (
              <span className="confidence-meter">
                CONFIDENCE: {(confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="caption-log">
        <h3>TRANSCRIPTION LOG</h3>
        <div className="log-entries">
          {captionLog.length === 0 ? (
            <span className="empty-log">NO SIGNS DETECTED YET</span>
          ) : (
            captionLog.map((caption, idx) => (
              <div key={idx} className="log-entry">{caption}</div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
