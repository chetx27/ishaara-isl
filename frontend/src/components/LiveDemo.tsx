import React, { useEffect, useRef, useState } from 'react';
import './LiveDemo.css';

interface PredictionMessage {
  status: 'success' | 'error' | 'tracking';
  sign?: string;
  confidence?: number;
  message?: string;
  landmarks?: [number, number][];
}

export const LiveDemo: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null); // For capture
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null); // For drawing landmarks
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
      
      if (data.status === 'tracking' && data.landmarks && overlayCanvasRef.current) {
        const ctx = overlayCanvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, 640, 480);
          ctx.fillStyle = '#33ff33';
          ctx.strokeStyle = '#33ff33';
          ctx.lineWidth = 2;
          
          const drawHand = (offset: number) => {
            const connections = [
              [0,1], [1,2], [2,3], [3,4],
              [0,5], [5,6], [6,7], [7,8],
              [5,9], [9,10], [10,11], [11,12],
              [9,13], [13,14], [14,15], [15,16],
              [13,17], [17,18], [18,19], [19,20],
              [0,17]
            ];
            
            // Draw lines
            connections.forEach(([i, j]) => {
              const p1 = data.landmarks![offset + i];
              const p2 = data.landmarks![offset + j];
              if (p1 && p2 && p1[0] >= 0 && p2[0] >= 0) {
                ctx.beginPath();
                ctx.moveTo(p1[0] * 640, p1[1] * 480);
                ctx.lineTo(p2[0] * 640, p2[1] * 480);
                ctx.stroke();
              }
            });
            
            // Draw points
            for (let i = 0; i < 21; i++) {
              const p = data.landmarks![offset + i];
              if (p && p[0] >= 0) {
                ctx.beginPath();
                ctx.arc(p[0] * 640, p[1] * 480, 3, 0, 2 * Math.PI);
                ctx.fill();
              }
            }
          };
          
          // Draw left hand
          drawHand(0);
          // Draw right hand
          if (data.landmarks.length > 21) {
             drawHand(21);
          }
        }
      } else if (data.status === 'success' && data.sign) {
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
        // Clear mapping when hand drops
        if (overlayCanvasRef.current) {
          const ctx = overlayCanvasRef.current.getContext('2d');
          if (ctx) ctx.clearRect(0, 0, 640, 480);
        }
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

  const [translatedText, setTranslatedText] = useState<string | null>(null);

  const handleTranslate = async () => {
    if (captionLog.length === 0) return;
    try {
      const response = await fetch('http://localhost:8000/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glosses: captionLog })
      });
      const data = await response.json();
      setTranslatedText(data.english);
    } catch (err) {
      console.error('Translation error:', err);
      setTranslatedText('Error translating sequence.');
    }
  };

  const handleClear = () => {
    setCaptionLog([]);
    setTranslatedText(null);
    setCurrentSign('LISTENING...');
  };

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
        <canvas ref={overlayCanvasRef} width="640" height="480" className="overlay-canvas" />
        
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

      <div className="transcription-panel">
        <div className="caption-log">
          <div className="log-header">
            <h3>DETECTED SIGNS (GLOSS)</h3>
            <button onClick={handleClear} className="clear-btn">CLEAR</button>
          </div>
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
        
        <div className="translation-box">
          <button onClick={handleTranslate} className="translate-btn" disabled={captionLog.length === 0}>
            TRANSLATE TO ENGLISH
          </button>
          <div className="translation-output">
            {translatedText ? (
              <p className="translated-text">{translatedText}</p>
            ) : (
              <p className="empty-translation">Waiting for translation...</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
