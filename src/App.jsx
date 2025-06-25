import { useState, useRef, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import { ErrorBoundary } from 'react-error-boundary'
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/hand-pose-detection';
import '@tensorflow/tfjs-backend-webgl';

function ErrorFallback({ error }) {
  return (
    <div>
      <h2>Something went wrong</h2>
      <p>{error.message}</p>
    </div>
  )
}

function App() {
  const [detector, setDetector] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isPinching, setIsPinching] = useState(false);
  const [handPosition, setHandPosition] = useState({ x: 0, y: 0 });
  const [handVisible, setHandVisible] = useState(false);
  const [status, setstatus] = useState(null)
  const [hands, setHands] = useState([]);
  const [SMOOTHER, setSMOOTHER] = useState(1)
  const positionHistory = [];
  const MOVEMENT_THRESHOLD = 300; // Maximum allowed movement in pixels (if exceeded, don't move)
  const CLICK_COOLDOWN = 1000; // 2s between allowed clicks
  const pinchThreshold = 20;
  const sendRef = useRef(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef();
  const fixhand = useRef();
  let SMOOTHING_WINDOW = 10;
  let lastPosition = null; // Store the last position
  let lastClickTimeL = 0; // for deference between lastTime and currenTime
  let lastClickTimeR = 0; // for deference between lastTime and currenTime

  // 1. Laods Model
  useEffect(() => {
    setstatus("Loading");
    const initDetector = async () => {
      try {
        await tf.setBackend('webgl');
        const detector = await handpose.createDetector(
          handpose.SupportedModels.MediaPipeHands,
          {
            runtime: 'tfjs',
            modelType: 'lite',
            maxHands: 1
          }
        );
        setDetector(detector);
        // setLoading(false);
      } catch (err) {
        console.error('Detector init error:', err);
        setstatus("Detector init error")
        // setLoading(false);
      }
    };
    initDetector();
    return () => detector?.dispose();
  }, []);

  // 2. Start Processing
  useEffect(() => {
    setstatus("Model loading")
    requestRef.current = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(requestRef.current);
  }, [detector]);

  // 3. Frame processing pipeline
  const processFrame = async () => {

    if (!webcamRef.current || !detector) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    try {
      // STEP 1: Capture and validate frame
      const video = webcamRef.current.video;
      if (!webcamRef.current?.video ||
        !webcamRef.current.video.videoWidth ||
        !detector) {
        console.log('Video stream not ready');
        setstatus("Video stream not ready")
        requestRef.current = requestAnimationFrame(processFrame);
        return;
      }

      if (video.readyState !== 4) { // HTMLMediaElement.HAVE_ENOUGH_DATA
        setstatus("Waiting for video data...")
        console.log('Waiting for video data...');
        requestRef.current = requestAnimationFrame(processFrame);
        return;
      }

      // STEP 2: Create processing canvas
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // STEP 3: Convert to tensor with validation
      const tensor = tf.browser.fromPixels(canvas);
      if (tensor.shape.length !== 3 || tensor.shape[2] !== 3) {
        tensor.dispose();
        throw new Error('Invalid tensor shape');
      }

      // STEP 4: Run detection with NaN protection
      const hands = await detector.estimateHands(tensor);
      const validatedHands = hands.map(hand => ({
        ...hand,
        score: Math.max(0, Math.min(1, hand.score || 0)), // Clamp 0-1
        keypoints: hand.keypoints.map(kp => ({
          ...kp,
          x: Number(kp.x) || 0,
          y: Number(kp.y) || 0,
          z: Number(kp.z) || 0,
          score: Math.max(0.1, Math.min(1, kp.score || 0.5)) // Clamp 0.1-1
        }))
      }));

      // STEP 5: Update state and clean up
      if (validatedHands.length > 0) {
        //console.log("Valid Hands");
        
        const wrist = validatedHands[0].keypoints.find(kp => kp.name === 'index_finger_tip');
        const thumbTip = validatedHands[0].keypoints.find(kp => kp.name === 'thumb_tip'); // Thumb tip
        const middleTip = validatedHands[0].keypoints.find(kp => kp.name === 'middle_finger_tip');

        if (wrist && wrist.score >= 0.5) {

          const distance = Math.sqrt(
            Math.pow(thumbTip.x - wrist.x, 2) +
            Math.pow(thumbTip.y - wrist.y, 2)
          );
          
          const middle_distance = Math.sqrt(
            Math.pow(middleTip.x - thumbTip.x, 2) +
            Math.pow(middleTip.y - thumbTip.y, 2)
          );
          
          if (fixhand.current) {
            setLoading(false)
            const { x, y } = await mapHandToScreen(wrist.x, wrist.y, video)

            if(distance > 50 && middle_distance > 50) {
              SMOOTHING_WINDOW = 2 - SMOOTHER;
              //console.log("Not smoothing:", send);
            } else {
              //console.log("smoothing:", send);
              SMOOTHING_WINDOW = 15 + SMOOTHER;
            }
            sendCursorPosition(x, y, validatedHands);
          }

          // Detect pinch gesture
          if (distance < pinchThreshold) {
            if (!isPinching) {
              handleLeftClick(); // Trigger click on pinch start
              setIsPinching(true);
            }
          } else {
            setIsPinching(false);
          }

          if (middle_distance < pinchThreshold) {
            if (!isPinching) {
              handleRightClick(); // Trigger click on pinch start
              setIsPinching(true);
            }
          } else {
            setIsPinching(false);
          }

          setHandPosition({ x: wrist.x, y: wrist.y });
          setHandVisible(true);
          // sendCursorPosition(679, 388, validatedHands);
        } else {
          setstatus("Hands not found")
          setHandVisible(false);
        }
      } else {
        setstatus("Hands not found")
        setHandVisible(false);
      }
      tensor.dispose();

      // STEP 6: Visualize results
      if (canvasRef.current) {
        const outputCtx = canvasRef.current.getContext('2d');
        outputCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        validatedHands.forEach(hand => {
          hand.keypoints.forEach(kp => {
            if (kp.score > 0.3) { // Only show confident points
              outputCtx.fillStyle = `hsl(${kp.score * 120}, 100%, 50%)`;
              outputCtx.beginPath();
              outputCtx.arc(kp.x, kp.y, 5 * kp.score, 0, Math.PI * 2);
              outputCtx.fill();
            }
          });
        });
      }
    } catch (err) {
      setstatus("Precessing error")
      console.error('Processing error:', err);
    } finally {
      requestRef.current = requestAnimationFrame(processFrame);
    }
  };

  // 4. set video coordinate to screen
  const mapHandToScreen = async (fingerX, fingerY, videoElement) => {
    const videoRect = videoElement.getBoundingClientRect();
    const videoNativeWidth = videoElement.videoWidth;
    const videoNativeHeight = videoElement.videoHeight;

    // Calculate content area (excluding black bars)
    const videoAspect = videoNativeWidth / videoNativeHeight;
    const displayAspect = videoRect.width / videoRect.height;

    let contentWidth, contentHeight, offsetX, offsetY;

    if (displayAspect > videoAspect) {
      contentHeight = videoRect.height;
      contentWidth = videoRect.height * videoAspect;
      offsetX = (videoRect.width - contentWidth) / 2;
      offsetY = 0;
    } else {
      contentWidth = videoRect.width;
      contentHeight = videoRect.width / videoAspect;
      offsetX = 0;
      offsetY = (videoRect.height - contentHeight) / 2;
    }

    // Clamp finger position to video content area
    const clampedX = Math.max(0, Math.min(videoNativeWidth, fingerX));
    const clampedY = Math.max(0, Math.min(videoNativeHeight, fingerY));

    // Convert to normalized coordinates (0-1)
    const normalizedX = clampedX / videoNativeWidth;
    const normalizedY = clampedY / videoNativeHeight;

    // Zone configuration
    const ACTIVE_ZONE_THICKNESS = 0.001; // 1% thick active area (very small)
    const HORIZONTAL_BOOST_FACTOR = 4.0; // 2x horizontal sensitivity in active zone
    const VERTICAL_BOOST_FACTOR = 5.0; // 3x vertical sensitivity in active zone
    const OUTER_SENSITIVITY = 0.01; // 50% sensitivity in outer area

    const mapCoordinates = (x, y) => {
      // Calculate active zone boundaries
      const innerLeft = ACTIVE_ZONE_THICKNESS;
      const innerRight = 1 - ACTIVE_ZONE_THICKNESS;
      const innerTop = ACTIVE_ZONE_THICKNESS;
      const innerBottom = 1 - ACTIVE_ZONE_THICKNESS;

      // Check if in active zone
      if (x > innerLeft && x < innerRight &&
        y > innerTop && y < innerBottom) {
        // Apply BOTH horizontal and vertical boost in active zone
        const centerX = 0.5;
        const centerY = 0.5;
        return {
          x: centerX + (x - centerX) * HORIZONTAL_BOOST_FACTOR,
          y: centerY + (y - centerY) * VERTICAL_BOOST_FACTOR
        };
      }

      // Outer area handling (no boost)
      let outerX = x;
      let outerY = y;

      // Horizontal mapping
      if (x <= innerLeft) {
        outerX = innerLeft * (x / innerLeft) * OUTER_SENSITIVITY;
      } else if (x >= innerRight) {
        outerX = innerRight + (x - innerRight) * OUTER_SENSITIVITY;
      }

      // Vertical mapping
      if (y <= innerTop) {
        outerY = innerTop * (y / innerTop) * OUTER_SENSITIVITY;
      } else if (y >= innerBottom) {
        outerY = innerBottom + (y - innerBottom) * OUTER_SENSITIVITY;
      }

      return { x: outerX, y: outerY };
    };

    // Apply mapping
    const { x: mappedX, y: mappedY } = mapCoordinates(normalizedX, normalizedY);

    // Convert to display coordinates
    const displayX = mappedX * contentWidth + offsetX;
    const displayY = mappedY * contentHeight + offsetY;

    // Get screen dimensions
    const screen = await getScreenDimensions();

    // Convert to screen coordinates
    const screenX = (displayX / videoRect.width) * screen.width;
    const screenY = (displayY / videoRect.height) * screen.height;

    // Apply smoothing
    positionHistory.push({ x: screenX, y: screenY });
    if (positionHistory.length > SMOOTHING_WINDOW) {
      positionHistory.shift();
    }

    const smoothedX = positionHistory.reduce((sum, pos) => sum + pos.x, 0) / positionHistory.length;
    const smoothedY = positionHistory.reduce((sum, pos) => sum + pos.y, 0) / positionHistory.length;

    // Clamp to screen bounds
    const newX = Math.min(Math.max(0, Math.floor(smoothedX)), screen.width - 1);
    const newY = Math.min(Math.max(0, Math.floor(smoothedY)), screen.height - 1);

    // If no last position, set it now
    if (!lastPosition) {
      lastPosition = { x: newX, y: newY };
      return { x: newX, y: newY };
    }

    // Calculate distance from last position
    const dx = newX - lastPosition.x;
    const dy = newY - lastPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // If distance exceeds threshold, keep old position
    if (distance > MOVEMENT_THRESHOLD) {
      return lastPosition;
    }

    // Otherwise, update last position and return new position
    lastPosition = { x: newX, y: newY };
    return { x: newX, y: newY };
  };

  // 5. Calculate screen dimensions
  const getScreenDimensions = async () => {
    try {
      // For Electron environment
      if (window.screen) {
        //const { width, height } = await window.electronAPI.screen.getPrimaryDisplay();
        const width = window.screen.width;
        const height = window.screen.height;
        //console.log("Dimension:", {width, height})
        return { width, height };
      }
      // For browser development fallback
      return { width: window.innerWidth, height: window.innerHeight };
    } catch (error) {
      console.warn('Screen dimensions error:', error);
      return { width: 1920, height: 1080 }; // Default fallback
    }
  };

  // 6. Send cursor position to electron
  const sendCursorPosition = (x, y) => {
    console.log("Inside Function send=",sendRef.current)
    if (window.electronAPI && sendRef.current) {
        window.electronAPI.sendCursorPosition(x, y);
      } else {
        //console.log('[IPC Simulated] move-cursor', { x, y });
      }
  }

  // 7. Trigger left click
  const handleLeftClick = () => {
    const now = Date.now();
    if (now - lastClickTimeL < CLICK_COOLDOWN) return;

    if (window.electronAPI && sendRef.current) {
      lastClickTimeL = now;
      playClickSound();
      window.electronAPI.ipcRenderer.send('left-click');
    } else {
      // Fallback for browser testing
      //const el = document.elementFromPoint(handX, handY);
      //if (el) el.click();
    }
  };

  // 8. Trigger right click
  const handleRightClick = () => {
   const now = Date.now();
    if (now - lastClickTimeR < CLICK_COOLDOWN) return;

    if (window.electronAPI && sendRef.current) {
      lastClickTimeR = now;
      playClickSound();
      window.electronAPI.ipcRenderer.send('right-click');
    } else {
      // Browser fallback
      //const el = document.elementFromPoint(handX, handY);
      //if (el) {
        //const event = new MouseEvent('contextmenu', {
         // bubbles: true,
         // clientX: handX,
         // clientY: handY
        //});
       // el.dispatchEvent(event);
      //}
    }
  };

  // 9. Play click sound
  function playClickSound() {
    const clickSound = document.getElementById('clickSound');
    clickSound.currentTime = 0; // Rewind to start (allows rapid playback)
    clickSound.play().catch(e => console.log("Audio play failed:", e));
  }

  // 10(Optional). If hand disappear than send cursor to center
  useEffect(() => {
    let timeoutId;
    
    if (handVisible) {
      if (handPosition) {
        timeoutId = setTimeout(() => {
          console.log("1s Complete");
          sendRef.current = true;
      }, 1000);
        fixhand.current = {
          x: handPosition.x,
          y: handPosition.y
        };
      }
    } else {
      // When hand disappears, move cursor to (640,388) and deactivate
      if (timeoutId) clearTimeout(timeoutId);
      sendRef.current = false;
      if (window.electronAPI) {
        window.electronAPI.sendCursorPosition(640, 388);
      }
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [handVisible])

  return (
    <div style={{ position: 'relative', width: '640px', height: '480px' }}>
      <audio id="clickSound" src="click.wav" preload="auto"></audio>
      <div style={{display: 'flex', flexDirection: 'column', fontWeight: 'bold'}}>
        <p>To Quit: Ctrl+Q</p>
        <p>To Force Quit: Ctrl+Shift+Q</p>
      </div>
      <button onClick={() => {
        window.electronAPI.toggleWindow();
      }}>
        hide window
      </button>
      <div style={{display: 'flex', flexDirection: 'column', fontWeight: 'bold', gap: '2px'}}>
        <p>Smoothness :</p>
      <button style={{width: '50px'}} onClick={() => {
        setSMOOTHER(SMOOTHER+1)
      }}>
        +
      </button>
      <button style={{width: '50px'}} onClick={() => {
        setSMOOTHER(SMOOTHER-1)
      }}>
        -
      </button>
      </div>
      {/* Movable div that follows your hand */}
      <div
        style={{
          position: 'absolute',
          width: '100px',
          height: '100px',
          backgroundColor: 'rgba(255, 0, 0, 0.5)',
          borderRadius: '50%',
          left: `${handPosition.x}px`,
          top: `${handPosition.y}px`,
          transform: 'translate(-50%, -50%)',
          transition: 'left 0.1s ease-out, top 0.1s ease-out',
          pointerEvents: 'none',
          display: handVisible ? 'block' : 'none',
          zIndex: 10
        }}
      >
        Follows Your Hand
      </div>

      {/* Visible Webcam Feed */}
      <Webcam
        ref={webcamRef}
        audio={false}
        videoConstraints={{
          width: 640,
          height: 480,
          facingMode: 'user'
        }}
        style={{
          display: 'block', // Make sure it's visible
          width: '100%',
          height: '100%',
          objectFit: 'cover'
        }}
      />

      {/* Transparent Overlay Canvas */}
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
      />

      {/* Debug Info */}
      <div style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        color: 'white',
        backgroundColor: 'rgba(0,0,0,0.7)',
        padding: '8px',
        borderRadius: '4px'
      }}>
        {loading ? status : (
          <>
            <div>Hands detected: {hands.length}</div>
            {hands.map((hand, i) => (
              <div key={i}>
                Hand {i + 1}: {hand.handedness} (score: {hand.score.toFixed(2)})
              </div>
            ))}
          </>
        )}
      </div>

    </div>
  );
}

export default App
