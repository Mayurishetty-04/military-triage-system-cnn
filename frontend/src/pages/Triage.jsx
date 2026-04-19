import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

function TriageApp() {
  const navigate = useNavigate();
  const [image, setImage] = useState(null);
  const [audio, setAudio] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [pulse, setPulse] = useState("");
  const [spo2, setSpo2] = useState("");
  const [systolicBP, setSystolicBP] = useState("");
  const [unconscious, setUnconscious] = useState(false);
  const [liveVitals, setLiveVitals] = useState(null);
  const [currentUsername, setCurrentUsername] = useState("");
  const [currentPatientId, setCurrentPatientId] = useState("");

  // Decode JWT to get logged-in username
  useEffect(() => {
    const token = localStorage.getItem("token");
    const storedPatientId = localStorage.getItem("patientId");
    
    if (storedPatientId) {
      setCurrentPatientId(storedPatientId);
    }

    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setCurrentUsername(payload.sub || payload.username || "");
      } catch (e) {
        // token not decodable, ignore
      }
    }
  }, []);

  // Poll for live vitals from Android Bridge
  useEffect(() => {
    const pollVitals = async () => {
      try {
        // Use hostname to work across devices on the same network
        const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
        const res = await axios.get(`http://${backendHost}:8000/latest-vitals`);

        if (res.data && res.data.timestamp) {
          console.log("Live vitals received:", res.data);
          setLiveVitals(res.data);
          // Auto-fill form if vitals are fresh (e.g., within last 30s)
          const pingTime = new Date(res.data.timestamp).getTime();
          const now = new Date().getTime();
          if (now - pingTime < 30000) {
            if (!pulse) setPulse(res.data.heart_rate);
            if (!spo2) setSpo2(res.data.spo2);
            if (!systolicBP) setSystolicBP(res.data.systolic_bp);
          }
        }
      } catch (err) {
        // Silent fail for polling errors to avoid UI noise
      }
    };

    const interval = setInterval(pollVitals, 3000);
    return () => clearInterval(interval);
  }, [pulse, spo2, systolicBP]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Camera management
  const startCamera = () => setCameraOn(true);
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  };

  useEffect(() => {
    if (!cameraOn) return;

    const startStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
        alert(err.name === "NotAllowedError" ? "Camera permission blocked." : "Camera error: " + err.message);
      }
    };

    startStream();
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const video = videoRef.current;
      if (video?.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, [cameraOn]);

  // Blur detection
  const isBlurry = (canvas) => {
    const ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let gray = [];
    const data = imgData.data;

    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      gray.push(avg);
    }

    let laplacian = [];
    const w = canvas.width;

    for (let i = w; i < gray.length - w; i++) {
      const val = -4 * gray[i] + gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w];
      laplacian.push(val);
    }

    const mean = laplacian.reduce((a, b) => a + b, 0) / laplacian.length;
    const variance = laplacian.reduce((a, b) => a + (b - mean) ** 2, 0) / laplacian.length;
    return variance < 120;
  };

  // Capture image
  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      alert("Camera not ready. Please wait.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (isBlurry(canvas)) {
      alert("⚠️ Image is blurry. Please retake.");
      return;
    }

    canvas.toBlob((blob) => {
      if (!blob || blob.size === 0) {
        alert("Capture failed. Try again.");
        return;
      }
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      setImage(file);
      stopCamera();
      alert("✅ Photo Captured Successfully");
    }, "image/jpeg", 0.95);
  };

  // File handlers
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };

  const handleAudioUpload = (e) => {
    const file = e.target.files[0];
    if (file) setAudio(file);
  };

  // Send data for analysis
  const sendData = async () => {
    if (!image && !audio && !text) {
      alert("Please upload image, audio, or enter description");
      return;
    }

    const formData = new FormData();
    if (image) formData.append("image", image);
    if (audio) formData.append("audio", audio);
    if (text) formData.append("text", text);
    if (pulse) formData.append("pulse", pulse);
    if (spo2) formData.append("spo2", spo2);
    if (systolicBP) formData.append("systolic_bp", systolicBP);
    formData.append("unconscious", unconscious);

    // Capture location before sending
    if ("geolocation" in navigator) {
      try {
        const position = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000 });
        });
        formData.append("latitude", position.coords.latitude);
        formData.append("longitude", position.coords.longitude);
      } catch (err) {
        console.warn("Location capture failed, proceeding without coordinates:", err);
      }
    }

    try {
      setLoading(true);
      const res = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
      });
      setResult(res.data);
      if (res.data.patientId) setCurrentPatientId(res.data.patientId);
    } catch (err) {
      console.error(err);
      alert("Backend error. Is server running?");
    } finally {
      setLoading(false);
    }
  };

  // PDF download
  const downloadPDF = async () => {
    if (!result) return;
    try {
      const backendHost = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
      const res = await axios.post(`http://${backendHost}:8000/download-report`, result, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
          "Content-Type": "application/json"
        },
        responseType: "blob"
      });
      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `triage_report_${result.patientId || 'patient'}.pdf`);
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
    } catch (err) {
      console.error("PDF download failed", err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("patientId");
    navigate("/");
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-black flex items-center justify-center overflow-hidden relative">
      {/* Animated Background Blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/3 left-1/4 w-72 h-72 bg-amber-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '0.5s' }}></div>
      </div>

      {/* Scroll wrapper for large content */}
      <div className="relative w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-h-[90vh] overflow-y-auto">

        {/* Header with Logout Button */}
        <div className="text-center mb-10 animate-fade-in relative">
          <button
            onClick={handleLogout}
            className="absolute top-0 right-0 px-4 py-2 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white font-bold rounded-lg shadow-lg transform hover:scale-105 transition-all duration-300 text-sm sm:text-base flex items-center gap-2"
          >
            🚪 Logout
          </button>
          <div className="inline-block mb-4 text-6xl sm:text-7xl drop-shadow-lg">🪖</div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 mb-2">
            Military Triage System
          </h1>
          <p className="text-gray-300 text-sm sm:text-base font-light">Multi-Modal Injury Assessment</p>
          {currentUsername && (
            <div className="mt-3 inline-flex items-center gap-2 bg-slate-700/60 border border-slate-600/50 rounded-full px-4 py-1.5">
              <span className="text-amber-400 text-sm font-semibold">👤 {currentUsername}</span>
              {currentPatientId && (
                <>
                  <span className="text-slate-500 text-sm">|</span>
                  <span className="text-slate-300 text-sm font-mono">{currentPatientId}</span>
                </>
              )}
            </div>
          )}
          <div className="h-1 w-20 bg-gradient-to-r from-amber-400 to-cyan-400 mx-auto rounded-full mt-4"></div>
        </div>

        {/* Main Card */}
        <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl border border-slate-700/50 p-6 sm:p-8 lg:p-10 space-y-8">

          {/* IMAGE SECTION */}
          <div className="space-y-4 animate-slide-in-left" style={{ animationDelay: '0.1s' }}>
            <h3 className="text-xl sm:text-2xl font-bold text-amber-400 flex items-center gap-2 uppercase tracking-wide">🖼 Injury Image</h3>

            {!cameraOn && (
              <button
                onClick={startCamera}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold rounded-lg shadow-lg transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
              >
                📷 Open Camera
              </button>
            )}

            {cameraOn && (
              <div className="space-y-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full rounded-xl shadow-xl border border-cyan-500/30 aspect-video object-cover"
                />
                <div className="flex gap-3 flex-wrap">
                  <button
                    onClick={captureImage}
                    className="flex-1 min-w-fit px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-bold rounded-lg shadow-lg transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    📸 Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className="flex-1 min-w-fit px-6 py-3 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white font-bold rounded-lg shadow-lg transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    ❌ Stop
                  </button>
                </div>
              </div>
            )}

            <div className="space-y-2">
              <label className="block text-gray-300 font-semibold uppercase text-sm tracking-wide">📁 Upload Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="block w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-amber-500 file:text-gray-900 hover:file:bg-amber-600 transition-all cursor-pointer"
              />
            </div>

            {image && (
              <div className="p-3 bg-green-500/20 border border-green-500/50 rounded-lg flex items-center gap-2 text-green-200 animate-pulse font-semibold">
                ✅ Image Ready
              </div>
            )}
            <canvas ref={canvasRef} hidden />
          </div>

          {/* AUDIO SECTION */}
          <div className="space-y-4 border-t border-slate-700/50 pt-8 animate-slide-in-right" style={{ animationDelay: '0.2s' }}>
            <h3 className="text-xl sm:text-2xl font-bold text-cyan-400 flex items-center gap-2 uppercase tracking-wide">🎤 Injury Audio</h3>
            <div className="space-y-2">
              <label className="block text-gray-300 font-semibold uppercase text-sm tracking-wide">Upload Audio File</label>
              <input
                type="file"
                accept="audio/*"
                onChange={handleAudioUpload}
                className="block w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-cyan-500 file:text-gray-900 hover:file:bg-cyan-600 transition-all cursor-pointer"
              />
            </div>
            {audio && (
              <div className="p-3 bg-cyan-500/20 border border-cyan-500/50 rounded-lg flex items-center gap-2 text-cyan-200 animate-pulse font-semibold">
                ✅ Audio Uploaded
              </div>
            )}
          </div>

          {/* TEXT SECTION */}
          <div className="space-y-4 border-t border-slate-700/50 pt-8 animate-slide-in-left" style={{ animationDelay: '0.3s' }}>
            <h3 className="text-xl sm:text-2xl font-bold text-yellow-400 flex items-center gap-2 uppercase tracking-wide">📝 Injury Description</h3>
            <textarea
              rows="4"
              className="w-full px-4 py-3 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-yellow-400 transition-all duration-300 resize-none font-medium"
              value={text}
              placeholder="Describe the injury, symptoms, location, and medical history..."
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          {/* VITALS SECTION */}
          <div className="space-y-4 border-t border-slate-700/50 pt-8 animate-slide-in-right" style={{ animationDelay: '0.4s' }}>
            {/* LIVE VITALS DISPLAY (From Watch) */}
            <div className="space-y-4 border-t border-slate-700/50 pt-8 animate-fade-in">
              <div className="flex justify-between items-center">
                <h3 className="text-xl sm:text-2xl font-bold text-emerald-400 flex items-center gap-2 uppercase tracking-wide">
                  📡 Live Watch Vitals
                </h3>
                {liveVitals && liveVitals.timestamp ? (
                  <span className="flex items-center gap-2 px-3 py-1 bg-emerald-500/20 text-emerald-300 text-xs font-bold rounded-full animate-pulse border border-emerald-500/30">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full"></span> LIVE
                  </span>
                ) : (
                  <span className="flex items-center gap-2 px-3 py-1 bg-slate-700/50 text-gray-400 text-[10px] font-bold rounded-full border border-slate-600/30">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-ping"></div> WAITING FOR BRIDGE...
                  </span>
                )}
              </div>

              {liveVitals && liveVitals.timestamp ? (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-slate-700/40 p-4 rounded-xl border border-slate-600/30 text-center text-sm sm:text-base">
                      <p className="text-gray-400 text-[10px] uppercase mb-1">Heart Rate</p>
                      <p className="text-xl font-black text-white">{liveVitals.heart_rate} <span className="text-[10px] font-normal text-gray-400">bpm</span></p>
                    </div>
                    <div className="bg-slate-700/40 p-4 rounded-xl border border-slate-600/30 text-center text-sm sm:text-base">
                      <p className="text-gray-400 text-[10px] uppercase mb-1">SpO2</p>
                      <p className="text-xl font-black text-white">{liveVitals.spo2} <span className="text-[10px] font-normal text-gray-400">%</span></p>
                    </div>
                    <div className="bg-slate-700/40 p-4 rounded-xl border border-slate-600/30 text-center text-sm sm:text-base">
                      <p className="text-gray-400 text-[10px] uppercase mb-1">BP (Sys/Dia)</p>
                      <p className="text-lg font-black text-white">{liveVitals.systolic_bp}/{liveVitals.diastolic_bp}</p>
                    </div>
                    <div className={`p-4 rounded-xl border text-center font-bold flex flex-col justify-center text-sm sm:text-base ${liveVitals.triage === 'RED' ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse' :
                      liveVitals.triage === 'YELLOW' ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400' :
                        liveVitals.triage === 'BLACK' ? 'bg-gray-800 border-gray-600 text-gray-300' :
                          'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                      }`}>
                      <p className="text-[10px] uppercase opacity-80 mb-1">Triage</p>
                      <p className="text-lg">{liveVitals.triage}</p>
                    </div>
                  </div>
                  <p className="text-[10px] text-gray-500 text-right italic">
                    Last updated: {new Date(liveVitals.timestamp).toLocaleTimeString()}
                  </p>
                </>
              ) : (
                <div className="bg-slate-800/30 border border-dashed border-slate-700 rounded-xl p-6 text-center">
                  <p className="text-gray-500 text-sm font-medium italic">
                    Waiting for data from the Android Bridge app...
                  </p>
                  <p className="text-[10px] text-gray-600 mt-2">
                    (Ensure your phone is on this WiFi and you've tapped "Start Sending")
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* VITALS SECTION (Manual Override) */}
          <div className="space-y-4 border-t border-slate-700/50 pt-8 animate-slide-in-right" style={{ animationDelay: '0.4s' }}>
            <h3 className="text-xl sm:text-2xl font-bold text-red-400 flex items-center gap-2 uppercase tracking-wide">❤️ {liveVitals && liveVitals.timestamp ? 'Manual Vitals Override' : 'Vital Signs (Optional)'}</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <input
                type="number"
                placeholder="Pulse (bpm)"
                value={pulse}
                onChange={(e) => setPulse(e.target.value)}
                className="px-4 py-3 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-red-400 transition-all duration-300 font-medium"
              />
              <input
                type="number"
                placeholder="SpO2 (%)"
                value={spo2}
                onChange={(e) => setSpo2(e.target.value)}
                className="px-4 py-3 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-red-400 transition-all duration-300 font-medium"
              />
              <input
                type="number"
                placeholder="Systolic BP (mmHg)"
                value={systolicBP}
                onChange={(e) => setSystolicBP(e.target.value)}
                className="px-4 py-3 rounded-lg bg-slate-700/50 border border-slate-600/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-red-400 transition-all duration-300 font-medium"
              />
            </div>
            <label className="flex items-center gap-3 px-4 py-3 bg-slate-700/30 rounded-lg border border-slate-600/30 cursor-pointer hover:bg-slate-700/50 transition-colors duration-300">
              <input
                type="checkbox"
                checked={unconscious}
                onChange={(e) => setUnconscious(e.target.checked)}
                className="w-5 h-5 rounded cursor-pointer"
              />
              <span className="text-gray-200 font-semibold">Patient Unconscious</span>
            </label>
          </div>

          {/* ANALYZE BUTTON */}
          <button
            onClick={sendData}
            disabled={loading}
            className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold text-lg rounded-xl shadow-2xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2 border-t border-slate-700/50 pt-8 mt-4 uppercase tracking-wide"
          >
            {loading ? (
              <>
                <div className="animate-spin h-6 w-6 border-2 border-white border-t-transparent rounded-full"></div>
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <span>🔍</span>
                <span>Analyze Injury</span>
              </>
            )}
          </button>

          {/* RESULT SECTION */}
          {result && (
            <div className="space-y-6 border-t border-slate-700/50 pt-8 mt-8 animate-fade-in">
              <h3 className="text-2xl sm:text-3xl font-bold text-white flex items-center gap-2 uppercase tracking-wide">🩺 Analysis Result</h3>

              {/* Triage Level Banner */}
              <div
                className={`p-6 rounded-2xl text-center font-bold text-2xl sm:text-3xl shadow-2xl border-2 ${result.triage_level === "Black"
                  ? "bg-gradient-to-r from-gray-700 to-gray-900 border-gray-600 text-gray-100"
                  : result.triage_level === "Red"
                    ? "bg-gradient-to-r from-red-700 to-red-900 border-red-600 text-white animate-pulse"
                    : result.triage_level === "Yellow"
                      ? "bg-gradient-to-r from-yellow-600 to-amber-700 border-yellow-600 text-gray-900"
                      : "bg-gradient-to-r from-green-600 to-emerald-700 border-green-600 text-white"
                  }`}
              >
                TRIAGE LEVEL: {result.triage_level}
              </div>

              {/* Patient ID + Name */}
              {result.patientId && (
                <div className="flex items-center gap-3 bg-slate-700/40 border border-slate-600/40 rounded-xl px-5 py-3">
                  <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Patient</span>
                  <span className="text-white font-bold">{result.patientId}</span>
                  {result.patientName && (
                    <span className="text-amber-400 font-semibold ml-1">· 👤 {result.patientName}</span>
                  )}
                </div>
              )}

              {/* Confidence */}
              <div className="bg-slate-700/30 p-6 rounded-xl border border-slate-600/50">
                <p className="text-lg text-gray-200">
                  <span className="font-bold text-white">Overall Confidence:</span>
                  <span className="ml-3 text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
                    {result.confidence}%
                  </span>
                </p>
              </div>

              {/* Override Reason */}
              {result.override_reason && (
                <div className="p-4 bg-red-500/20 border-2 border-red-500/50 rounded-lg flex items-start gap-3 animate-pulse">
                  <span className="text-2xl mt-1">⚠️</span>
                  <p className="text-red-200 font-semibold">{result.override_reason}</p>
                </div>
              )}

              {/* Recommended Actions */}
              {result.recommended_action && (
                <div className="space-y-3">
                  <h4 className="text-xl sm:text-2xl font-bold text-lime-400 flex items-center gap-2 uppercase tracking-wide">💡 Recommended Actions</h4>
                  <ul className="bg-lime-500/20 border-2 border-lime-500/50 rounded-lg p-6 space-y-3">
                    {result.recommended_action.map((item, i) => (
                      <li key={i} className="text-lime-200 font-semibold flex items-start gap-3">
                        <span className="text-lg mt-1">→</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Probability Bars */}
              <div className="space-y-4">
                <h4 className="text-xl sm:text-2xl font-bold text-blue-400 flex items-center gap-2 uppercase tracking-wide">📊 Integrated Triage Probability</h4>
                {Object.entries(result.probabilities).map(([k, v]) => {
                  const colors = {
                    Black: "from-gray-600 to-gray-700",
                    Red: "from-red-600 to-red-700",
                    Yellow: "from-yellow-500 to-amber-600",
                    Green: "from-green-600 to-emerald-700"
                  };
                  return (
                    <div key={k} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-base sm:text-lg text-gray-200">{k}</span>
                        <span className="text-base sm:text-lg font-bold text-blue-300">{(v * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-3 bg-slate-700/50 rounded-full overflow-hidden border border-slate-600/50">
                        <div
                          className={`h-full bg-gradient-to-r ${colors[k] || 'from-blue-600 to-cyan-600'} transition-all duration-500`}
                          style={{ width: `${v * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Download Button */}
              <button
                onClick={downloadPDF}
                className="w-full px-6 py-4 bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white font-bold rounded-xl shadow-lg transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2 text-base sm:text-lg uppercase tracking-wide"
              >
                📥 Download PDF Report
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TriageApp;
