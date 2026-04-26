import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { BASE_URL } from "../config";
import ChatPanel from "../components/ChatPanel";
import useMessageNotifications from "../hooks/useMessageNotifications";

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
  const [isManualOverride, setIsManualOverride] = useState(false);
  const [currentUsername, setCurrentUsername] = useState("");
  const [currentPatientId, setCurrentPatientId] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [doctorId, setDoctorId] = useState(null);
  const [doctorName, setDoctorName] = useState("Doctor");
  const [doctorReady, setDoctorReady] = useState(false);

  // Decode JWT to get logged-in username and store user ID
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
        // Store user ID for chat integration
        if (payload.user_id) {
          localStorage.setItem("userId", payload.user_id);
        }

        // Fallback: if no patientId in localStorage, use username or a temp ID
        if (!storedPatientId) {
          const fallbackId = payload.patientId || payload.username || `TEMP-${Math.floor(Date.now() / 1000)}`;
          setCurrentPatientId(fallbackId);
        }
      } catch (e) {
        // token not decodable, ignore
      }
    }
  }, []);

  // Select an available doctor for chat
  useEffect(() => {
    const pickDoctor = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) return;

        const res = await axios.get(`${BASE_URL}/users/doctors`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (Array.isArray(res.data) && res.data.length > 0) {
          const preferredUsername = "doc1";
          const preferred = res.data.find((d) => d.username === preferredUsername);

          const savedDoctorIdRaw = localStorage.getItem("preferredDoctorId");
          const savedDoctorId = savedDoctorIdRaw ? Number(savedDoctorIdRaw) : null;
          const savedDoctorName = localStorage.getItem("preferredDoctorName");
          const savedStillExists = savedDoctorId
            ? res.data.some((d) => Number(d.id) === savedDoctorId)
            : false;

          // Always prefer the default doctor username when available; otherwise
          // keep the saved doctor if it still exists; else fall back to first doctor.
          const chosen =
            preferred ||
            (savedStillExists ? res.data.find((d) => Number(d.id) === savedDoctorId) : null) ||
            res.data[0];

          setDoctorId(chosen.id);
          setDoctorName(chosen.username || savedDoctorName || "Doctor");
          localStorage.setItem("preferredDoctorId", String(chosen.id));
          localStorage.setItem("preferredDoctorName", String(chosen.username || savedDoctorName || "Doctor"));
          setDoctorReady(true);

          // If we're using the configured default doctor, merge any old threads so history shows up.
          if (chosen.username === preferredUsername) {
            try {
              await axios.post(
                `${BASE_URL}/messages/merge-to-default-doctor`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
              );
            } catch (e) {
              // Non-fatal: chat will still work, but old history might remain in older threads.
            }
          }
        } else {
          setDoctorReady(false);
        }
      } catch (e) {
        // If unavailable, ChatPanel will fall back and show an error on send/fetch.
        setDoctorReady(false);
      }
    };
    pickDoctor();
  }, []);

  // Enable message notifications
  useMessageNotifications();

  // Detect manual override reset
  useEffect(() => {
    if (!pulse && !spo2 && !systolicBP) {
      // If all cleared, resume auto
      setIsManualOverride(false);
    }
  }, [pulse, spo2, systolicBP]);

  // Vitals Simulator (Background Process)
  useEffect(() => {
    if (!currentPatientId) return;

    const generateVitals = async () => {
      const simulatedData = {
        heart_rate: Math.floor(Math.random() * (140 - 80 + 1)) + 80,
        spo2: Math.floor(Math.random() * (100 - 85 + 1)) + 85,
        systolic_bp: Math.floor(Math.random() * (120 - 70 + 1)) + 70,
        diastolic_bp: Math.floor(Math.random() * (90 - 60 + 1)) + 60,
        patient_id: currentPatientId
      };

      try {
        await axios.post(`${BASE_URL}/realtime-vitals`, simulatedData);
      } catch (err) {
        console.error("Simulation error:", err);
      }
    };

    const timeoutId = setTimeout(generateVitals, 3000);
    const intervalId = setInterval(generateVitals, 180000);
    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
    };
  }, [currentPatientId]);

  // Poll for live vitals
  useEffect(() => {
    const pollVitals = async () => {
      try {
        const res = await axios.get(`${BASE_URL}/latest-vitals`, {
          params: { patient_id: currentPatientId }
        });

        if (res.data && res.data.timestamp) {
          setLiveVitals(res.data);

          // Auto-fill form if manual override is not active
          if (!isManualOverride) {
            setPulse(res.data.heart_rate);
            setSpo2(res.data.spo2);
            setSystolicBP(res.data.systolic_bp);
          }
        }
      } catch (err) {
        // Silent fail
      }
    };

    const timeoutId = setTimeout(pollVitals, 3000);
    const intervalId = setInterval(pollVitals, 180000);
    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
    };
  }, [currentPatientId, isManualOverride]);

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
      const res = await axios.post(`${BASE_URL}/predict`, formData, {
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
      const res = await axios.post(`${BASE_URL}/download-report`, result, {
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
    <div className="w-full min-h-screen bg-[#0f172a] text-white overflow-x-hidden font-['Inter'] relative">
      {/* Tactical Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-[60%] h-[60%] bg-blue-600/5 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-0 left-0 w-[60%] h-[60%] bg-cyan-600/5 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 pt-6 pb-32">
        {/* Top Navigation */}
        <nav className="flex justify-end items-center mb-8 animate-fade-in">
          <div className="flex gap-2">
            <button
              onClick={() => {
                if (!doctorId) {
                  alert("Connecting to field medic...");
                  return;
                }
                setChatOpen(true);
              }}
              disabled={!doctorReady}
              className={`px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${doctorReady
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 glow-on-hover'
                  : 'bg-white/5 text-gray-500 border border-white/5'
                }`}
            >
              💬 {doctorReady ? 'Live Medic' : 'Medic Offline'}
            </button>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-500/10 text-red-400 border border-red-500/20 rounded-xl text-xs font-black uppercase tracking-widest"
            >
              Logout
            </button>
          </div>
        </nav>

        {/* Header Section */}
        <header className="mb-10 text-center animate-fade-in">
          <div className="inline-block px-3 py-1 bg-cyan-500/10 border border-cyan-500/20 rounded-full text-[10px] font-black uppercase tracking-[0.3em] text-cyan-400 mb-4">
            Triage Module v4.2
          </div>
          <h1 className="text-4xl sm:text-5xl font-black font-['Outfit'] tracking-tight mb-2">
            Injury <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Assessment</span>
          </h1>
          {currentUsername && (
            <p className="text-gray-500 font-mono text-xs">Unit: {currentUsername} | ID: {currentPatientId || 'PENDING'}</p>
          )}
        </header>

        <div className="space-y-6">
          {/* LIVE VITALS MONITOR */}
          <section className="glass-panel rounded-[2rem] p-6 border-white/5 animate-fade-in">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-gray-400 flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></span>
                Biometric Stream
              </h3>
              <span className="text-[10px] font-black uppercase tracking-widest text-emerald-400 flex items-center gap-1 animate-pulse">
                👉 Live IoT Vitals Streaming Enabled
              </span>
              {liveVitals?.timestamp && (
                <span className="text-[10px] font-mono text-emerald-400/70">
                  {new Date(liveVitals.timestamp).toLocaleTimeString()}
                </span>
              )}
            </div>

            {liveVitals?.timestamp ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="glass-card p-4 rounded-2xl border-white/5">
                  <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">HR</p>
                  <p className="text-2xl font-black font-['Outfit']">{liveVitals.heart_rate}<span className="text-xs text-gray-600 ml-1">BPM</span></p>
                </div>
                <div className="glass-card p-4 rounded-2xl border-white/5">
                  <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">SpO2</p>
                  <p className="text-2xl font-black font-['Outfit']">{liveVitals.spo2}<span className="text-xs text-gray-600 ml-1">%</span></p>
                </div>
                <div className="glass-card p-4 rounded-2xl border-white/5">
                  <p className="text-[10px] text-gray-500 uppercase font-bold mb-1">BP</p>
                  <p className="text-2xl font-black font-['Outfit']">{liveVitals.systolic_bp}<span className="text-xs text-gray-600 ml-1">SYS</span></p>
                </div>
                <div className={`p-4 rounded-2xl flex flex-col justify-center border ${liveVitals.triage === 'RED' ? 'bg-red-500/20 border-red-500/30 text-red-400' :
                    liveVitals.triage === 'YELLOW' ? 'bg-amber-500/20 border-amber-500/30 text-amber-400' :
                      'bg-emerald-500/20 border-emerald-500/30 text-emerald-400'
                  }`}>
                  <p className="text-[10px] uppercase font-bold opacity-70 mb-1">Status</p>
                  <p className="text-lg font-black">{liveVitals.triage}</p>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center border border-dashed border-white/10 rounded-2xl">
                <p className="text-xs text-gray-500 italic">Waiting for biometric bridge connection...</p>
              </div>
            )}
          </section>

          {/* SENSORY INPUTS */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Visual Assessment */}
            <section className="glass-panel rounded-[2rem] p-6 border-white/5 space-y-4">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-amber-400">📷 Visual Data</h3>

              {!cameraOn ? (
                <button
                  onClick={startCamera}
                  className="w-full aspect-video glass-card rounded-2xl flex flex-col items-center justify-center gap-3 border-white/5 hover:border-amber-500/30 transition-all group"
                >
                  <span className="text-4xl group-hover:scale-110 transition-transform">📸</span>
                  <span className="text-[10px] font-black uppercase tracking-widest text-gray-400">Activate Lens</span>
                </button>
              ) : (
                <div className="space-y-4">
                  <video ref={videoRef} autoPlay playsInline className="w-full rounded-2xl aspect-video object-cover border border-amber-500/30 shadow-[0_0_20px_rgba(245,158,11,0.1)]" />
                  <div className="flex gap-2">
                    <button onClick={captureImage} className="flex-1 py-3 bg-amber-500 text-slate-900 font-black rounded-xl text-xs uppercase tracking-widest">Capture</button>
                    <button onClick={stopCamera} className="px-4 py-3 bg-white/10 text-white font-black rounded-xl text-xs uppercase tracking-widest">Off</button>
                  </div>
                </div>
              )}

              <div className="relative">
                <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="img-upload" />
                <label htmlFor="img-upload" className="flex items-center justify-center gap-2 py-3 w-full glass-card rounded-xl text-[10px] font-black uppercase tracking-widest cursor-pointer hover:bg-white/5">
                  <span>📂</span> {image ? 'Replace Image' : 'Upload File'}
                </label>
              </div>
              {image && (
                <div className="space-y-2 mt-4">
                  <div className="w-full rounded-2xl overflow-hidden border border-emerald-500/30">
                    <img src={URL.createObjectURL(image)} alt="Preview" className="w-full max-h-48 object-contain bg-black/50" />
                  </div>
                  <p className="text-[10px] text-emerald-400 font-bold text-center animate-pulse">✓ IMAGE READY FOR ANALYSIS</p>
                </div>
              )}
              <canvas ref={canvasRef} hidden />
            </section>

            {/* Audio & Text Assessment */}
            <section className="glass-panel rounded-[2rem] p-6 border-white/5 space-y-6">
              <div className="space-y-4">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-cyan-400">🎤 Acoustic Data</h3>
                <div className="relative">
                  <input type="file" accept="audio/*" onChange={handleAudioUpload} className="hidden" id="audio-upload" />
                  <label htmlFor="audio-upload" className="flex flex-col items-center justify-center gap-3 py-6 w-full glass-card rounded-2xl border-white/5 cursor-pointer hover:border-cyan-500/30 transition-all group">
                    <span className="text-3xl group-hover:scale-110 transition-transform">🎙️</span>
                    <span className="text-[10px] font-black uppercase tracking-widest text-gray-400">
                      {audio ? 'Audio Loaded' : 'Upload Field Audio'}
                    </span>
                  </label>
                </div>
                {audio && (
                  <div className="mt-2">
                    <audio controls src={URL.createObjectURL(audio)} className="w-full h-10 outline-none" />
                    <p className="text-[10px] text-cyan-400 font-bold text-center mt-2 animate-pulse">✓ AUDIO READY FOR ANALYSIS</p>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-blue-400">📝 Manual Intel</h3>
                <textarea
                  rows="3"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="w-full glass-card rounded-2xl p-4 text-sm placeholder:text-gray-600 resize-none"
                  placeholder="Enter patient status, trauma details, or observations..."
                />
              </div>
            </section>
          </div>

          {/* MANUAL OVERRIDES */}
          <section className="glass-panel rounded-[2rem] p-6 border-white/5">
            <h3 className="text-xs font-black uppercase tracking-[0.2em] text-red-400 mb-6">⚙️ Manual Vitals Override</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="space-y-1">
                <p className="text-[10px] text-gray-500 font-bold uppercase ml-1">Pulse</p>
                <input type="number" value={pulse} onChange={(e) => { setPulse(e.target.value); setIsManualOverride(true); }} className="w-full glass-card rounded-xl p-3 text-sm" placeholder="BPM" />
              </div>
              <div className="space-y-1">
                <p className="text-[10px] text-gray-500 font-bold uppercase ml-1">SpO2</p>
                <input type="number" value={spo2} onChange={(e) => { setSpo2(e.target.value); setIsManualOverride(true); }} className="w-full glass-card rounded-xl p-3 text-sm" placeholder="%" />
              </div>
              <div className="space-y-1">
                <p className="text-[10px] text-gray-500 font-bold uppercase ml-1">Sys BP</p>
                <input type="number" value={systolicBP} onChange={(e) => { setSystolicBP(e.target.value); setIsManualOverride(true); }} className="w-full glass-card rounded-xl p-3 text-sm" placeholder="mmHg" />
              </div>
            </div>
            <label className="flex items-center gap-3 p-4 glass-card rounded-2xl cursor-pointer hover:bg-white/5 transition-colors">
              <input type="checkbox" checked={unconscious} onChange={(e) => setUnconscious(e.target.checked)} className="w-5 h-5 rounded-lg border-white/10 bg-white/5 text-cyan-500 focus:ring-cyan-500" />
              <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Unconscious State Detected</span>
            </label>
          </section>

          {/* RESULTS SECTION */}
          {result && (
            <section className="glass-panel rounded-[3rem] p-8 border-white/5 animate-fade-in shadow-[0_0_50px_rgba(0,0,0,0.5)]">
              <div className="text-center mb-10">
                <p className="text-[10px] font-black uppercase tracking-[0.4em] text-gray-500 mb-2">Final Classification</p>
                <div className={`inline-block px-12 py-6 rounded-3xl text-4xl font-black font-['Outfit'] shadow-2xl border-2 ${result.triage_level === 'Red' ? 'bg-red-500/20 border-red-500/40 text-red-400 animate-pulse' :
                    result.triage_level === 'Yellow' ? 'bg-amber-500/20 border-amber-500/40 text-amber-400' :
                      result.triage_level === 'Black' ? 'bg-gray-800 border-gray-700 text-gray-400' :
                        'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                  }`}>
                  {result.triage_level.toUpperCase()}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
                <div className="space-y-6">
                  <div className="glass-card p-6 rounded-3xl border-white/5">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-blue-400 mb-4">🧠 AI Rationale</h4>
                    <p className="text-sm leading-relaxed text-gray-300 font-medium italic">"{result.explanation}"</p>
                  </div>

                  <div className="glass-card p-6 rounded-3xl border-white/5">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-emerald-400 mb-4">💡 Protocol Advice</h4>
                    <ul className="space-y-3">
                      {result.recommended_action.map((action, i) => (
                        <li key={i} className="text-xs font-bold text-gray-200 flex items-start gap-3">
                          <span className="text-emerald-500">→</span> {action}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="glass-card p-6 rounded-3xl border-white/5">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-amber-400 mb-4">🚀 Optimized Logistics</h4>
                    <div className="space-y-3">
                      {result.resource_advice.output.map((advice, i) => (
                        <p key={i} className="text-xs font-bold text-amber-100 flex items-start gap-3">
                          <span className="text-amber-500">⚡</span> {advice}
                        </p>
                      ))}
                    </div>
                  </div>

                  <div className="glass-card p-6 rounded-3xl border-white/5">
                    <h4 className="text-[10px] font-black uppercase tracking-widest text-cyan-400 mb-6">📊 Confidence Matrix</h4>
                    <div className="space-y-4">
                      {Object.entries(result.probabilities).map(([label, val]) => (
                        <div key={label} className="space-y-1.5">
                          <div className="flex justify-between text-[10px] font-black uppercase">
                            <span className="text-gray-500">{label}</span>
                            <span className="text-cyan-400">{(val * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div className={`h-full transition-all duration-1000 ${label === 'Red' ? 'bg-red-500' :
                                label === 'Yellow' ? 'bg-amber-500' :
                                  label === 'Black' ? 'bg-gray-600' : 'bg-emerald-500'
                              }`} style={{ width: `${val * 100}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={downloadPDF}
                className="w-full py-5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-black rounded-2xl text-sm uppercase tracking-[0.2em] shadow-2xl transition-all"
              >
                Download Full Report PDF
              </button>
            </section>
          )}
        </div>
      </div>

      {/* STICKY ANALYZE BUTTON (Mobile Friendly) */}
      {!result && (
        <div className="fixed bottom-0 left-0 right-0 p-6 z-50 pointer-events-none">
          <div className="max-w-4xl mx-auto pointer-events-auto">
            <button
              onClick={sendData}
              disabled={loading}
              className="w-full py-5 glass-panel border-cyan-500/30 text-white font-black text-sm uppercase tracking-[0.3em] rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-3 glow-on-hover bg-gradient-to-r from-blue-600/20 to-cyan-600/20"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>Analyze Case</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Chat Panel */}
      <ChatPanel
        recipientId={doctorId}
        recipientName={doctorName}
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
      />
    </div>
  );
}

export default TriageApp;
