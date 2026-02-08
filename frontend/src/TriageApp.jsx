import React, { useState, useRef, useEffect } from "react";

import axios from "axios";
import jsPDF from "jspdf";

function TriageApp() {

  // ================= STATES =================

  const [image, setImage] = useState(null);
  const [audio, setAudio] = useState(null);
  const [text, setText] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [cameraOn, setCameraOn] = useState(false);
  

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

// ================= START CAMERA =================

const startCamera = () => {
  setCameraOn(true);
};

  // ================= CAMERA =================

  useEffect(() => {

  if (!cameraOn) return;

  const startStream = async () => {
    try {

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true
      });

      if (videoRef.current) {
  videoRef.current.srcObject = stream;
  videoRef.current.play();
}


    } catch (err) {

      console.error("Camera error:", err);

      if (err.name === "NotAllowedError") {
        alert("Camera permission blocked.");
      } else {
        alert("Camera error: " + err.message);
      }
    }
  };

  startStream();
return () => {
  if (videoRef.current?.srcObject) {
    videoRef.current.srcObject
      .getTracks()
      .forEach(t => t.stop());
  }
};

}, [cameraOn]);



const stopCamera = () => {

  if (videoRef.current?.srcObject) {

    videoRef.current.srcObject
      .getTracks()
      .forEach(track => track.stop());

    videoRef.current.srcObject = null;
  }

  setCameraOn(false);
};

// ================= BLUR DETECTOR =================

const isBlurry = (canvas) => {

  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );

  let gray = [];
  const data = imgData.data;

  // Convert to grayscale
  for (let i = 0; i < data.length; i += 4) {
    const avg =
      (data[i] + data[i + 1] + data[i + 2]) / 3;
    gray.push(avg);
  }

  // Laplacian
  let laplacian = [];
  const w = canvas.width;

  for (let i = w; i < gray.length - w; i++) {
    const val =
      -4 * gray[i] +
      gray[i - 1] +
      gray[i + 1] +
      gray[i - w] +
      gray[i + w];

    laplacian.push(val);
  }

  // Variance
  const mean =
    laplacian.reduce((a, b) => a + b, 0) /
    laplacian.length;

  const variance =
    laplacian.reduce(
      (a, b) => a + (b - mean) ** 2,
      0
    ) / laplacian.length;

  return variance < 120; // threshold
};


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

  ctx.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );


  // ================= BLUR CHECK =================

  const blurry = isBlurry(canvas);

  if (blurry) {

    alert("⚠️ Image is blurry. Please retake.");

    // Do NOT stop camera
    return;
  }


  // ================= SAVE IMAGE =================

  canvas.toBlob(
    (blob) => {

      if (!blob || blob.size === 0) {
        alert("Capture failed. Try again.");
        return;
      }

      const file = new File(
        [blob],
        "capture.jpg",
        { type: "image/jpeg" }
      );

      setImage(file);

      stopCamera();

      alert("✅ Photo Captured Successfully");

    },
    "image/jpeg",
    0.95
  );
};



  // ================= HANDLERS =================

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };


  const handleAudioUpload = (e) => {
    const file = e.target.files[0];
    if (file) setAudio(file);
  };


  // ================= SEND DATA =================

  const sendData = async () => {

    if (!image && !audio && !text) {
      alert("Please upload image, audio, or enter description");
      return;
    }

    const formData = new FormData();

    if (image) formData.append("image", image);
    if (audio) formData.append("audio", audio);
    if (text) formData.append("text", text);

    try {

      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(res.data);

    } catch (err) {

      console.error(err);
      alert("Backend error. Is server running?");

    } finally {

      setLoading(false);
    }
  };


  // ================= HELPERS =================

  const getColor = (level) => {

    if (level === "Black") return "darkred";
    if (level === "Red") return "red";
    if (level === "Yellow") return "orange";
    if (level === "Green") return "green";

    return "black";
  };
// ================= PDF REPORT =================

const downloadPDF = async () => {
  if (!result) return;

  try {
    const res = await axios.post(
      "http://127.0.0.1:8000/download-report",
      result, // 🔥 SEND THE FULL RESULT OBJECT
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
          "Content-Type": "application/json"
        },
        responseType: "blob"
      }
    );

    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "triage_report.pdf");
    document.body.appendChild(link);
    link.click();
    link.remove();
  } catch (err) {
    console.error("PDF download failed", err);
  }
};

  

  // ================= UI =================

  return (

    <div style={{
      maxWidth: 700,
      margin: "auto",
      padding: 25,
      fontFamily: "Arial"
    }}>


      {/* TITLE */}

      <h2 style={{ textAlign: "center" }}>
        🪖 Military Triage System
      </h2>



      {/* ================= IMAGE ================= */}

      <h3>🖼 Injury Image</h3>

      {!cameraOn && (
        <button onClick={startCamera}>
          📷 Open Camera
        </button>
      )}

      <br /><br />

{cameraOn && (
  <>
    <video
      ref={videoRef}
      autoPlay
      playsInline
      style={{
        width: "100%",
        border: "1px solid gray",
        borderRadius: 6
      }}
    />

    <br />

    <button onClick={captureImage}>📸 Capture</button>
    <button onClick={stopCamera}>❌ Stop</button>
  </>
)}


      <br /><br />


      <div>
        📁 Upload From Dataset:
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
        />
      </div>


      {image && (
        <p style={{ color: "green" }}>
          ✔ Image Ready (Preview Disabled)
        </p>
      )}


      <canvas ref={canvasRef} hidden />



      {/* ================= AUDIO ================= */}

      <h3>🎤 Injury Audio</h3>

      <input
        type="file"
        accept="audio/*"
        onChange={handleAudioUpload}
      />


      {audio && (
        <p style={{ color: "green" }}>
          ✔ Audio Uploaded (Playback Disabled)
        </p>
      )}



      {/* ================= TEXT ================= */}

      <h3>📝 Description</h3>

      <textarea
        rows="3"
        style={{ width: "100%" }}
        value={text}
        placeholder="Describe injury..."
        onChange={(e) => setText(e.target.value)}
      />



      {/* ================= BUTTON ================= */}

      <br /><br />

      <button
        onClick={sendData}
        disabled={loading}
        style={{
          width: "100%",
          padding: 12,
          background: "#2b7cff",
          color: "white",
          border: "none",
          borderRadius: 6,
          fontSize: 16,
          cursor: "pointer"
        }}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>



      {/* ================= RESULT ================= */}

      {result && (

        <div style={{
          border: "1px solid #ccc",
          padding: 20,
          borderRadius: 8,
          marginTop: 35,
          background: "#f9f9f9"
        }}>


          <h3>🩺 Analysis Result</h3>


          {/* MAIN */}

          <p>
            <b>Triage Level:</b>{" "}
            <span style={{
              color: getColor(result.triage_level),
              fontWeight: "bold"
            }}>
              {result.triage_level}
            </span>
          </p>


          <p>
            <b>Overall Confidence:</b>{" "}
            {(result.confidence * 100).toFixed(1)}%
          </p>


          <hr />


          {/* ADVICE */}

          {result.advice && (
            <>
              <h4>💡 Recommended Action</h4>

              <ul style={{
                background: "#fff3cd",
                padding: 15,
                borderRadius: 5,
                border: "1px solid #ffeeba"
              }}>
                {result.advice.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>

              <hr />
            </>
          )}



          {/* AUDIO */}

          <h4>🔊 Audio Analysis</h4>

          <p>
            {result.audio_raw.label} (
            {(result.audio_raw.confidence * 100).toFixed(1)}%)
          </p>



          {/* IMAGE */}

          <h4>🖼 Image Analysis</h4>

          <p>
            {result.visual_raw.label} (
            {(result.visual_raw.confidence * 100).toFixed(1)}%)
          </p>



          {/* TEXT */}

          <h4>📝 Text Analysis</h4>

          <p>
            {result.text_raw.label} (
            {(result.text_raw.confidence * 100).toFixed(1)}%)
          </p>



          <hr />
<button
  onClick={downloadPDF}
  style={{
    width: "100%",
    padding: 10,
    background: "#28a745",
    color: "white",
    border: "none",
    borderRadius: 5,
    marginBottom: 15,
    cursor: "pointer"
  }}
>
  📄 Download Report (PDF)
</button>


          {/* PROBABILITY */}

          <h4>📊 Triage Probabilities</h4>

          {Object.entries(result.probabilities).map(([k, v]) => (

            <div key={k} style={{ marginBottom: 12 }}>

              <b>{k}</b>

              <div style={{
                background: "#ddd",
                height: 10,
                width: "100%",
                borderRadius: 5,
                overflow: "hidden"
              }}>

                <div style={{
                  background: "#2b7cff",
                  height: "100%",
                  width: `${v * 100}%`
                }} />

              </div>

              <small>{(v * 100).toFixed(1)}%</small>

            </div>

          ))}

        </div>
      )}

    </div>
  );
}

export default TriageApp;
