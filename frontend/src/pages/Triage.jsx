import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import jsPDF from "jspdf";

const Triage = () => {
  const webcamRef = useRef(null);

  const [useCamera, setUseCamera] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [text, setText] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Capture image
  const captureImage = async () => {
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return;

    const blob = await fetch(screenshot).then(r => r.blob());
    setImageFile(new File([blob], "capture.jpg", { type: "image/jpeg" }));
    setUseCamera(false);
  };

  // Analyze
  const analyze = async () => {
    setError("");
    setResult(null);

    if (!imageFile && !audioFile && !text.trim()) {
      setError("Provide at least one input.");
      return;
    }

    setLoading(true);
    const form = new FormData();

    if (imageFile) form.append("image", imageFile);
    if (audioFile) form.append("audio", audioFile);
    if (text.trim()) form.append("text", text);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        form,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // PDF
  const downloadPDF = () => {
    if (!result) return;

    const pdf = new jsPDF();
    let y = 15;

    pdf.text("Military Triage Report", 10, y); y += 10;
    pdf.text(`Triage Level: ${result.triage_level}`, 10, y); y += 8;
    pdf.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 10, y); y += 8;

    pdf.text("Modalities Used:", 10, y); y += 8;
    result.modalities_used.forEach(m => {
      pdf.text(`- ${m}`, 12, y);
      y += 6;
    });

    pdf.save("triage_report.pdf");
  };

  return (
    <div style={{ padding: 20, maxWidth: 900 }}>
      <h2>🪖 Military Triage System</h2>

      {/* IMAGE */}
      <h4>📷 Image</h4>
      {!useCamera && (
        <>
          <button onClick={() => setUseCamera(true)}>Use Camera</button><br /><br />
          <input type="file" accept="image/*" onChange={e => setImageFile(e.target.files[0])} />
        </>
      )}

      {useCamera && (
        <>
          <Webcam ref={webcamRef} screenshotFormat="image/jpeg" width={320} />
          <br />
          <button onClick={captureImage}>Capture</button>
          <button onClick={() => setUseCamera(false)}>Cancel</button>
        </>
      )}

      {imageFile && <p>✔ Image selected</p>}

      <hr />

      {/* AUDIO */}
      <h4>🎧 Audio</h4>
      <input type="file" accept="audio/*" onChange={e => setAudioFile(e.target.files[0])} />
      {audioFile && <p>✔ Audio selected</p>}

      <hr />

      {/* TEXT */}
      <h4>✍️ Description</h4>
      <textarea rows={3} style={{ width: "100%" }} value={text} onChange={e => setText(e.target.value)} />

      <br /><br />
      <button onClick={analyze} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
  <div style={{ marginTop: 20, border: "1px solid #ccc", padding: 16 }}>
    <h3>🩺 Analysis Result</h3>

    <p><b>Triage Level:</b> {result.triage_level}</p>
    <p><b>Overall Confidence:</b> {(result.confidence * 100).toFixed(1)}%</p>

    {/* RECOMMENDED ACTION */}
    {result.advice && (
      <>
        <h4>💡 Recommended Action</h4>
        <ul>
          {result.advice.map((a, i) => (
            <li key={i}>{a}</li>
          ))}
        </ul>
      </>
    )}

    {/* AUDIO */}
    {result.audio_raw && (
      <>
        <h4>🎧 Audio Analysis</h4>
        <p>
          {result.audio_raw.label} (
          {(result.audio_raw.confidence * 100).toFixed(1)}%)
        </p>
      </>
    )}

    {/* IMAGE */}
    {result.visual_raw && (
      <>
        <h4>🖼 Image Analysis</h4>
        <p>
          {result.visual_raw.label} (
          {(result.visual_raw.confidence * 100).toFixed(1)}%)
        </p>
      </>
    )}

    {/* TEXT */}
    {result.text_raw && (
      <>
        <h4>✍️ Text Analysis</h4>
        <p>
          {result.text_raw.label} (
          {(result.text_raw.confidence * 100).toFixed(1)}%)
        </p>
      </>
    )}

    {/* PROBABILITIES */}
    {result.probabilities && (
      <>
        <h4>📊 Triage Probabilities</h4>
        {Object.entries(result.probabilities).map(([k, v]) => (
          <div key={k}>
            <b>{k}</b>: {(v * 100).toFixed(1)}%
            <div
              style={{
                height: 8,
                background: "#eee",
                marginBottom: 8
              }}
            >
              <div
                style={{
                  width: `${v * 100}%`,
                  height: "100%",
                  background: "#4caf50"
                }}
              />
            </div>
          </div>
        ))}
      </>
    )}<button onClick={downloadPDF} style={{ marginTop: 10 }}>
  Download PDF
</button>

  </div>
  
)}

    </div>
  );
};

export default Triage;
