import { useState } from "react";
import API from "../services/api";

function Triage() {
  const [image, setImage] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const submit = async () => {
    const formData = new FormData();

    if (image) formData.append("image", image);
    if (text) formData.append("text", text);

    try {
      const res = await API.post("/predict", formData);

      setResult(res.data);

    } catch {
      alert("Prediction failed");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">

      <h1 className="text-2xl font-bold text-yellow-400 mb-6">
        Patient Triage
      </h1>

      <input
        type="file"
        onChange={(e) => setImage(e.target.files[0])}
        className="mb-4"
      />

      <textarea
        placeholder="Describe symptoms..."
        className="w-full p-2 bg-gray-800 rounded mb-4"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        onClick={submit}
        className="bg-yellow-400 text-black px-4 py-2 rounded"
      >
        Analyze
      </button>

      {result && (
        <div className="mt-6 bg-gray-800 p-4 rounded">

          <h2 className="text-xl font-bold mb-2">
            Result: {result.triage_level}
          </h2>

          <p>Confidence: {Math.round(result.confidence * 100)}%</p>

          <ul className="mt-2 list-disc ml-6">
            {result.advice.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>

        </div>
      )}

    </div>
  );
}

export default Triage;
