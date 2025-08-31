import React, { useState } from "react";
import axios from "axios";
import "./App.css";

// Change this if your API runs elsewhere
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8888";

export default function App() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("overlay"); // "overlay" | "mask"
  const [alpha, setAlpha] = useState(0.5);      // overlay transparency
  const [maskFormat, setMaskFormat] = useState("color"); // "color" | "gray"
  const [threshold, setThreshold] = useState(0.5); // for binary models
  const [resultURL, setResultURL] = useState(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const onFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setResultURL(null);
    setMsg("");
  };

  const handleSubmit = async () => {
    if (!file) {
      setMsg("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setMsg("");
    setResultURL(null);

    try {
      let url = "";
      if (mode === "overlay") {
        url = `${API_BASE}/segment?alpha=${alpha}&threshold=${threshold}`;
      } else {
        url = `${API_BASE}/segment/mask?format=${maskFormat}&threshold=${threshold}`;
      }

      const res = await axios.post(url, formData, { responseType: "blob" });
      const blobUrl = URL.createObjectURL(res.data);
      setResultURL(blobUrl);
    } catch (err) {
      console.error(err);
      setMsg(
        err?.response?.data?.detail ||
          "Request failed. Check that the API is running and CORS is enabled."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setResultURL(null);
    setMsg("");
  };

  return (
    <div className="container">
      <header>
        <h1>Cat and Dog Image Segmentation Project</h1>
      </header>

      <section className="card">
        <div className="grid">
          <div className="field">
            <label>Image</label>
            <input type="file" accept="image/*" onChange={onFileChange} />
          </div>

          <div className="field">
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="overlay">Overlay (/segment)</option>
              <option value="mask">Mask Only (/segment/mask)</option>
            </select>
          </div>

          {mode === "overlay" && (
            <div className="field">
              <label>Alpha ({alpha})</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
              />
            </div>
          )}

          {mode === "mask" && (
            <div className="field">
              <label>Mask Format</label>
              <select
                value={maskFormat}
                onChange={(e) => setMaskFormat(e.target.value)}
              >
                <option value="color">color</option>
                <option value="gray">gray</option>
              </select>
            </div>
          )}

          <div className="field">
            <label>Threshold ({threshold})</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              title="Used for binary models only"
            />
          </div>
        </div>

        <div className="actions">
          <button onClick={handleSubmit} disabled={loading}>
            {loading ? "Processing…" : "Run Segmentation"}
          </button>
          <button className="ghost" onClick={resetAll} disabled={loading}>
            Reset
          </button>
        </div>

        {msg && <div className="note error">{msg}</div>}
      </section>

      {resultURL && (
        <section className="card">
          <h3>Result</h3>
          <img className="preview" src={resultURL} alt="Segmentation result" />
          <div className="hint">Right-click → Save image as…</div>
        </section>
      )}

<footer className="credits">
  <h3>👩‍💻👨‍💻 Developed by</h3>
  <p>Nwe Ni Oo Wai · Hein Htet Nyi · Ngwe Sin Linn Latt · Aye Chan Htun Naing</p>

  <h3>🎓 Supervised by</h3>
  <p><b>Ko Thar Htet San</b></p>

  <p className="program">
    Final Project of <b>ML Summer School Program</b> by <b>WAI Myanmar</b>
  </p>
</footer>


    </div>
  );
}
