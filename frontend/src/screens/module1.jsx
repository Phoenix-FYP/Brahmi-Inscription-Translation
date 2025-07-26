// screens/module1.jsx
import React, { useState } from "react";
import { FaUpload } from "react-icons/fa";
import "../css/module1.css";

const Module1 = () => {
  const [selectedImages, setSelectedImages] = useState([]);
  const [threshold, setThreshold] = useState("");

  const denoisedImages = ["/sample1.jpg", "/sample2.jpg"];
  const segmentedImages = ["/char1.jpg", "/char2.jpg", "/char3.jpg"];

  const handleSelectImage = (img) => {
    setSelectedImages((prev) =>
      prev.includes(img)
        ? prev.filter((i) => i !== img)
        : [...prev, img]
    );
  };

  return (
    <div className="module1-container">
      <h2 className="module1-title">Image Preprocessing Component</h2>

      <div className="module1-box">
        <p className="module1-subtitle">Please upload your Estampage Image:</p>
        <label className="module1-upload">
          <FaUpload className="upload-icon" />
          <input type="file" className="hidden" />
        </label>

        <div className="threshold-section">
          <label htmlFor="threshold">Enter the Threshold Value:</label>
          <input
            type="number"
            id="threshold"
            className="threshold-input"
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
          />
        </div>
        <p className="threshold-note">
          Threshold value should be adjusted accordingly.
        </p>
      </div>

      <div className="module1-box">
        <h3 className="module1-box-title">Denoised Process Images</h3>
        <div className="image-preview-section">
          {denoisedImages.map((src, idx) => (
            <img key={idx} src={src} alt="denoised" className="preview-img" />
          ))}
        </div>
      </div>

      <div className="module1-box">
        <h3 className="module1-box-title">Segmented Characters</h3>
        <p className="module1-subtitle">
          You can select the image and discard if it's a noise.
        </p>
        <div className="image-select-section">
          {segmentedImages.map((src, idx) => (
            <div
              key={idx}
              className={`selectable-img-box ${
                selectedImages.includes(src) ? "selected" : ""
              }`}
              onClick={() => handleSelectImage(src)}
            >
              <img src={src} alt={`char-${idx}`} className="preview-img" />
            </div>
          ))}
        </div>
      </div>

      <button className="proceed-button">Proceed to Next Stage</button>
    </div>
  );
};

export default Module1;
