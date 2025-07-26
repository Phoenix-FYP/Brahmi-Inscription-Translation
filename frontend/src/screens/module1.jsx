// screens/module1.jsx
import React, { useState } from "react";
import { FaUpload } from "react-icons/fa";

const Module1 = () => {
  const [selectedImages, setSelectedImages] = useState([]);
  const [threshold, setThreshold] = useState("");

  const denoisedImages = ["/sample1.jpg", "/sample2.jpg"];
  const segmentedImages = ["/char1.jpg", "/char2.jpg", "/char3.jpg"];

  const handleSelectImage = (img) => {
    setSelectedImages((prev) =>
      prev.includes(img) ? prev.filter((i) => i !== img) : [...prev, img]
    );
  };

  return (
    <div className="m-8">
      <h2 className="text-3xl font-bold text-center mb-8">
        Image Preprocessing Component
      </h2>

      {/* Upload Section */}
      <div className="bg-gray-100 rounded-xl p-6 mb-8">
        <p className="font-semibold mb-4">Please upload your Estampage Image:</p>

        <label className="w-full max-w-md flex flex-col items-center justify-center border-2 border-dashed border-gray-400 rounded-lg p-6 bg-white cursor-pointer">
          <FaUpload className="text-2xl text-gray-600 mb-2" />
          <span className="text-sm text-gray-600">Click to upload</span>
          <input type="file" className="hidden" />
        </label>

        <div className="flex items-center gap-4 mt-6">
          <label htmlFor="threshold" className="font-medium">
            Enter the Threshold Value:
          </label>
          <input
            type="number"
            id="threshold"
            className="px-3 py-1 border border-gray-300 rounded-md w-32"
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
          />
        </div>
        <p className="text-red-500 mt-2 text-sm">
          Threshold value should be adjusted accordingly.
        </p>
      </div>

      {/* Denoised Images */}
      <div className="bg-gray-100 rounded-xl p-6 mb-8">
        <h3 className="text-xl font-semibold mb-4">Denoised Process Images</h3>
        <div className="flex gap-4 flex-wrap">
          {denoisedImages.map((src, idx) => (
            <img
              key={idx}
              src={src}
              alt="denoised"
              className="w-24 h-24 rounded-md object-cover border border-gray-300"
            />
          ))}
        </div>
      </div>

      {/* Segmented Characters */}
      <div className="bg-gray-100 rounded-xl p-6 mb-8">
        <h3 className="text-xl font-semibold mb-2">Segmented Characters</h3>
        <p className="text-sm text-gray-600 mb-4">
          You can select the image and discard if it's a noise.
        </p>
        <div className="flex gap-4 flex-wrap">
          {segmentedImages.map((src, idx) => (
            <div
              key={idx}
              onClick={() => handleSelectImage(src)}
              className={`p-1 rounded-md border-2 cursor-pointer ${
                selectedImages.includes(src)
                  ? "border-blue-500"
                  : "border-transparent"
              }`}
            >
              <img
                src={src}
                alt={`char-${idx}`}
                className="w-24 h-24 rounded-md object-cover border border-gray-300"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Proceed Button */}
      <div className="text-center">
        <button className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700">
          Proceed to Next Stage
        </button>
      </div>
    </div>
  );
};

export default Module1;
