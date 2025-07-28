import React, { useState, useEffect } from "react";
import axios from "axios";
import CharacterSection from "../components/CharacterSection";

const Module2 = ({ setCorpusOutput, image_no, segmentedImages }) => {
  const [module2Data, setModule2Data] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Fetch Module 2 results when component mounts
  useEffect(() => {
    const fetchModule2Results = async () => {
      if (!image_no || !segmentedImages || segmentedImages.length === 0) {
        setError("No segmented images or image number provided.");
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response = await axios.post(
          "http://localhost:8000/api/run-module2/",
          {
            image_no,
            segmented_images: segmentedImages,
          },
          {
            headers: { "Content-Type": "application/json" },
          }
        );

        // Transform backend response to match CharacterSection format
        const predictions = response.data.predictions || {};
        const formattedData = [
          {
            title: "Character Recognition Results",
            characterImages: segmentedImages,
            mappedCorpus: predictions.final_sequence || "No predictions available",
          },
        ];
        setModule2Data(formattedData);
      } catch (err) {
        setError("Error fetching Module 2 results. Please try again.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchModule2Results();
  }, [image_no, segmentedImages]);

  // Combine all mappedCorpus strings into one
  const combinedCorpus = module2Data.map((s) => s.mappedCorpus).join(" ");

const handleProceed = async () => {
    if (segmentedImages.length === 0) {
      setError("No segmented images to process.");
      return;
    }

    try {
      setLoading(true);
      // Navigate to Module2 with image_no and segmentedImages
      navigate("/module2", {
        state: { image_no, segmentedImages },
      });
    } catch (err) {
      setError("Error navigating to Module 2. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 bg-slate-50 min-h-screen m-auto max-w-4xl">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Module 2</h2>

      {loading && (
        <p className="text-center text-gray-600">Processing Module 2...</p>
      )}

      {error && (
        <p className="text-center text-red-500 mb-4">{error}</p>
      )}

      {!loading && module2Data.length === 0 && !error && (
        <p className="text-center text-gray-600">
          No results available. Please process an image in Module 1 first.
        </p>
      )}

      {module2Data.map((section, index) => (
        <CharacterSection
          key={index}
          title={section.title}
          characterImages={section.characterImages}
          mappedCorpus={section.mappedCorpus}
        />
      ))}

      {module2Data.length > 0 && (
        <div className="mt-6 text-center">
          <button
            onClick={handleProceed}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-xl shadow hover:bg-blue-700 transition disabled:bg-gray-400"
          >
            Proceed to Word Segmentation
          </button>
        </div>
      )}
    </div>
  );
};

export default Module2;