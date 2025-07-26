import React, { useState } from "react";
import CharacterSection from "../components/CharacterSection";


const module2Data = [
  {
    title: "Random Forest",
    characterImages: [
      "/images/char1.png",
      "/images/char2.png",
      "/images/char3.png",
    ],
    mappedCorpus: "අභිවන්දනා",
  },
  {
    title: "Support Vector Machine",
    characterImages: [
      "/images/char4.png",
      "/images/char5.png",
      "/images/char6.png",
    ],
    mappedCorpus: "සංඝානුභාවනා",
  },
];

const Module2 = ({ setCorpusOutput }) => {
  // Combine all mappedCorpus strings into one
  const combinedCorpus = module2Data.map((s) => s.mappedCorpus).join(" ");

  const handleProceed = () => {
    // Send final output string to parent (Home.jsx)
    setCorpusOutput(combinedCorpus);
    // Optional: Scroll to Module 3
    document.getElementById("module3")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="p-8 bg-slate-50 min-h-screen m-auto max-w-4xl">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Module 2</h2>

      {module2Data.map((section, index) => (
        <CharacterSection
          key={index}
          title={section.title}
          characterImages={section.characterImages}
          mappedCorpus={section.mappedCorpus}
        />
      ))}

      <div className="mt-6 text-center">
        <button
          onClick={handleProceed}
          className="px-6 py-2 bg-blue-600 text-white rounded-xl shadow hover:bg-blue-700 transition"
        >
          Proceed to Word Segmentation
        </button>
      </div>
    </div>
  );
};

export default Module2;