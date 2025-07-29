import React from "react";
import SectionBox from "../components/SectionBox";

const sections = [
  {
    title: "Input",
    content: "අභිවන්දනාසංඝානුභාවනා",
  },
  {
    title: "Bert Based Model Segementation",
    content: "Reverence... Blessings of the Sangha",
  },
  {
    title: "POS Tagging",
    content: "අභිවන්දනා: Noun | සංඝානුභාවනා: Noun",
  },
  {
    title: "Greedy Based Segmentation",
    content: "Subject: අභිවන්දනා | Object: සංඝානුභාවනා",
  },
  {
    title: "Calculations",
    content: "Reverence to the Sangha's blessings.",
  },
  {
    title: "byT5 Mopdel Prediction",
    content: "අභිවන්දනා → අභිවන්දනාව\nසංඝානුභාවනා → සංඝානුභාවනය",
  },
  {
    title: "Segmented Word",
    content: "අභිවන්දනා → අභිවන්දනාව\nසංඝානුභාවනා → සංඝානුභාවනය",
  },
];

const Module3= () => {
  const handleGoToStart = () => {
    document.getElementById("module1")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="bg-[#e3d6bf] mb-8 justify-center flex flex-col items-center p-15 max-w-3xl mx-auto rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Module 3</h2>

      {sections.map((section, idx) => (
        <SectionBox key={idx} title={section.title} content={section.content} />
      ))}

      <div className="mt-8 text-center">
        <button
          onClick={handleGoToStart}
          className="px-6 py-2 bg-green-600 text-white rounded-xl shadow hover:bg-green-700 transition"
        >
          Go to Start
        </button>
      </div>
    </div>
  );
};

export default Module3
