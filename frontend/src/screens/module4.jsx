import React from "react";
import SectionBox from "../components/SectionBox";

const sections = [
  {
    title: "Input",
    content: "අභිවන්දනා සංඝානුභාවනා",
  },
  {
    title: "Direct Translation",
    content: "Reverence... Blessings of the Sangha",
  },
  {
    title: "POS Tagging",
    content: "අභිවන්දනා: Noun | සංඝානුභාවනා: Noun",
  },
  {
    title: "Deep Parsing",
    content: "Subject: අභිවන්දනා | Object: සංඝානුභාවනා",
  },
  {
    title: "Sentence Restructuring",
    content: "Reverence to the Sangha's blessings.",
  },
  {
    title: "Morphological Correction",
    content: "අභිවන්දනා → අභිවන්දනාව\nසංඝානුභාවනා → සංඝානුභාවනය",
  },
];

const Module4 = () => {
  const handleGoToStart = () => {
    document.getElementById("module1")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="p-8 bg-slate-50 min-h-screen m-auto max-w-4xl">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Module 4</h2>

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

export default Module4;
