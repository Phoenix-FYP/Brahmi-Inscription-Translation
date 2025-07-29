import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React, { useState } from "react";
import Home from "./screens/home";
import Module2 from "./screens/module2";

const App = () => {
  const [corpusOutput, setCorpusOutput] = useState("");

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home setCorpusOutput={setCorpusOutput} />} />
        <Route
          path="/module2"
          element={<Module2 setCorpusOutput={setCorpusOutput} />}
        />
      </Routes>
    </Router>
  );
};

export default

 App;