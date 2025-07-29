// screens/Home.jsx
import React from "react";
import Module1 from "./module1";
import Module2 from "./module2";
import Module3 from "./module3";
import Module4 from "./module4";

const Home = ({ corpusOutput, setCorpusOutput }) => {
  return (
    <div>
      <header className="bg-[#D8AF78] text-white p-4 text-center rounded">
        <h1 className="text-3xl font-bold py-2">BRHAMIYA</h1>
        <p className="text-lg">A tool for translating ancient Brahmi inscriptions</p>
      </header> 
      <section id="module1" class="py-20">
        <Module1 
        corpusOutput={corpusOutput}
        setCorpusOutput={setCorpusOutput} 
        />
      </section>
      <section id="module2" class="py-20">
        <Module2 />
      </section>
      <section id="module3" class=" py-20">
        <Module3 />
      </section>
      <section id="module4" class="py-20 ">
        <Module4 />
      </section>
    </div>
  );
};

export default Home;
