// screens/Home.jsx
import React from "react";
import Module1 from "./module1";
import Module2 from "./module2";
import Module3 from "./module3";
import Module4 from "./module4";

const Home = () => {
  return (
    <div>
      <section id="module1">
        <Module1 />
      </section>
      <section id="module2">
        <Module2 />
      </section>
      <section id="module3">
        <Module3 />
      </section>
      <section id="module4">
        <Module4 />
      </section>
    </div>
  );
};

export default Home;
