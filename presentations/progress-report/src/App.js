import React from 'react'
import { useEffect } from 'react'
import Revealjs from "reveal.js"
import RevealMarkdown from "reveal.js/plugin/markdown/markdown.esm.js"
import RevealMath from "reveal.js/plugin/math/math.esm.js"

import "reveal.js/dist/reveal.css"
import "reveal.js/dist/theme/white.css"
import "katex/dist/katex.min.css"

import MD from "./README.md"
function App() {
  useEffect(()=>{
    Reveal.initialize({
      plugins: [ 
        RevealMarkdown,
        RevealMath.KaTeX ]
    });
  })

  return (
    <div className="App">
      <section data-markdown={MD}>

      </section>
    </div>
  );
}

export default App;
