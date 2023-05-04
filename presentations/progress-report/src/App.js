import React from 'react'
import { useEffect } from 'react'
import Reveal from "reveal.js"
import RevealMarkdown from "reveal.js/plugin/markdown/markdown.esm.js"
import RevealMath from "reveal.js/plugin/math/math.esm.js"
import styles from "./App.module.css"

import ETHLogo from "./svg/ethz.js"
import "reveal.js/dist/reveal.css"
import "reveal.js/dist/theme/white.css"
// import "katex/dist/katex.min.css"

import markdown from "./README.md"
function App() {
  useEffect(()=>{
    
    var deck = new Reveal({
      katex: {
        // local:'node_modules/katex',
        version:"0.16.6",
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true}
       ],
       extensions:['mhchem'],
       ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      },
      plugins: [ 
        RevealMarkdown,
        RevealMath.KaTeX ]
    });
    deck.initialize()

  },[])

  return (
    <div className={styles.container}>
      <div className="reveal">
        <div className="slides">
          <section data-markdown={markdown}
            data-separator="---"
            data-separator-vertical="==="
            >
          
          </section>
        </div>
      </div>
      <div className={styles.logo}>
        <ETHLogo/>
      </div>
      <div className={styles.me}>
        Mingyuan Chi
      </div>
    </div>
  );
}

export default App;
