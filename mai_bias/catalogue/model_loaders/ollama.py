from mammoth_commons.models import LLM
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v054", python="3.13")
def ollama_model(
    name: str = "llama3.2:latest", url: str = "http://localhost:11434"
) -> LLM:
    """
        <img src="https://ollama.com/public/ollama.png" alt="logo" style="float: left; margin-right: 15px; height: 36px;"/>
        <h3>interacts with an ollama LLM</h3>
        Allows interaction with a locally hosted <a href="https://ollama.com" target="_blank">ollama</a> large
        language model (LLM). The interaction can either aim to assess biases of that model, or to use it as an
        aid in discovering qualitative biases in text.
        <details><summary><i>A simple guide to get ollama running.</i></summary>
        Set up this up in the machine where MAI-BIAS runs. Here, information required to prompt that model is provided.
        <div class="wrap">
          <b>1) Install Ollama</b>
          <div class="grid">
            <div class="card">
              <i>macOS (Homebrew)</i>
              <pre><code>brew install ollama
    ollama --version</code></pre>
            </div>
            <div class="card">
              <i>Linux (install script)</i>
              <pre><code>curl -fsSL https://ollama.com/install.sh | sh
    ollama --version</code></pre>
            </div>
            <div class="card">
              <i>Windows (winget)</i>
              <pre><code>winget install Ollama.Ollama</code></pre>
            </div>
          </div>
          <b>2) Start the service</b>
          <div class="card">
            <p>Start the local service and keep it running while you work.</p>
            <pre><code>ollama serve</code></pre>
          </div>
          <b>3) Pull a model</b>
          <div class="card">
            <p>You only need to pull a model once; updates reuse most weights via deltas.</p>
            <pre><code>ollama pull llama3</code></pre>
          </div>
        </div>
        </details>

        Args:
            name: The model name, as pulled in the ollama endpoint.
            url: The url of the ollama endpoint. Default is <code>http://localhost:11434</code>/
    """
    return LLM(name, url)
