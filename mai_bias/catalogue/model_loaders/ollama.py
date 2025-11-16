from mammoth_commons.models import LLM
from mammoth_commons.integration import loader


@loader(namespace="mammotheu", version="v053", python="3.13")
def ollama_model(
    name: str = "llama3.2:latest", url: str = "http://localhost:11434"
) -> LLM:
    """References a locally provided <a href="https://ollama.com" target="_blank">ollama</a> model.
        Set up this up in the machine where MAI-BIAS runs. Here, information required to prompt that model is provided.
        <div class="wrap">
          <h2>1) Install Ollama</h2>
          <div class="grid">
            <div class="card">
              <h3>macOS (Homebrew)</h3>
              <pre><code>brew install ollama
    ollama --version</code></pre>
            </div>
            <div class="card">
              <h3>Linux (install script)</h3>
              <pre><code>curl -fsSL https://ollama.com/install.sh | sh
    ollama --version</code></pre>
            </div>
            <div class="card">
              <h3>Windows (winget)</h3>
              <pre><code>winget install Ollama.Ollama</code></pre>
            </div>
          </div>
          <h2>2) Start the service</h2>
          <div class="card">
            <p>Start the local service and keep it running while you work.</p>
            <pre><code>ollama serve</code></pre>
          </div>
          <h2>3) Pull a model</h2>
          <div class="card">
            <p>You only need to pull a model once; updates reuse most weights via deltas.</p>
            <pre><code>ollama pull llama3</code></pre>
          </div>
        </div>

        Args:
            name: The model name, as pulled in the ollama endpoint.
            url: The url of the ollama endpoint. Default is <code>http://localhost:11434</code>/
    """
    return LLM(name, url)
