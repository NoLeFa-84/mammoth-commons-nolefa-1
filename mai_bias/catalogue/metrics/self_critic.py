import markdown2
from mammoth_commons.datasets import Text
from mammoth_commons.integration import metric
from mammoth_commons.models import LLM
from mammoth_commons.exports import HTML
from mammoth_commons.exports.HTML import simplified_formatter_style
from mammoth_commons.reminders import logo_mai_bias


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=(),
)
def llm_audit(
    dataset: Text,
    model: LLM,
    sensitive: list[str],
    share_text: bool = False,
    chain_of_votes: int = 10,
) -> HTML:
    """<img src="https:///icons/ai.png" alt="ai" style="float: left; margin-right: 15px; height: 36px;"/>
    <h3>use an LLM as text auditor</h3>

    This assessment methodology sets an LLM at the role of fairness auditor and asks it to provide
    a sequence of votes, obtaining an assessment of whether given text is biased or neutral.
    Then, it follows a chain-of-thought approach for summarizing the reasoning associated with all
    valid votes (some votes may be invalid due to erroneous formatting) and eventually identifying
    actionable insights or explanations.

    Args:
        share_text: If the original text comes from a web page, uncheck this option to embed it within the report (this is not the default, because it may violate ownership).
        chain_of_votes: How many votes should be cast.
    """
    from mammoth_commons.externals import notify_progress, notify_end

    if isinstance(sensitive, str):
        sensitive = sensitive.split(",")

    commentaries = list()
    sentiment = 0
    chain_of_votes = int(chain_of_votes)
    assert chain_of_votes >= 1, "At least one chained vote is required"

    for i in range(chain_of_votes):
        notify_progress(
            i / (chain_of_votes + 2.0),
            f"Retrieving bias vote and its reasoning: {i+1}/{chain_of_votes}",
        )
        commentary = model.prompt(
            context=f'You are a fairness auditor that aims to uncover biases in the next user statement. Do you consider the user statement biased, and if so why? Start your reply with "Yes." or "No."L. Consider at least the following attributes as sensitive, and examine those separately: {sensitive}',
            prompt=dataset.text,
        )
        if commentary.startswith("Yes."):
            sentiment += 1
            commentary = commentary[4:]
            commentaries.append(commentary)
        elif commentary.startswith("No."):
            commentary = commentary[3:]
            commentaries.append(commentary)
    title = "Neutral text" if sentiment < chain_of_votes * 0.5 else "Biased text"
    notify_progress(
        (chain_of_votes) / (chain_of_votes + 2.0),
        "Aggregating opinions",
    )
    commentary = model.prompt(
        context=f"You are a fairness auditor that needs to summarize in 250 words the following commentaries to declare a text as {title}. Do not acknowledge the existence of intermediate commentaries and do not make any bullet points.",
        prompt=str(commentaries),
    )
    notify_progress(
        (chain_of_votes + 1) / (chain_of_votes + 2.0),
        "Suggesting insights",
    )
    result = model.prompt(
        context=f"You are a fairness auditor that consider the following user input as {title}. The reasoning is provided by the user. Please provide one list of bullet points for {'addressing' if title.startswith('Biased') else 'explaining'} the reasoning as a markdown list. Consider at least the following attributes as sensitive: {sensitive}",
        prompt="Input:" + dataset.text + "\n" + str(commentary),
    )
    notify_end()
    technology = """<img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/ai.png?raw=true" alt="ai" style="float: left; margin-right: 15px; height: 36px;"/>used an LLM to audit text biases"""
    subtitle = logo_mai_bias + "MAI-BIAS analysis of: " + dataset.source
    html = simplified_formatter_style + f"""
    <div class="banner-container">
        <h1 class="banner {'biased' if title.startswith('Biased') else 'fair'}">{title}</h1>
        <div class="subtitle">{subtitle}</div>
        <div class="technology">{technology}</div>
    </div>
    <div class="pill-buttons">
        <div class="pill-btn" data-target="whatis">What is this?
        <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/question.png?raw=true" height="128px"/>
        </div>
        <div class="pill-btn" data-target="method">Analysis methodology
        <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/methodology.png?raw=true" height="128px"/>
        </div>
        <div class="pill-btn" data-target="pipeline">Data pipeline
        <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/data.png?raw=true" height="128px"/>
        </div>
        <div class="pill-btn" data-target="reasoning">Reasoning
        <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/ai.png?raw=true" height="128px"/>
        </div>
        <div class="pill-btn" data-target="actions">{'Action points' if title.startswith('Biased') else 'Explanation'}
        <br><img src="https://github.com/mammoth-eu/mammoth-commons/blob/dev/docs/icons/checklist.png?raw=true" height="128px"/>
        </div>
    </div>
    <div id="whatis" class="section-panel">
        <p>This module uses a large language model (LLM) as a fairness auditor of provided text.
        A voting technique is used to make its verdict and explanations more robust.</p>
    </div>
    <div id="method" class="section-panel">
        <p>The input was subjected to <b>{chain_of_votes}</b>
        independent LLM assessments. 
        {'It focused on '+','.join(sensitive)+' as sensitive attributes.' if sensitive else 'There was no particular focus on a potentially sensitive attributes.'} 
        These cast votes on whether the text is biased or not.
        The reasoning highlights which aspects of the text contribute to this
        judgement and offers {'mitigation steps to address biases' if title.startswith('Biased') else 'an explanation of why the text is considered neutral'}.</p>
        There are reasoning outputs and action points to improve fairness.
        However, these should be used as guidance rather than definitive answers.
        Manual inspection is recommended.</p>
        
        <details><summary><i>Full text</i></summary>
        <small>{dataset.text if share_text else dataset.source}</small>
        </details>
    </div>
    <div id="pipeline" class="section-panel">{dataset.to_description()}<br><br>{model.to_description()}</div>
    <div id="reasoning" class="section-panel">{markdown2.markdown(commentary)}</div>
    <div id="actions" class="section-panel">{markdown2.markdown(result)}</div>
    
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        const buttons = document.querySelectorAll(".pill-btn");
        const sections = document.querySelectorAll(".section-panel");

        buttons.forEach(btn => {{
            btn.addEventListener("click", () => {{
                let target = btn.getAttribute("data-target");

                buttons.forEach(b => b.classList.remove("active"));
                sections.forEach(s => s.classList.remove("active"));

                btn.classList.add("active");
                document.getElementById(target).classList.add("active");
            }});
        }});

        // Activate first section
        document.querySelector(".pill-btn").classList.add("active");
        document.querySelector(".section-panel").classList.add("active");
    }});
    </script>
    """
    return HTML(html)
