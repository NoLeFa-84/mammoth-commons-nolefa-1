import markdown2
from mammoth_commons.datasets import Text
from mammoth_commons.integration import metric
from mammoth_commons.models import LLM
from mammoth_commons.exports import HTML


@metric(
    namespace="mammotheu",
    version="v053",
    python="3.13",
    packages=(),
)
def llm_audit(
    dataset: Text, model: LLM, sensitive: list[str], chain_of_votes: int = 10
) -> HTML:
    """
    This assessment methodology sets an LLM at the role of fairness auditory and asks it to provide
    a sequence of votes, obtaining an assessment of whether given text is biased or neutral.
    Then, it follows a chain-of-thought approach for summarizing the reasoning associated with all
    valid votes (some votes may be invalid due to erroneous formatting) and eventually identifying
    actionable insights or explanations.

    Args:
        chain_of_votes: How many votes should be casted.
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
    faq_html = f"""
    <style>
    .faq-container {{
      max-width: 600px;
      margin: 20px auto;
      font-family: Arial, sans-serif;
    }}
    .faq-box {{
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
      background: #fff;
    }}
    .faq-box h3 {{
      margin-top: 0;
      font-size: 1.2em;
      color: #333;
    }}
    .faq-box p {{
      margin: 0;
      color: #555;
    }}
    </style>

    <div class="faq-container">
        <div class="faq-box">
            <h3>❓ What is this?</h3>
            <p>This module uses a large language model (LLM) as a fairness auditor of a free text snippet. 
            The model is prompted to cast multiple votes on whether the input text is biased or neutral. 
            Each vote includes reasoning, and valid votes are aggregated to capture a common perspective.</p>
            <br/>
            <p>Through this chain-of-votes methodology, the LLM provides both a verdict and a narrative explanation. 
            This helps uncover hidden patterns of bias in text and yields actionable insights for improvement.</p>
        </div>

        <div class="faq-box">
            <h3>❗ Summary</h3>
            
            <p><b>The input has been classified as {title}</b> after being subjected to {chain_of_votes} 
            independent LLM assessments. The reasoning highlights which aspects of the text contribute to this 
            judgement and  offers {'mitigation steps to address biases' if title.startswith('Biased') else 'an explanation of why the text is considered neutral'}.</p>
            <br/>
            <p>There are reasoning outputs and action points to improve fairness.
            However, these should be used as guidance rather than definitive answers or course of action. 
            Remember that LLM auditors may reflect the biases of their training or finetuning corpora. 
            Manual inspection is recommended to validate findings.</p>
        </div>
    </div>
    """
    html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Text analysis</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <div class="container py-5">
                <h1 class="mb-4">{title}</h1>
                <hr/>
                {faq_html}
                <hr/>

                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">Original text</div>
                    <div class="card-body">
                        <p>{dataset.text}</p>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header bg-warning text-dark">Verdict</div>
                    <div class="card-body">
                        <p class="mb-0"><strong>{title}</strong></p>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header bg-info text-white">Reasoning</div>
                    <div class="card-body">
                        {markdown2.markdown(commentary)}
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header bg-success text-white">{
    'Action points' if title.startswith('Biased') else 'Explanation'
    }</div>
                    <div class="card-body">
                        {markdown2.markdown(result)}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    return HTML(html)
