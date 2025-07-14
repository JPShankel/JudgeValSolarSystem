# An evaluation of ChatGPT's knowledge of the solar system using Judgement Labs evaluators
# Author: Jason Shankel

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerCorrectnessScorer
from judgeval.scorers import FaithfulnessScorer
from judgeval.tracer import Tracer, wrap
from judgeval.data.datasets import EvalDataset

from openai import OpenAI
gpt_client = wrap(OpenAI())  # tracks all LLM calls

judgment = Tracer(project_name="JasonTest")

correctness_scorer = AnswerCorrectnessScorer(threshold=0.5)
faithfulness_scorer = FaithfulnessScorer(threshold = 0.5)

client = JudgmentClient()

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    return f"Question : {question}"

@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = gpt_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content

moon_examples = [Example(
    input="How many moons does Mercury have?",
    actual_output=run_agent("How many moons does Mercury have?"),
    expected_output="Mercury has no moons."
),
    Example(
        input="How many moons does Venus have?",
        actual_output=run_agent("How many moons does Venus have?"),
        expected_output="Venus has no moons."
),
    Example(
        input="How many moons does Earth have?",
        actual_output=run_agent("How many moons does Earth have?"),
        expected_output="Earth has one moon."
),
    Example(
        input="How many moons does Mars have?",
        actual_output=run_agent("How many moons does Mars have?"),
        expected_output="Mars has two moons."
),
    Example(
        input="How many moons does Jupiter have?",
        actual_output=run_agent("How many moons does Jupiter have?"),
        expected_output="Jupiter has 95 moons."
)
]

moons_dataset = EvalDataset(examples=moon_examples)
# client.push_dataset(alias="moons_dataset",dataset=moons_dataset,project_name="JasonTest",overwrite=True)

moon_results = client.run_evaluation(
    examples=moons_dataset.examples,
    eval_run_name="How Many Moons",
    scorers=[correctness_scorer],
    model="gpt-4.1",
    project_name="JasonTest",
    override=True
)
print(moon_results)

venus_examples = [Example(
    input="Why can't we see Venus in the middle of the night?",
    actual_output=run_agent("Why can't we see Venus in the middle of the night?"),
    retrieval_context=["Venus is a planet in the inner solar system that is closer to the sun than the Earth, therefore the night side of Earth always points away from Venus."]),
    
    Example(
        input="Why are there no rovers on Venus?",
        actual_output=run_agent("Why are there no rovers on Venus?"),
        retrieval_context=["The temperature and pressure on Venus are too high for probes to survive for more than a few hours. The atmosphere is also highly corrosive."]
    )]

venus_dataset = EvalDataset(examples=venus_examples)
# client.push_dataset(alias="venus_dataset",dataset=venus_dataset,project_name="JasonTest",overwrite=True)


venus_results = client.run_evaluation(
    examples=venus_dataset.examples,
    eval_run_name="Venus Questions",
    scorers=[faithfulness_scorer],
    model="gpt-4.1",
    project_name="JasonTest",
    override=True
)

print(venus_results)

mars_examples = [Example(
    input="Where is the largest mountain in the solar system?",
    actual_output=run_agent("Where is the largest mountain in the solar system?"),
    retrieval_context=["The solar system's largest mountain is Olympus Mons, located on the fourth planet, Mars"],
    expected_output="The largest planet in the solar system is Olympus Mons on Mars."
),
    Example(
        input="Does Mars have an atmosphere?",
        actual_output=run_agent("Does Mars have an atmosphere?"),
        retrieval_context=["Mars is the fourth planet in the solar system and has a very thin atmosphere compared to Earth and Venus"],
        expected_output="Mars has a thin atmosphere about one percent as thick as Earth's comprised mostly of carbon dioxide"
    ),
    Example(
        input="Is there life on Mars?",
        actual_output=run_agent("Is there life on Mars?"),
        retrieval_context=["There is no definitive proof that life ever existed on the fourth planet of the solar system, Mars."],
        expected_output=["There is no definitive evidence of life on Mars."]
    )
]

mars_dataset = EvalDataset(examples=mars_examples)
# client.push_dataset(alias="mars_dataset",dataset=mars_dataset,project_name="JasonTest",overwrite=True)


mars_results = client.run_evaluation(
    examples=mars_dataset.examples,
    eval_run_name="Mars Questions",
    scorers=[correctness_scorer,faithfulness_scorer],
    model="gpt-4.1",
    project_name="JasonTest",
    override=True
)

print(mars_results)