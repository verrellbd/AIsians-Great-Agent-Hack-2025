# Project Overview — “Dear Grandma” AIsians

## 1. Introduction
This project presents a structured red‑teaming evaluation designed to measure the robustness of AI agents within the “Dear Grandma” hackathon environment. Our goal was to analyze how different jailbreak techniques affect agent behavior, identify patterns linked to underlying models and frameworks, and highlight vulnerabilities that may inform safer system design.

## 2. Goals
- Evaluate the susceptibility of multiple agent models to jailbreak strategies  
- Identify behavioral and architectural patterns across agents  
- Develop a repeatable evaluation protocol using benign, harmful, and jailbreak tests  
- Summarize vulnerabilities and propose insights for mitigation  

## 3. Methodology

### 3.1 Red-Team Techniques
We applied a range of jailbreak and adversarial prompt strategies, including:

- DAN-style prompts  
- “Please Attack” direct-instruction prompts  
- Many-shot jailbreaks  
- Role-play-based jailbreaks  
- “Back to the Past” temporal attacks  
- Thought experiment / hypothetical scenario attacks  

Each attack type probes different alignment surfaces such as context steering, persona boundaries, and safety-layer bypasses.

### 3.2 Evaluation Protocol
Each agent was tested under three conditions:

1. **Benign Test** — Safe prompts to measure normal behavior  
2. **Harmful Test** — Disallowed content to measure refusal strength  
3. **Jailbreak Test** — Adversarial prompts to measure the Attack Success Rate (ASR)

We measured both quantitative outcomes (ASR) and qualitative patterns (refusal style, consistency, loopholes).

### 3.3 Pattern Identification
During testing, we recorded patterns related to:

- Model architecture characteristics  
- Framework-level safety implementations  
- Context-window vulnerabilities  
- Stability of behavioral rules under long prompts  
- Over-reliance on template refusals  

These patterns helped predict ASR and identify which attack types were most effective for specific model families.

## 4. Results Summary
We compiled the findings into a cross-agent comparison, focusing on:

- Relative success rates of each jailbreak technique  
- Differences between agents under identical test conditions  
- Which architectures showed stronger or weaker guardrails  
- How vulnerabilities generalized across attack types  
- Refusal consistency and brittleness under multi-step prompts  

## 5. Identified Vulnerabilities
Several recurring vulnerabilities emerged:

- Over-dependence on surface-level refusal patterns  
- Weak persona boundaries during role-play prompts  
- Susceptibility to contextual framing and time-shift prompts  
- Safety degradation in long many-shot jailbreaks  
- Inconsistent enforcement of system-level restrictions  

These vulnerabilities suggest that deeper architectural improvements are needed beyond prompt-level defenses.

## 6. Conclusion
Through systematic red-teaming and comparative testing, we uncovered meaningful insights into agent weaknesses and behavioral patterns. These findings can guide the development of more robust safety alignment strategies and contribute to more reliable future AI agent deployments.

# Setup instructions

This section explains how to install all required dependencies and prepare the environment used in the “Dear Grandma” jailbreak evaluation project.

## 1. Prerequisites

Before starting, make sure you have the following installed:

- **Python 3.10+**
- **pip** (Python package manager)
- **git** (optional, for cloning the repo)
- **virtualenv** (recommended but not required)

Check your versions:

```bash
python3 --version
pip --version
```

## 2. Clone this github
```bash
git clone <your-repository-url>
cd <your-project-folder>
```

## 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 4. Environment Variables
Check **.env.example** file and change it to **.env**. Then, change the id key for api that you have

# How to run/test
## 1. Running the Main Attack Script (`attack.py`)

The `attack.py` file contains the command‑line version of the evaluation pipeline.  
It performs **Benign Test**, **Harmful Test**, and **Jailbreaking Test** automatically.

Run it with:

```bash
python attack.py
```

## 2. Running Step‑by‑Step Tests (Interactive Notebook)

For a step‑by‑step explanation and interactive testing, open: 
```bash
attack_description.ipynb
```

## 3. Viewing Finished Results
Each agent has a dedicated results CSV file. You can find them in:
```bash
/results/
```

To compare models against each other, use: **results_comparative.csv**

## 4. Random Tests (Pattern Analysis / Reverse Engineering Inputs)

To analyze hidden behavioral patterns or potential vulnerabilities, use the random test modules:
```bash
/Math Test/
/Random Test/
```

## 7. Recommended Workflow

1. Start with attack_description.ipynb to understand the attack method
2. Run full tests using attack.py
3. Review individual agent CSVs
4. Compare models using comparison.csv
5. Use random and math tests to explore deeper behavioral patterns
6. All collected prompts, responses, and analysis results are stored in the **Results** folder. This folder serves as the primary reference for all attack tests, pattern analyses, and observations used in the project.






