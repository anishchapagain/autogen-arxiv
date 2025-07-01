# AutoGen with Ollama (free LLMs)
This project use multiple AI agents (multi-agent) working collectively to extract data from Arxiz.com and preparing a literature review type of MarkDown document.
Final document is ready to be shared in blog and articles.

1. Researcher Agent
- Collects raw data from arxiv
2. Summarizer Agent
- Summarizes the collected content based on prompt and returns the output.

Output is collected and shared found from ```Console```, ```team.run```, and ```team.run_stream```.

###  AutoGen: A framework from Microsoft for building AI agents and applications
[AutoGen Details](https://microsoft.github.io/autogen/stable/index.html)

[Installation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/installation.html)

### Ollama: Install | Pull & run free LLMs in your machine
[Ollama: llama3.2:latest](https://ollama.com/library/llama3.2:latest)

### Agent Workflow
Prompt
```python
task = "Provide a literature review on the topic related to AI Agent and Machine Learning."
```
System Receiver
```
TextMessage(source='user', models_usage=None, metadata={}, created_at=datetime.datetime...... tzinfo=datetime.timezone.utc), content='Provide a literature review on the topic related to AI Agent and Machine Learning.', type='TextMessage')
```
Tool Request: Arxiv_ResearcherAgent
```
ToolCallRequestEvent(source='Arxiv_ResearcherAgent', models_usage=RequestUsage(prompt_tokens=296, completion_tokens=29), metadata={}, created_at=datetime.datetime.....), content=[FunctionCall(id='0', arguments='{"max_results": "5", "query": "AI Agent and Machine Learning"}', name='arxiv_search')], type='ToolCallRequestEvent'), 

# collects content
content=[FunctionExecutionResult(content='[{\'title\': \'S.......
```
Content Collected -> SummarizerAgent
```
TextMessage(source='SummarizerAgent', models_usage=RequestUsage(prompt_tokens=2695, completion_tokens=1144), metadata={}
```

### Output Files
1. ```team_output.md```: Streaming results
2. ```arxiv_results.json```: Raw data from arxiv
3. ```team_output_final.md```: Desired ouput
4. ```console_output.png```: streaming results from terminal using 'Console'

### Final Output
```
# Literature Review on AI Agent and Machine Learning
=============================================

The field of Artificial Intelligence (AI) has witnessed tremendous growth in recent years, with advancements in machine learning (ML) techniques being a crucial aspect of this progress. This literature review aims to provide an overview of several key papers published in 2025 that have made significant contributions to the field of AI agents and machine learning.

## Paper 1: How to Design and Train Your Implicit Neural Representation for Video Compression
-------------------------------------------

*   **Title:** [How to Design and Train Your Implicit Neural Representation for Video Compression](http://arxiv.org/abs/2506.24127v1)
*   **Authors:** Matthew Gwilliam, Roy Zhang, Namitha Padmanabhan, Hongyang Du, Abhinav Shrivastava
*   **Research Problem(s):** This paper addresses the challenges associated with training implicit neural representation (INR) methods for video compression. The authors aim to improve the efficiency of these methods while maintaining their visual quality and compression ratios.
*   **Key Contributions:** The authors develop a library that allows disentanglement of method components from the NeRV family, reframing their performance in terms of size-quality trade-offs and training time. They also propose a state-of-the-art configuration called Rabbit NeRV (RNeRV) and investigate the viability of hyper-networks to tackle the encoding speed issue.
*   **Keywords:** Implicit Neural Representation, Video Compression, Hyper-Networks

## Paper 2: Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives
-----------------------------------------------------------------------------------------

*   **Title:** [Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives](http://arxiv.org/abs/2506.24124v1)
*   **Authors:** Dong Sixun, Fan Wei, Teresa Wu, Fu Yanjie
*   **Research Problem(s):** This paper tackles the challenge of representing time series data in a more interpretable form using aligned visual and textual perspectives.
*   **Key Contributions:** The authors propose a multimodal contrastive learning framework that transforms raw time series into structured visual and textual representations. They also introduce a variate selection module to identify the most informative variables for multivariate forecasting.
*   **Keywords:** Time Series Forecasting, Multimodal Learning, Contrastive Learning

## Paper 3: Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime
---------------------------------------------------------------------------------------------

*   **Title:** [Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime](http://arxiv.org/abs/2506.24120v1)
*   **Authors:** Yuqing Wang, Shangding Gu
*   **Research Problem(s):** This paper investigates the impact of data uniformity on training efficiency and performance in large language models (LLMs).
*   **Key Contributions:** The authors demonstrate that selecting more uniformly distributed data can improve training efficiency while enhancing performance. They also establish a convergence framework beyond the Neural Tangent Kernel (NTK) regime, applicable to a broad class of architectures.
*   **Keywords:** Data Uniformity, Convergence Framework, NTK Regime

## Paper 4: SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning
----------------------------------------------------------------------------------------------------------------

*   **Title:** [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2506.24119v1)
*   **Authors:** Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques
*   **Research Problem(s):** This paper explores the use of self-play on zero-sum games to develop reasoning capabilities in language models.
*   **Key Contributions:** The authors introduce a self-play framework called SPIRAL, where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves. They also propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training.
*   **Keywords:** Self-Play, Zero-Sum Games, Multi-Agent Reinforcement Learning

## Paper 5: Scaling Human Judgment in Community Notes with LLMs
---------------------------------------------------------

*   **Title:** [Scaling Human Judgment in Community Notes with LLMs](http://arxiv.org/abs/2506.24118v1)
*   **Authors:** Haiwen Li, Soham De, Manon Revel, Andreas Haupt, Brad Miller, Keith Coleman, Jay Baxter, Martin Saveski, Michiel A. Bakker
*   **Research Problem(s):** This paper proposes a new paradigm for Community Notes in the LLM era, where both humans and LLMs can write notes.
*   **Key Contributions:** The authors describe how such a system can work, its benefits, key risks and challenges it introduces, and a research agenda to solve those challenges.
*   **Keywords:** Human Judgment, Community Notes, Large Language Models

Each of these papers contributes significantly to the field of AI agents and machine learning. By addressing pressing challenges in areas like video compression, time series forecasting, data uniformity, self-play on zero-sum games, and human judgment in community notes, these research endeavors have the potential to drive meaningful advancements in these domains.
```

### Want to learn more about multi-agent?
There are various frameworks available, trying few of them.
- [What is a Multiagent System? | IBM] (https://www.ibm.com/think/topics/multiagent-system)
- [What are multi-agent systems? | SAP] (https://www.sap.com/bulgaria/resources/what-are-multi-agent-systems)
- [Multi agent system - HuggingFace.co] (https://huggingface.co/learn/agents-course/unit2/smolagents/multi_agent_systems)