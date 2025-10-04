<div align="center">
  <img src="assets/logo4(1).png" alt="Graph2Eval Logo" width="100"/>
</div>

# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs

<div align="center">

📄 [arXiv](https://arxiv.org/abs/2510.00507) | 🐦 [X](https://x.com/YRChen_AIsafety/status/1974361550279549192)

</div> 

## 🎯 Overview

As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation.

---

### 🔍 Key Features

- **📚 Document Processing**: Automatic document ingestion, cleaning, and chunking
- **🕸️ Graph Construction**: Build knowledge graphs from documents using GraphRAG
- **🎯 Task Generation**: Automatically generate diverse tasks from knowledge graphs
- **🤖 Multi-Agent System**: Five specialized agents (Planner, Retriever, Reasoner, Verifier, Summarizer)
- **🌐 Web Agent**: Web-based multi-hop task generation and evaluation
- **📊 Comprehensive Metrics**: Task success rate, safety compliance, attribution accuracy
- **🔧 Configurable**: Flexible configuration for different use cases

## 🏗️ Agent Architecture

The Graph2Eval system implements multiple agent architectures with flexible RAG capabilities:

### 1. Single Agent Architecture
- **No-RAG Mode**: Direct reasoning without knowledge graph retrieval
- **RAG Mode**: Enhanced with Retrieval-Augmented Generation capabilities

### 2. Multi-Agent System
A collaborative framework with five specialized agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                        │
├─────────────────────────────────────────────────────────────┤
│  Planner Agent  │  Retriever Agent  │  Reasoner Agent       │
│  (Planning)     │  (Information)    │  (Analysis)           │
├─────────────────────────────────────────────────────────────┤
│  Verifier Agent │  Summarizer Agent │                       │
│  (Validation)   │  (Synthesis)      │                       │
└─────────────────────────────────────────────────────────────┘
```

**Agent Roles:**
- **Planner**: Breaks down complex tasks into manageable steps
- **Retriever**: Searches and retrieves relevant information from knowledge graphs
- **Reasoner**: Performs logical analysis and reasoning
- **Verifier**: Validates answers and checks for consistency
- **Summarizer**: Synthesizes information into final responses

### 3. Web Agent Architecture
Specialized agents for web-based tasks:
- **Agent S 2.5**: Web agent with built-in reflection mechanism for self-improvement ([project](https://github.com/simular-ai/Agent-S))
- **SoM Agent**: Web agent that performs Set-of-marks annotation on input images for precise interaction

### Configuration Options
- **Agent Mode**: `single`, `multi`, `web`
- **Agent Type**: `no_rag`, `rag`, `agent_s`, `som_agent`
- **RAG Integration**: Optional knowledge graph retrieval for all agent types


## 📖 Citation

We would be grateful if you could cite our paper if you find this work helpful for your research:

```bibtex
@misc{chen2025graph2evalautomaticmultimodaltask,
      title={Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs}, 
      author={Yurun Chen and Xavier Hu and Yuhan Liu and Ziqi Wang and Zeyi Liao and Lin Chen and Feng Wei and Yuxi Qian and Bo Zheng and Keting Yin and Shengyu Zhang},
      year={2025},
      eprint={2510.00507},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.00507}, 
}
```

If you find this project useful, we would also appreciate a ⭐ star on GitHub!


## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone git@github.com:YurunChen/Graph2Eval.git
cd Graph2Eval

# Create conda environment
chmod +x setup_environment.sh
./setup_environment.sh 

```
This creates a conda environment named `graphrag-bench`. Activate it:

```bash
conda activate graphrag-bench
```

### 2. Configuration Setup

#### API Keys Configuration

**Option 1: Using .env file (Recommended)**

Create a `.env` file in the root directory by copying from the example file:

```bash
# Copy the example file to create your .env file
cp .env.example .env
```

Example `.env` file:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1/
OPENAI_ORGANIZATION=your-openai-organization-id

# Anthropic API Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1/

# Hugging Face API Configuration
HUGGINGFACE_API_KEY=your-huggingface-api-key-here
HUGGINGFACE_CACHE_DIR=models/huggingface
```

**Option 2: Configure directly in `configs/main_config.yaml`:**

```yaml
# configs/main_config.yaml
apis:
  # Anthropic API Configuration
  anthropic:
    api_key: your-anthropic-api-key-here
    base_url: https://api.anthropic.com/v1/
  
  # Hugging Face API Configuration
  huggingface:
    api_key: your-huggingface-api-key-here
    cache_dir: models/huggingface
  
  # OpenAI API Configuration
  openai:
    api_key: your-openai-api-key-here
    base_url: https://api.openai.com/v1/
    organization: your-openai-organization-id
```

#### Multi-Agent Configuration

Configure individual agent models in `configs/agent_config.yaml`:

```yaml
# Enable multi-agent system
enable_multi_agent: true

# Multi-agent configuration
multi_agent:
  # Individual LLM models for each agent
  planner_model: gpt-4o-mini
  retriever_model: gpt-4o-mini
  reasoner_model: gpt-4o-mini
  verifier_model: gpt-4o-mini
  summarizer_model: gpt-4o-mini
  
  # System configuration
  enable_parallel_execution: false
  enable_agent_communication: true
  max_iterations: 3
  confidence_threshold: 0.7
  verbose: true
```

### 3. Run Your First Benchmark

The benchmark system supports four main modes: `collect`, `graph`, `generate`, and `evaluate`. Here's how to use each mode:

#### Mode 1: Collect Data from Documents/URLs

```bash
# Collect data from documents
python benchmark_runner.py --mode collect \
    --documents documents/sample.pdf documents/another.pdf

# Or using short form
python benchmark_runner.py -m collect \
    -d documents/sample.pdf documents/another.pdf

# Collect data from web URLs
python benchmark_runner.py --mode collect \
    --urls https://example.com https://httpbin.org/html

# Or using short form
python benchmark_runner.py -m collect \
    -u https://example.com https://httpbin.org/html

# Custom output folder name using -n parameter
python benchmark_runner.py --mode collect \
    --documents documents/sample.pdf \
    --run-name my_custom_collection

# Or using short form
python benchmark_runner.py -m collect \
    -d documents/sample.pdf \
    -n my_custom_collection
```

#### Mode 2: Build Knowledge Graph

```bash
# Build graph from collected data
python benchmark_runner.py --mode graph \
    --collection output/collect/run_*/collections/ \
    --output-dir output/graph/run_*/

# Or using short form
python benchmark_runner.py -m graph \
    -c output/collect/run_*/collections/ \
    -o output/graph/run_*/

# Custom output folder name using -n parameter
python benchmark_runner.py --mode graph \
    --collection output/collect/run_*/collections/ \
    --output-dir output/graph/run_*/ \
    --run-name my_custom_graph

# Or using short form
python benchmark_runner.py -m graph \
    -c output/collect/run_*/collections/ \
    -o output/graph/run_*/ \
    -n my_custom_graph
```

#### Mode 3: Generate Tasks from Graph

```bash
# Generate tasks from knowledge graph
python benchmark_runner.py --mode generate \
    --graph output/graph/run_*/graph/ \
    --output-dir output/generate/run_*/datasets/

# Or using short form
python benchmark_runner.py -m generate \
    -g output/graph/run_*/graph/ \
    -o output/generate/run_*/datasets/

# Custom output folder name using -n parameter
python benchmark_runner.py --mode generate \
    --graph output/graph/run_*/graph/ \
    --output-dir output/generate/run_*/datasets/ \
    --run-name my_custom_tasks

# Or using short form
python benchmark_runner.py -m generate \
    -g output/graph/run_*/graph/ \
    -o output/generate/run_*/datasets/ \
    -n my_custom_tasks
```

#### Mode 4: Evaluate Agent Performance

```bash
# Evaluate on a single dataset file
python benchmark_runner.py --mode evaluate \
    --file output/generate/run_*/datasets/tasks.jsonl

# Or using short form
python benchmark_runner.py -m evaluate \
    -f output/generate/run_*/datasets/tasks.jsonl

# Batch evaluate on multiple datasets
python benchmark_runner.py --mode evaluate \
    --datasets-folder output/generate/run_*/datasets/ \
    --dataset-type all  # normal or all

# Or using short form
python benchmark_runner.py -m evaluate \
    -df output/generate/run_*/datasets/ \
    -t all

# Resume evaluation from existing results
python benchmark_runner.py --mode evaluate \
    --resume output/evaluate/run_20241201_120000/

# Or using short form
python benchmark_runner.py -m evaluate \
    -r output/evaluate/run_20241201_120000/

# Only evaluate existing results (skip execution)
python benchmark_runner.py --mode evaluate \
    --file output/generate/run_*/datasets/tasks.jsonl \
    --evaluate-only

# Or using short form
python benchmark_runner.py -m evaluate \
    -f output/generate/run_*/datasets/tasks.jsonl \
    -eo

# Custom output folder name using -n parameter
python benchmark_runner.py --mode evaluate \
    --file output/generate/run_*/datasets/tasks.jsonl \
    --run-name my_custom_evaluation

# Or using short form
python benchmark_runner.py -m evaluate \
    -f output/generate/run_*/datasets/tasks.jsonl \
    -n my_custom_evaluation
```

#### Complete Workflow Example

```bash
# Step 1: Collect data from documents
python benchmark_runner.py -m collect -d documents/sample.pdf

# Step 2: Build knowledge graph
python benchmark_runner.py -m graph \
    -c output/collect/run_*/collections/ \
    -o output/graph/run_*/

# Step 3: Generate tasks
python benchmark_runner.py -m generate \
    -g output/graph/run_*/graph/ \
    -o output/generate/run_*/datasets/

# Step 4: Evaluate agents
python benchmark_runner.py -m evaluate \
    -df output/generate/run_*/datasets/ \
    -t all
```

## 📁 Project Structure

```
Graph2Eval/
├── 📋 benchmark_runner.py          # Main execution script
├── ⚙️ config_manager.py            # Configuration management
├── 📚 ingestion/                   # Document processing and web collection
│   ├── parsers.py                  # Document parsers
│   ├── cleaners.py                 # Text cleaning utilities
│   ├── chunkers.py                 # Text chunking strategies
│   ├── web_collector.py            # Web data collection
│   └── tool.py                     # Ingestion tools
├── 🕸️ graph_rag/                  # Graph construction and storage
│   ├── graph_builder.py            # Knowledge graph builder
│   ├── embeddings.py               # Vector embeddings
│   ├── storage.py                  # Graph storage backends
│   ├── node_types.py               # Node type definitions
│   └── edge_types.py               # Edge type definitions
├── 🎯 task_craft/                 # Task generation and optimization
│   ├── task_generator.py           # Main task generator
│   ├── task_templates.py           # Task templates
│   ├── subgraph_sampler.py         # Subgraph sampling
│   └── task_coverage_optimizer.py  # Task optimization
├── 🤖 agent_framework/             # Agent execution framework
│   ├── agent.py                    # Base agent classes
│   ├── multi_agent_system.py       # Multi-agent coordination
│   ├── executors.py                # LLM execution
│   ├── evaluators.py               # Task evaluation
│   ├── retrievers.py               # Knowledge retrieval
│   └── attributors.py              # Attribution analysis
├── 🗂️ configs/                    # Configuration files
│   ├── main_config.yaml            # Main configuration
│   ├── agent_config.yaml           # Agent settings
│   ├── task_craft_config.yaml      # Task generation settings
│   ├── graph_rag_config.yaml       # Graph construction settings
│   ├── datasets_config.yaml        # Dataset configuration
│   ├── ingestion_config.yaml       # Data ingestion settings
├── 📊 output/                     # Results and evaluation outputs
│   ├── collect/                    # Data collection results
│   ├── graph/                      # Graph construction results
│   ├── generate/                   # Task generation results
│   └── evaluate/                   # Evaluation results
└── 🧪 logs/                       # Execution logs
```

## 🛠️ Configuration

### Core Configuration Files

| File | Purpose |
|------|---------|
| `main_config.yaml` | Main configuration file for benchmark |
| `agent_config.yaml` | Agent execution and multi-agent settings |
| `task_craft_config.yaml` | Task generation parameters |
| `graph_rag_config.yaml` | Graph construction and storage settings |
| `datasets_config.yaml` | Dataset creation and quality control |
| `ingestion_config.yaml` | Document processing and web collection settings |

### Key Configuration Parameters

```yaml
# agent_config.yaml
agent_mode: single  # single, multi, web
agent_type: rag     # no_rag, rag, agent_s, som_agent

# Single agent configuration
single_agent:
  model:
    model_name: gpt-4o-mini
    max_tokens: 4000
    temperature: 0.1
    response_format: structured

# Multi-agent configuration
multi_agent:
  planner_model: gpt-4o-mini
  retriever_model: gpt-4o-mini
  reasoner_model: gpt-4o-mini
  verifier_model: gpt-4o-mini
  summarizer_model: gpt-4o-mini

# task_craft_config.yaml
generation:
  max_total_tasks: 500
  require_gold_answer: true
  require_citations: true
  llm_model_name: "gpt-4o-mini"
  llm_temperature: 0.1
  llm_max_tokens: 4000
  use_llm_quality_check: true
  llm_quality_threshold: 0.7

# datasets_config.yaml
dataset_creation:
  save_format: "jsonl"
  max_total_samples: 5
  min_samples_per_type: 1
  min_quality_score: 0.5
  min_success_rate: 0.3

# graph_rag_config.yaml
graph_builder:
  create_chunk_nodes: true
  create_entity_nodes: true
  semantic_similarity_threshold: 0.7
  chunk_size: 500
  chunk_overlap: 50

storage:
  backend: "json"
  file_path: "output/graph/run_*/graph/knowledge_graph.json"
```

## 🎯 Usage Examples

### Command Line Usage

#### Single Document Processing
```bash
# Process a single PDF document
python benchmark_runner.py -m collect -d documents/research_paper.pdf
python benchmark_runner.py -m graph -c output/collect/run_*/collections/
python benchmark_runner.py -m generate -g output/graph/run_*/graph/
python benchmark_runner.py -m evaluate -f output/generate/run_*/datasets/tasks.jsonl
```

#### Batch Processing Multiple Documents
```bash
# Process multiple documents
python benchmark_runner.py -m collect -d documents/*.pdf documents/*.txt

# Batch evaluate multiple datasets
python benchmark_runner.py -m evaluate \
    -df output/generate/run_*/datasets/ \
    -t all
```

#### Web-based Task Generation
```bash
# Collect data from web URLs
python benchmark_runner.py -m collect -u https://example.com https://httpbin.org/html

# Continue with normal workflow
python benchmark_runner.py -m graph -c output/collect/run_*/collections/
python benchmark_runner.py -m generate -g output/graph/run_*/graph/
python benchmark_runner.py -m evaluate -df output/generate/run_*/datasets/
```

#### Resume and Debug Options
```bash
# Resume interrupted evaluation
python benchmark_runner.py -m evaluate -r output/evaluate/run_20241201_120000/

# Debug mode with verbose logging
python benchmark_runner.py -m evaluate \
    -f output/generate/run_*/datasets/tasks.jsonl \
    --debug

# Only evaluate existing results (skip execution)
python benchmark_runner.py -m evaluate \
    -f output/generate/run_*/datasets/tasks.jsonl \
    -eo
```

### Configuration Examples

#### Single Agent with RAG
```yaml
# agent_config.yaml
agent_mode: single
agent_type: rag
single_agent:
  model:
    model_name: gpt-4o-mini
    max_tokens: 4000
    temperature: 0.1
    response_format: structured
```

#### Multi-Agent System
```yaml
# agent_config.yaml
agent_mode: multi
agent_type: rag
multi_agent:
  planner_model: gpt-4o-mini
  retriever_model: gpt-4o-mini
  reasoner_model: gpt-4o-mini
  verifier_model: gpt-4o-mini
  summarizer_model: gpt-4o-mini
```

#### Web Agent Configuration
```yaml
# agent_config.yaml
agent_mode: web
agent_type: agent_s  # or som_agent
web_agent:
  model:
    model_name: gpt-4o-mini
    max_tokens: 4000
    temperature: 0.1
```

## 📊 Output Structure

The system creates organized output directories for each mode:

### Collect Mode Output
```
output/collect/run_collect_*/
├── documents/                          # Processed document images
│   └── images/
│       ├── page_1_image_0.png
│       ├── page_1_image_1.png
│       ├── page_12_image_0.png
│       └── ...
├── results/                            # Collection results
│   └── document_collection_results.json
└── web_info/                           # Web collection data (if applicable)
    └── (web-related files)
```

### Graph Mode Output
```
output/graph/run_graph_*
├── graph/
│   └── knowledge_graph.json           # Built knowledge graph
├── vectors/                            # Vector embeddings
│   ├── vectors_faiss.faiss
│   ├── vectors_faiss.metadata
│   └── vectors_nodes.pkl
└── results/
    └── graph_build_results.json
```

### Generate Mode Output
```
output/generate/run_gen_*/
├── datasets/                           # Generated task datasets
│   └── web_tasks.jsonl
├── graph/                              # Knowledge graph data
│   └── knowledge_graph.json
├── subgraphs/                          # Subgraph samples
│   ├── web_element_centric_unknown_subgraphs.json
│   └── web_subgraphs_summary.json
├── vectors/                            # Vector embeddings
│   ├── vectors_faiss.faiss
│   ├── vectors_faiss.metadata
│   └── vectors_nodes.pkl
└── results/
    └── task_generation_results.json
```

### Evaluate Mode Output
```
output/evaluate/run_eval_*/
└── results/
    ├── individual_results/             # Individual task results
    │   ├── task_*.json                 # Individual task result files
    │   └── ...
    ├── batch_evaluation/               # Batch evaluation results
    │   ├── batch_evaluation_results_*.json
    │   ├── batch_evaluation_summary_*.json
    │   ├── detailed_task_metrics_*.csv
    │   ├── overall_metrics_*.csv
    │   └── task_type_metrics_*.csv
    ├── evaluation_results.json         # Main evaluation results
    ├── full_res.csv                    # Full results CSV
    └── summary.csv                     # Summary results
```

### Data Flow Structure

```
output/
├── collect/run_*/collections/          # Collected document data
│   ├── documents.jsonl
│   └── web_data.jsonl
├── graph/run_*/graph/                  # Knowledge graph data
│   ├── knowledge_graph.json
│   └── vectors/
│       ├── vectors_faiss.faiss
│       └── vectors_faiss.metadata
└── generate/run_*/datasets/            # Generated task datasets
    ├── all_tasks.jsonl
    ├── normal_tasks.jsonl
    └── safety_tasks.jsonl
```

## 🤝 Contact

For questions, issues, or contributions:

📧 Email: [yurunchen.research@gmail.com](mailto:yurunchen.research@gmail.com)
🐛 Issues: [GitHub Issues](https://github.com/YurunChen/Graph2Eval/issues)


---

**⭐ Star this repository if you find it helpful!**
