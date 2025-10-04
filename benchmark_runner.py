#!/usr/bin/env python3
"""
重构版的Benchmark Runner - 分离数据集生成和评估功能
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm

from loguru import logger

# Import our configuration system
from config_manager import get_config

# Import task-related modules
from task_craft.task_templates import TaskType

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class BenchmarkRunner:
    
    def __init__(self, config_dir: str = "configs", mode: str = "generate", run_name: str = None, existing_output_dir: str = None):
        self.config_dir = config_dir
        self.mode = mode  # "generate" or "evaluate"
        self.config = get_config(config_dir)
        
        # Create timestamp for this run first
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_timestamp_int = int(time.time())
        self.custom_run_name = run_name
        self.existing_output_dir = existing_output_dir
        
        # Setup logging (now run_timestamp is available)
        self._setup_logging()
        
        # Create unified output directories once at initialization
        if existing_output_dir and mode == "evaluate":
            # Use existing output directory for resume
            self.output_dirs = self._use_existing_output_directories(existing_output_dir)
        else:
            self.output_dirs = self._create_output_directories()
        
        # Initialize components based on mode
        self.components = {}
        self._initialize_components()
        
        # Parser output directories are now set during initialization
        
        # Track execution state
        self.start_time = None
        self.results = {}
    
    def _get_timestamped_path(self, base_path: str) -> str:
        """Get path with timestamp prefix"""
        output_base = self._get_output_base_dir()
        if str(base_path).startswith(str(output_base) + '/'):
            # Insert timestamp after output base
            return str(base_path).replace(f'{output_base}/', f'{output_base}/run_{self.run_timestamp}/')
        else:
            # Add output/run_timestamp prefix
            return f'{output_base}/run_{self.run_timestamp}/{base_path}'
    
    def _get_output_base_dir(self) -> Path:
        """Get output base directory from configuration"""
        main_config_path = Path(self.config_dir) / "main_config.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r', encoding='utf-8') as f:
                import yaml
                main_config_data = yaml.safe_load(f)
                orchestration_config = main_config_data.get('orchestration', {})
        else:
            orchestration_config = {}
        
        output_config = orchestration_config.get('output', {})
        return Path(output_config.get('base_dir', 'output'))
    
    def _create_output_directories(self) -> Dict[str, Path]:
        """Create unified output directory structure with timestamp based on mode"""
        # Use existing timestamp if available, otherwise create new one
        if hasattr(self, 'run_timestamp_int'):
            timestamp = self.run_timestamp_int
        else:
            timestamp = int(time.time())
            
        # Create mode-specific subdirectory
        output_base = self._get_output_base_dir()
        mode_dir = output_base / self.mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory under mode directory
        if self.custom_run_name:
            # Use custom run name if provided
            timestamped_dir = mode_dir / self.custom_run_name
        else:
            # Use default timestamp-based naming
            if self.mode == "collect":
                timestamped_dir = mode_dir / f"run_collect_{timestamp}"
            elif self.mode == "graph":
                timestamped_dir = mode_dir / f"run_graph_{timestamp}"
            elif self.mode == "generate":
                timestamped_dir = mode_dir / f"run_gen_{timestamp}"
            elif self.mode == "evaluate":
                timestamped_dir = mode_dir / f"run_eval_{timestamp}"
            else:
                timestamped_dir = mode_dir / f"run_{timestamp}"

        timestamped_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories based on mode
        if self.mode == "collect":
            directories = {
                "base": timestamped_dir,
                "web_info": timestamped_dir / "web_info",
                "documents": timestamped_dir / "documents",  # For saving extracted images
                "results": timestamped_dir / "results"
            }
        elif self.mode == "graph":
            directories = {
                "base": timestamped_dir,
                "graph": timestamped_dir / "graph",
                "vectors": timestamped_dir / "vectors",
                "results": timestamped_dir / "results"
            }
        elif self.mode == "generate":
            directories = {
                "base": timestamped_dir,
                "datasets": timestamped_dir / "datasets",
                "subgraphs": timestamped_dir / "subgraphs",
                "results": timestamped_dir / "results"
            }
        elif self.mode == "evaluate":
            directories = {
                "base": timestamped_dir,
                "results": timestamped_dir / "results",
                "evaluation": timestamped_dir / "evaluation",
                "file_images": timestamped_dir / "file_images"
            }
        else:
            # Fallback for unknown modes
            directories = {
                "base": timestamped_dir,
                "results": timestamped_dir / "results"
            }
        
        # Create all directories
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Output directory created: {timestamped_dir}")

        # Parser output directories are now set during component initialization

        return directories

    def _use_existing_output_directories(self, existing_dir: str) -> Dict[str, Path]:
        """Use existing output directories for resume functionality"""
        existing_path = Path(existing_dir)
        
        if not existing_path.exists():
            raise ValueError(f"Existing output directory not found: {existing_dir}")
        
        # Create directory structure based on existing directory
        directories = {
            "base": existing_path,
            "results": existing_path / "results",
            "evaluation": existing_path / "evaluation",
            "file_images": existing_path / "file_images"
        }
        
        # Ensure all directories exist
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 Using existing output directory: {existing_path}")
        logger.info(f"📁 Results directory: {directories['results']}")
        
        return directories

    def _auto_detect_latest_collection(self) -> Optional[str]:
        """Auto-detect the latest collection directory"""
        try:
            collection_base = self._get_output_base_dir() / "collect"
            if not collection_base.exists():
                return None

            # Find all run_collect_* directories
            run_dirs = []
            for item in collection_base.iterdir():
                if item.is_dir() and item.name.startswith("run_collect_"):
                    try:
                        # Extract timestamp from directory name
                        timestamp_str = item.name.replace("run_collect_", "")
                        timestamp = int(timestamp_str)
                        run_dirs.append((timestamp, item))
                    except ValueError:
                        continue

            if not run_dirs:
                return None

            # Sort by timestamp (newest first) and return the latest
            run_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_dir = run_dirs[0][1]

            # Check if it contains valid collection data
            if (latest_dir / "results" / "web_collection_results.json").exists() or \
               (latest_dir / "results" / "document_collection_results.json").exists() or \
               (latest_dir / "web_info").exists():
                return str(latest_dir)

            return None

        except Exception as e:
            logger.warning(f"Failed to auto-detect collection directory: {e}")
            return None

    def _auto_detect_latest_graph(self) -> Optional[str]:
        """Auto-detect the latest graph directory"""
        try:
            graph_base = self._get_output_base_dir() / "graph"
            if not graph_base.exists():
                return None

            # Find all run_graph_* directories
            run_dirs = []
            for item in graph_base.iterdir():
                if item.is_dir() and item.name.startswith("run_graph_"):
                    try:
                        # Extract timestamp from directory name
                        timestamp_str = item.name.replace("run_graph_", "")
                        timestamp = int(timestamp_str)
                        run_dirs.append((timestamp, item))
                    except ValueError:
                        continue

            if not run_dirs:
                return None

            # Sort by timestamp (newest first) and return the latest
            run_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_dir = run_dirs[0][1]

            # Check if it contains valid graph data
            if (latest_dir / "graph" / "knowledge_graph.json").exists() or \
               (latest_dir / "results" / "graph_construction_results.json").exists():
                return str(latest_dir)

            return None

        except Exception as e:
            logger.warning(f"Failed to auto-detect graph directory: {e}")
            return None

    def _auto_detect_latest_generate(self) -> Optional[str]:
        """Auto-detect the latest generate directory"""
        try:
            generate_base = self._get_output_base_dir() / "generate"
            if not generate_base.exists():
                return None

            # Find all run_gen_* directories
            run_dirs = []
            for item in generate_base.iterdir():
                if item.is_dir() and item.name.startswith("run_gen_"):
                    try:
                        # Extract timestamp from directory name
                        timestamp_str = item.name.replace("run_gen_", "")
                        timestamp = int(timestamp_str)
                        run_dirs.append((timestamp, item))
                    except ValueError:
                        continue

            if not run_dirs:
                logger.info("No generate directories found")
                return None

            # Sort by timestamp and get the latest
            run_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_dir = run_dirs[0][1]

            # Verify it contains datasets
            datasets_dir = latest_dir / "datasets"
            if datasets_dir.exists() and any(datasets_dir.iterdir()):
                logger.info(f"🎯 Auto-detected generate directory: {latest_dir}")
                return str(latest_dir)

            return None

        except Exception as e:
            logger.warning(f"Failed to auto-detect generate directory: {e}")
            return None

    def _update_parser_output_dirs(self, images_dir: Path):
        """Update parser output directories for image saving (legacy method)"""
        try:
            # Update PDF parser
            if 'pdf_parser' in self.components:
                self.components['pdf_parser'].output_dir = images_dir
                logger.info(f"✅ Updated PDF parser output directory: {images_dir}")
            
            # Update HTML parser
            if 'html_parser' in self.components:
                self.components['html_parser'].output_dir = images_dir
                logger.info(f"✅ Updated HTML parser output directory: {images_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to update parser output directories: {e}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        logger.remove()
        
        # Read configuration from main_config.yaml
        main_config_path = Path(self.config_dir) / "main_config.yaml"
        if main_config_path.exists():
            try:
                with open(main_config_path, 'r', encoding='utf-8') as f:
                    import yaml
                    main_config_data = yaml.safe_load(f)
                    orchestration_config = main_config_data.get('orchestration', {})
            except Exception as e:
                logger.warning(f"Failed to read main_config.yaml: {e}, using default logging config")
                orchestration_config = {}
        else:
            logger.warning(f"main_config.yaml not found at {main_config_path}, using default logging config")
            orchestration_config = {}
        
        log_level = orchestration_config.get('monitoring', {}).get('log_level', 'INFO')
        
        # Always add console logging
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Always add file log handler to ensure logs are saved while running
        log_dir = Path(orchestration_config.get('monitoring', {}).get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Create log file using current run timestamp and mode
        log_file = log_dir / f"benchmark_{self.mode}_{self.run_timestamp}.log"
        
        # Read real-time write configuration
        real_time_write = orchestration_config.get('monitoring', {}).get('real_time_write', True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=not real_time_write  # Decide whether to enable queue based on configuration
        )
        
        logger.info(f"Logging setup completed - Console: INFO+, File: {log_file}")
        logger.info(f"Log level: {log_level}, Log directory: {log_dir}")
    
    
    def _get_agent_model_config(self, agent_mode: str, agent_type: str) -> Dict[str, Any]:
        """Get model configuration based on agent_mode and agent_type"""
        if agent_mode == 'single':
            # Use single_agent configuration
            single_agent_config = self.config.agent.get('single_agent', {})
            return single_agent_config.get('model', {})
        elif agent_mode == 'multi':
            # Use multi_agent planner configuration
            multi_agent_config = self.config.agent.get('multi_agent', {})
            planner_config = multi_agent_config.get('agents', {}).get('planner', {})
            return planner_config.get('model', {})
        elif agent_mode == 'web':
            # Use web agent configuration based on agent_type
            if agent_type == 'agent_s':
                agent_s_config = self.config.agent.get('agent_s_web', {})
                return agent_s_config.get('model', {})
            else:
                web_agent_config = self.config.agent.get('web_agent', {})
                return web_agent_config.get('model', {})
        else:
            # Fallback to execution configuration
            return self.config.agent.get('execution', {})

    def _initialize_components(self, mode: str = None, collect_type: str = None):
        """Unified component initialization method"""
        if mode is None:
            mode = self.mode
            
        logger.info(f"🔧 Initializing components for {mode} mode...")
        
        # Initialize components based on mode
        if mode == "collect":
            # Collect mode needs different components based on input type
            if collect_type == "documents":
                logger.info("📄 Collect mode (documents): Initializing ingestion components")
                self._initialize_ingestion_components()
            elif collect_type == "urls":
                logger.info("🌐 Collect mode (URLs): Initializing minimal components")
                # No additional components needed for URL collection
            else:
                logger.info("📋 Collect mode: Initializing minimal components")
                # No additional components needed for collect mode
        elif mode == "graph":
            # Graph mode needs GraphRAG components
            logger.info("🕸️ Graph mode: Initializing GraphRAG components")
            self._initialize_graph_rag_components()
        elif mode == "generate":
            # Generate mode only needs minimal components for task generation
            logger.info("🎯 Generate mode: Initializing minimal components for task generation")
            # No additional components needed - task generation uses existing graph data
        elif mode == "evaluate":
            # Evaluate mode needs Agent and GraphRAG components
            logger.info("📊 Evaluate mode: Initializing Agent and GraphRAG components")
            self._initialize_agent_components()
            self._initialize_graph_rag_components()
        else:
            # Default: initialize all components for backward compatibility
            logger.info("🔄 Default mode: Initializing all components")
            self._initialize_agent_components()
            self._initialize_graph_rag_components()
            if mode == "generate":
                self._initialize_ingestion_components()

    def _initialize_ingestion_components(self):
        """Initialize ingestion components (only for generate mode)"""
        logger.info("📖 Initializing ingestion components...")
        
        try:
            from ingestion.parsers import PDFParser, HTMLParser
            from ingestion.chunkers import SemanticChunker, HierarchicalChunker  
            from ingestion.cleaners import TextCleaner, CleaningRules
            
            # Configure cleaning rules
            cleaning_rules = CleaningRules(
                **self.config.ingestion.get('cleaning', {})
            )
            
            # Initialize parsers and cleaners with proper output directory
            image_config = self.config.ingestion.get('image_processing', {})
            
            # Determine the correct output directory for images
            if self.mode == "collect" and "documents" in self.output_dirs:
                # For collect mode, save images to documents/images subdirectory
                images_output_dir = self.output_dirs["documents"] / "images"
            elif self.mode == "evaluate" and "file_images" in self.output_dirs:
                # For evaluate mode, save images to file_images directory
                images_output_dir = self.output_dirs["file_images"]
            else:
                # Fallback to default data/images directory
                images_output_dir = Path("data/images")
            
            self._initialize_component('pdf_parser', PDFParser, config_section=None,
                                     extract_tables=self.config.ingestion.get('parsing', {}).get('pdf_extract_tables', True),
                                     extract_images=self.config.ingestion.get('parsing', {}).get('pdf_extract_images', False),
                                     image_config=image_config,
                                     output_dir=images_output_dir)
            
            self._initialize_component('html_parser', HTMLParser, config_section=None,
                                     extract_links=self.config.ingestion.get('parsing', {}).get('html_extract_links', True),
                                     extract_images=self.config.ingestion.get('parsing', {}).get('html_extract_images', False),
                                     image_config=image_config,
                                     output_dir=images_output_dir)
            
            self._initialize_component('text_cleaner', TextCleaner, config_section=None, rules=cleaning_rules)
            
            # Initialize chunker
            chunking_config = self.config.ingestion.get('chunking', {})
            if chunking_config.get('enabled', True):
                chunker_class = SemanticChunker if chunking_config.get('strategy', 'semantic') == "semantic" else HierarchicalChunker
                self._initialize_component('chunker', chunker_class, config_section=None,
                                         max_chunk_size=chunking_config.get('chunk_size', 1000),
                                         overlap_size=chunking_config.get('overlap_size', 100))
            
            logger.info("✅ Ingestion components initialized")
            
        except ImportError as e:
            logger.warning(f"Some ingestion components not available: {e}")
        
    def _initialize_graph_rag_components(self):
        """Initialize GraphRAG components (only for generate mode)"""
        logger.info("🕸️ Initializing GraphRAG components...")
        
        try:
            from graph_rag.graph_builder import GraphBuilder, GraphBuildConfig
            from graph_rag.embeddings import EmbeddingManager, EmbeddingConfig
            from graph_rag.storage import JSONStorage
            
            # Initialize embedding manager
            embedding_config = EmbeddingConfig(
                **self.config.graph_rag.get('embeddings', {})
            )
            self.components['embedding_manager'] = EmbeddingManager(config=embedding_config)
            logger.info("✅ embedding_manager initialized")
            
            # Initialize storage
            storage_config = self.config.graph_rag.get('storage', {})
            storage_backend = storage_config.get('backend', 'json')
            if storage_backend == "json":
                storage_path = storage_config.get('file_path', 'data/graphs/graph.json')
                self.components['storage'] = JSONStorage(file_path=storage_path)
                logger.info("✅ storage initialized")
            else:
                logger.warning(f"Unsupported storage backend: {storage_backend}")
                self.components['storage'] = JSONStorage()
                logger.info("✅ storage initialized")
            
            # Initialize graph builder
            graph_build_config = GraphBuildConfig(
                **self.config.graph_rag.get('graph_building', {})
            )
            self.components['graph_builder'] = GraphBuilder(
                config=graph_build_config,
                embedding_manager=self.components['embedding_manager'],
                storage=self.components['storage']
            )
            logger.info("✅ graph_builder initialized")
            
            logger.info("✅ GraphRAG components initialized")
            
        except ImportError as e:
            logger.warning(f"GraphRAG components not available: {e}")

    def _initialize_agent_components(self):
        """Initialize Agent components (shared between generate and evaluate modes)"""
        logger.info("🤖 Initializing Agent components...")
        
        try:
            from agent_framework.retrievers import HybridRetriever, RetrievalConfig
            from agent_framework.executors import LLMExecutor, ExecutionConfig
            from agent_framework.agent import RAGAgent, AgentConfig
            from agent_framework.no_rag_agent import NoRAGAgent, NoRAGAgentConfig
            from agent_framework.multi_agent_system import MultiAgentSystem, create_multi_agent_system
            
            # Get agent configuration
            agent_mode = self.config.agent.get('agent_mode', 'single')  # single, multi
            agent_type = self.config.agent.get('agent_type', 'rag')     # no_rag, rag, som_agent (only used when mode is single)
            
            # Initialize LLMExecutor first (shared across all components)
            logger.info("🔧 Initializing LLMExecutor")
            
            # Get model config using unified method
            model_config = self._get_agent_model_config(agent_mode, agent_type)
            logger.info(f"🔍 Model config: {model_config}")
            logger.info(f"🔍 Max tokens from config: {model_config.get('max_tokens', 'NOT_FOUND')}")
            logger.info(f"🔍 Model name from config: {model_config.get('model_name', 'NOT_FOUND')}")
            logger.info(f"🔍 Agent mode: {agent_mode}, Agent type: {agent_type}")
            
            execution_config = ExecutionConfig(
                model_name=model_config.get('model_name', 'gpt-4o-mini'),
                model_provider=model_config.get('model_provider', 'openai'),
                temperature=model_config.get('temperature', 0.1),
                max_tokens=model_config.get('max_tokens', 4000),
                timeout=model_config.get('timeout', 30),
                max_retries=model_config.get('max_retries', 2),
                response_format=model_config.get('response_format', 'json'),
            )
            logger.info(f"🔍 Execution config max_tokens: {execution_config.max_tokens}")
            
            # Create new executor instance with correct configuration
            self.components['executor'] = LLMExecutor(execution_config)
            logger.info(f"✅ LLMExecutor initialized with model: {execution_config.model_name}, response_format: {execution_config.response_format}")
            
            if agent_mode == 'multi':
                # Initialize Multi-Agent System
                logger.info("🤖 Initializing Multi-Agent System")
                
                multi_agent_config = self.config.agent.get('multi_agent', {})
                system_config = self.config.agent.get('multi_agent', {}).get('system', {})
                
                # Use shared executor for Multi-Agent System
                shared_executor = self.components.get('executor')
                self.components['agent'] = create_multi_agent_system(
                    multi_agent_config=multi_agent_config,
                    system_config=system_config,
                    agent_type=agent_type,
                    executor=shared_executor
                )
                logger.info("✅ Multi-Agent System initialized")
                
            elif agent_type == 'no_rag':
                # Initialize No-RAG Agent
                logger.info("🤖 Initializing No-RAG Agent")
                
                # Get model config using unified method
                model_config = self._get_agent_model_config(agent_mode, agent_type)
                
                # Get agent config
                if agent_mode == 'single':
                    single_agent_config = self.config.agent.get('single_agent', {})
                    agent_config_data = single_agent_config.get('agent', {})
                else:
                    agent_config_data = {}
                
                no_rag_config = NoRAGAgentConfig(
                    model_name=model_config.get('model_name', 'gpt-4o-mini'),
                    model_provider=model_config.get('model_provider', 'openai'),
                    temperature=model_config.get('temperature', 0.1),
                    max_tokens=model_config.get('max_tokens', 1000),
                    timeout=model_config.get('timeout', 30),
                    max_retries=model_config.get('max_retries', 3),
                    retry_delay=1.0,  # Default value
                    max_context_length=4000,  # Default value
                    verbose=agent_config_data.get('verbose', False),
                    require_citations=model_config.get('require_citations', False),
                    require_reasoning=model_config.get('require_reasoning', False),
                    response_format=model_config.get('response_format', 'text')
                )
                
                # Use the shared executor that was already initialized
                self.components['agent'] = NoRAGAgent(no_rag_config, executor=self.components['executor'])
                logger.info("✅ No-RAG Agent initialized")
                
            else:
                # Initialize RAG Agent (default)
                logger.info("🤖 Initializing RAG Agent")
                
                # Configure retrieval
                if agent_mode == 'single':
                    retrieval_config_data = self.config.agent.get('single_agent', {}).get('retrieval', {})
                else:
                    retrieval_config_data = self.config.agent.get('retrieval', {})
                retrieval_config = RetrievalConfig(
                    max_nodes=retrieval_config_data.get('max_nodes', 10),
                    max_hops=retrieval_config_data.get('max_hops', 3),
                    similarity_threshold=retrieval_config_data.get('similarity_threshold', 0.7),
                    include_neighbors=retrieval_config_data.get('include_neighbors', True),
                    expand_with_context=retrieval_config_data.get('expand_with_context', True),
                    prefer_gold_nodes=retrieval_config_data.get('prefer_gold_nodes', True)
                )
                
                self.components['retriever'] = HybridRetriever(retrieval_config)
                
                # Get model config using unified method
                model_config = self._get_agent_model_config(agent_mode, agent_type)
                
                # Configure execution
                execution_config = ExecutionConfig(
                    model_name=model_config.get('model_name', 'gpt-4o-mini'),
                    model_provider=model_config.get('model_provider', 'openai'),
                    temperature=model_config.get('temperature', 0.1),
                    max_tokens=model_config.get('max_tokens', 1000),
                    timeout=model_config.get('timeout', 30),
                    max_retries=model_config.get('max_retries', 3),
                    retry_delay=1.0,  # Default value
                    require_citations=model_config.get('require_citations', True),
                    require_reasoning=model_config.get('require_reasoning', False),
                    response_format=model_config.get('response_format', 'structured'),
                    max_context_length=4000  # Default value
                )
                
                self.components['executor'] = LLMExecutor(execution_config)
                
                # Initialize RAG Agent
                if agent_mode == 'single':
                    system_config = self.config.agent.get('single_agent', {}).get('agent', {})
                else:
                    system_config = self.config.agent.get('system', {})
                rag_agent_config = AgentConfig(
                    retriever_type="hybrid",
                    executor_type="llm",
                    enable_evaluation=system_config.get('enable_evaluation', True),
                    retrieval_config=retrieval_config,
                    execution_config=execution_config,
                    max_context_length=model_config.get('max_context_length', 4000),
                    enable_memory=system_config.get('enable_memory', True),
                    memory_size=system_config.get('memory_size', 10),
                    verbose=system_config.get('verbose', False),
                    log_intermediate=system_config.get('log_intermediate', False)
                )
                
                # Use shared executor for RAG Agent
                shared_executor = self.components.get('executor')
                self.components['agent'] = RAGAgent(rag_agent_config, shared_executor)
                logger.info("✅ RAG Agent initialized")
                
                # Note: Graph will be set during task execution stage
            
        except ImportError as e:
            logger.warning(f"Agent components not available: {e}")
        
        # Initialize Safety components (for safety task generation only) - COMMENTED OUT
        # try:
        #     from task_craft.safety_task_generator import SafetyTaskGenerator
        #     
        #     # Configure safety task generation
        #     safety_config = self.config.safety
        #     safety_generation_config = safety_config.get('safety_task_generation', {})
        #     graph_based_config = safety_generation_config.get('graph_based_generation', {})
        #     
        #     # Initialize safety task generator
        #     # Get task generation config for safety tasks
        #     task_craft_config = self.config.task_craft
        #     generation_config = task_craft_config.get('generation', {})
        #     
        #     # Get safety task generation config
        #     safety_task_config = safety_config.get('safety_task_generation', {})
        #     
        #     safety_config = {
        #         **graph_based_config,
        #         'require_citations': generation_config.get('require_citations', True),
        #         'require_reasoning': generation_config.get('require_reasoning', False),
        #         'max_tasks_per_rule': safety_task_config.get('max_tasks_per_rule', 2),
        #         'max_tasks_per_threat': safety_task_config.get('max_tasks_per_threat', 2),
        #         'max_total_safety_tasks': safety_task_config.get('max_total_safety_tasks', 20),
        #         'difficulty_levels': safety_task_config.get('difficulty_levels', ['easy', 'medium', 'hard'])
        #     }
        #     
        #     self.components['safety_generator'] = SafetyTaskGenerator(
        #         config=safety_config
        #     )
        #     
        #     # Dataset manager functionality integrated directly
        #     
        #     logger.info("✅ Safety task generation components initialized")
        #     
        # except ImportError as e:
        #     logger.warning(f"Safety components not available: {e}")
        
        logger.info("⚠️ Safety task generation components disabled")
        
        # Initialize GraphRAG components (minimal for evaluation - only for loading graphs)
        try:
            from graph_rag.embeddings import EmbeddingManager, EmbeddingConfig
            from graph_rag.storage import JSONStorage
            
            # Configure embedding manager (minimal config for loading)
            graph_rag_config = self.config.graph_rag
            embeddings_config = graph_rag_config.get('embeddings', {})
            embedding_config = EmbeddingConfig(
                model_name=embeddings_config.get('model_name', 'all-MiniLM-L6-v2'),
                batch_size=embeddings_config.get('batch_size', 32),
                max_seq_length=embeddings_config.get('max_seq_length', 512),
                normalize_embeddings=embeddings_config.get('normalize_embeddings', True),
                cache_embeddings=embeddings_config.get('cache_embeddings', True),
                device=embeddings_config.get('device', 'auto')
            )
            
            self.components['embedding_manager'] = EmbeddingManager(embedding_config)
            
            # Configure storage (minimal for loading)
            storage_config = graph_rag_config.get('storage', {})
            storage_backend = storage_config.get('backend', 'json')
            if storage_backend == "json":
                storage_path = storage_config.get('file_path', 'data/graphs/graph.json')
                self.components['storage'] = JSONStorage(file_path=storage_path)
                logger.info("✅ storage initialized")
            else:
                logger.warning(f"Unsupported storage backend: {storage_backend}")
                self.components['storage'] = JSONStorage()
                logger.info("✅ storage initialized")
            
            logger.info("✅ GraphRAG components initialized (minimal for evaluation)")
            
        except ImportError as e:
            logger.warning(f"GraphRAG components not available: {e}")
        
        logger.info("✅ Evaluation components initialization completed")
    
    def _initialize_component(self, component_name: str, component_class, config_section: str = None, **kwargs):
        """Generic component initialization method"""
        try:
            # Get config section if specified
            if config_section:
                if hasattr(self.config, config_section):
                    config = getattr(self.config, config_section)
                else:
                    config = {}
            else:
                config = {}
            
            # Check if component_class accepts config parameter
            import inspect
            sig = inspect.signature(component_class.__init__)
            if 'config' in sig.parameters:
                component = component_class(config=config, **kwargs)
            else:
                component = component_class(**kwargs)
            self.components[component_name] = component
            logger.info(f"✅ {component_name} initialized")
            return component
        except Exception as e:
            logger.warning(f"Failed to initialize {component_name}: {e}")
            return None
            
    def generate_dataset_from_documents(self, input_documents: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate dataset from input documents (including graph construction and task generation)"""
        logger.info("🚀 Starting dataset generation from documents")
        logger.info(f"📄 Processing {len(input_documents)} documents")
        
        self.start_time = time.time()
        
        # Use existing output directories
        output_dirs = self.output_dirs
        datasets_dir = output_dirs["datasets"]
        
        # Get agent configuration
        agent_mode = self.config.agent.get('agent_mode', 'single')
        agent_type = self.config.agent.get('agent_type', 'rag')
        
        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": agent_mode,
                "agent_type": agent_type,
                "max_tasks": self.config.task_craft.get('generation', {}).get('max_total_tasks', 5),
                "storage_backend": self.config.graph_rag.get('storage', {}).get('backend', 'json')
            },
            "documents": input_documents,
            "stages": {}
        }
        
        try:
            # Stage 1: Document Ingestion
            logger.info("📖 Stage: Document Ingestion")
            ingestion_results = self._run_ingestion_stage(input_documents)
            results["stages"]["ingestion"] = ingestion_results
            
            # Stage 2: Graph Construction
            logger.info("🕸️ Stage: Graph Construction")
            graph_results = self._run_graph_construction_stage(ingestion_results, str(datasets_dir))
            results["stages"]["graph_construction"] = graph_results
            
            # Stage 3: Task Generation
            logger.info("🎯 Stage: Task Generation")
            task_results = self._run_task_generation_stage(graph_results, str(datasets_dir))
            results["stages"]["task_generation"] = task_results
            
            results["success"] = True
            results["total_time"] = time.time() - self.start_time
            
            # Save results to unified output directory
            if output_dir:
                self._save_dataset_generation_results(results, output_dir)
            else:
                self._save_dataset_generation_results(results, str(datasets_dir))
            
            logger.info(f"✅ Dataset generation completed successfully in {results['total_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Dataset generation failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise
        
        return results

    def _normalize_urls(self, urls: List[str]) -> List[str]:
        """Normalize URLs to ensure they have proper protocol"""
        normalized_urls = []
        
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            original_url = url
            
            # If URL doesn't start with http:// or https://, add https://
            if not url.startswith(('http://', 'https://')):
                # Handle common cases
                if url.startswith('www.'):
                    url = 'https://' + url
                elif url.startswith('//'):
                    # Protocol-relative URL, add https:
                    url = 'https:' + url
                elif '.' in url and not url.startswith('/'):
                    # Assume it's a domain name (contains dots and doesn't start with slash)
                    # Check if it looks like a domain (has at least one dot and valid characters)
                    import re
                    if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', url):
                        url = 'https://' + url
                    else:
                        # Keep as is if it doesn't look like a domain
                        pass
                else:
                    # Keep as is if it's a relative path or other format
                    pass
            
            normalized_urls.append(url)
            if original_url != url:
                logger.debug(f"🌐 Normalized URL: {original_url} → {url}")
        
        return normalized_urls

    async def collect_web_data(self, urls: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Collect web data from URLs (separated collection stage)"""
        logger.info(f"{'='*60}")
        logger.info("🌐 WEB DATA COLLECTION STAGE")
        logger.info(f"{'='*60}")
        
        # Normalize URLs to handle various formats
        normalized_urls = self._normalize_urls(urls)
        logger.info(f"🌐 Processing {len(normalized_urls)} URLs (normalized from {len(urls)} input URLs)")
        
        # Show URL normalization results
        if len(normalized_urls) != len(urls):
            logger.info("🌐 URL normalization applied:")
            for original, normalized in zip(urls, normalized_urls):
                if original != normalized:
                    logger.info(f"   {original} → {normalized}")

        self.start_time = time.time()

        # Use existing output directories or create new ones
        if output_dir:
            output_base = Path(output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            web_info_dir = output_base / "web_info"
            web_info_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dirs = self.output_dirs
            web_info_dir = output_dirs["web_info"]

        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": self.config.agent.get('agent_mode', 'web'),
            },
            "urls": normalized_urls,
            "stages": {}
        }

        try:
            # Import Web Collector components
            from ingestion.web_collector import WebCollector

            # Stage 1: Web Page Collection with Exploration
            logger.info("🌐 Stage: Web Page Collection with Multi-Step Exploration")
            stage_start = time.time()

            # Create web collection config
            web_collection_config = self.config.ingestion.get('web_collection', {}).copy()
            web_collection_config['output_dir'] = str(web_info_dir)
            web_collector = WebCollector(web_collection_config)

            # Use exploration mode for multi-step cross-page collection
            max_depth = web_collection_config.get('exploration', {}).get('max_depth', 10)
            max_pages_per_depth = web_collection_config.get('exploration', {}).get('max_pages_per_depth', 20)

            logger.debug(f"🔍 Web collection config: max_depth={max_depth}, max_pages_per_depth={max_pages_per_depth}")
            logger.debug(f"🔍 Starting web collection for URLs: {normalized_urls}")

            # Check if click simulation is enabled
            if web_collection_config.get('enable_click_simulation', False):
                logger.info("🖱️ Using enhanced web collection with click simulation")
                web_pages = await web_collector.collect_with_click_simulation(normalized_urls)
            else:
                logger.info("📋 Using standard web collection")
                web_pages = await web_collector.collect_web_data_with_exploration(
                    normalized_urls, max_depth=max_depth, max_pages_per_depth=max_pages_per_depth
                )

            logger.debug(f"🔍 Web collection completed: {len(web_pages)} pages collected")
            for i, page in enumerate(web_pages[:3]):  # Show first 3 pages
                logger.debug(f"🔍 Page {i}: URL={page.url}, title={page.title}, elements={len(page.elements) if page.elements else 0}")

            # Save web collection results
            results["stages"]["web_collection"] = {
                "collected_pages": len(web_pages),
                "pages": [page.to_dict() for page in web_pages],
                "processing_time": time.time() - stage_start,
                "saved_files": {
                    "web_info_dir": str(web_info_dir),
                    "dom_files_dir": str(web_info_dir)
                }
            }

            results["success"] = True
            results["total_time"] = time.time() - self.start_time

            # Save collection results directly
            save_dir = output_dir if output_dir else str(output_dirs["base"])
            try:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Create results subdirectory
                results_subdir = output_path / "results"
                results_subdir.mkdir(parents=True, exist_ok=True)

                # Save main results
                results_file = results_subdir / "web_collection_results.json"

                # Custom JSON encoder to handle numpy types and Path objects
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, Path):
                            return str(obj)
                        return super().default(obj)

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                logger.info(f"💾 Web collection results saved to: {results_file}")

            except Exception as e:
                logger.error(f"❌ Failed to save web collection results: {e}")

            logger.info(f"✅ Web data collection completed successfully in {results['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"❌ Web data collection failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise

        return results

    def collect_document_data(self, input_documents: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Collect document data from files (separated collection stage)"""
        logger.info(f"{'='*60}")
        logger.info("📄 DOCUMENT DATA COLLECTION STAGE")
        logger.info(f"{'='*60}")
        logger.info(f"📄 Processing {len(input_documents)} documents")

        self.start_time = time.time()

        # Use existing output directories or create new ones
        if output_dir:
            output_base = Path(output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            output_dirs = {
                "base": output_base,
                "documents": output_base / "documents",
                "datasets": output_base / "datasets"
            }
            for dir_path in output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
        else:
            output_dirs = self.output_dirs

        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": self.config.agent.get('agent_mode', 'single'),
            },
            "documents": input_documents,
            "stages": {}
        }

        try:
            # Stage 1: Document Ingestion
            logger.info("📖 Stage: Document Ingestion")
            ingestion_results = self._run_ingestion_stage(input_documents)
            
            # Create a copy for saving (without raw_documents)
            ingestion_results_for_save = ingestion_results.copy()
            ingestion_results_for_save.pop("raw_documents", None)
            results["stages"]["ingestion"] = ingestion_results_for_save

            results["success"] = True
            results["total_time"] = time.time() - self.start_time

            # Save collection results directly
            save_dir = output_dir if output_dir else str(output_dirs["base"])
            try:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Create results subdirectory
                results_subdir = output_path / "results"
                results_subdir.mkdir(parents=True, exist_ok=True)

                # Save main results
                results_file = results_subdir / "document_collection_results.json"

                # Custom JSON encoder to handle numpy types and Path objects
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, Path):
                            return str(obj)
                        return super().default(obj)

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                logger.info(f"💾 Document collection results saved to: {results_file}")

            except Exception as e:
                logger.error(f"❌ Failed to save document collection results: {e}")

            logger.info(f"✅ Document data collection completed successfully in {results['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"❌ Document data collection failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise

        return results

    def _setup_output_directories(self, output_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Setup output directories for graph construction"""
        if output_dir:
            output_base = Path(output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            graph_dir = output_base / "graph"
            vectors_dir = output_base / "vectors"
            graph_dir.mkdir(parents=True, exist_ok=True)
            vectors_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dirs = self.output_dirs
            graph_dir = output_dirs["graph"]
            vectors_dir = output_dirs["vectors"]
        
        return graph_dir, vectors_dir

    def _init_graph_construction_results(self, collection_path: str) -> Dict[str, Any]:
        """Initialize results structure for graph construction"""
        return {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": self.config.agent.get('agent_mode', 'single'),
                "storage_backend": self.config.graph_rag.get('storage', {}).get('backend', 'json')
            },
            "collection_path": collection_path,
            "stages": {}
        }

    def _get_graph_statistics(self, graph) -> Dict[str, Any]:
        """Get statistics from graph (handle both TaskGraph and DocumentGraph)"""
        if hasattr(graph, 'nodes'):
            # TaskGraph with @property
            nodes_dict = graph.nodes
            edges_dict = graph.edges
        else:
            # DocumentGraph with storage
            nodes_dict = graph.storage.nodes
            edges_dict = graph.storage.edges
        
        return {
            "total_nodes": len(nodes_dict),
            "total_edges": len(edges_dict),
            "node_types": {node_type.value: len([n for n in nodes_dict.values() if n.node_type == node_type]) 
                          for node_type in set(n.node_type for n in nodes_dict.values())},
            "edge_types": {edge_type.value: len([e for e in edges_dict.values() if e.edge_type == edge_type]) 
                          for edge_type in set(e.edge_type for e in edges_dict.values())}
        }

    def build_graph_from_collection(self, collection_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Build graph from collected data (separated graph construction stage)"""
        logger.info(f"{'='*60}")
        logger.info("🕸️ GRAPH CONSTRUCTION FROM COLLECTION STAGE")
        logger.info(f"{'='*60}")
        logger.info(f"🕸️ Collection path: {collection_path}")

        self.start_time = time.time()
        
        # Setup output directories
        graph_dir, vectors_dir = self._setup_output_directories(output_dir)
        
        # Initialize results structure
        results = self._init_graph_construction_results(collection_path)

        try:
            collection_path_obj = Path(collection_path)

            # Determine collection type and load data
            if (collection_path_obj / "results" / "web_collection_results.json").exists():
                # Web collection - load from results JSON
                logger.info("🌐 Detected web collection data")
                from graph_rag.graph_builder import GraphBuilder, WebGraphBuildConfig
                from ingestion.web_collector import WebPageData, WebElement

                # Load web pages from collection results
                results_file = collection_path_obj / "results" / "web_collection_results.json"
                with open(results_file, 'r', encoding='utf-8') as f:
                    collection_results = json.load(f)

                # Reconstruct WebPageData objects from JSON
                web_pages = []
                for page_data in collection_results.get("stages", {}).get("web_collection", {}).get("pages", []):
                    elements = []
                    for elem_data in page_data.get("elements", []):
                        element = WebElement(**elem_data)
                        elements.append(element)

                    page = WebPageData(
                        url=page_data["url"],
                        title=page_data["title"],
                        elements=elements,
                        screenshots=page_data.get("screenshots", [])
                    )
                    web_pages.append(page)

                logger.info(f"🌐 Loaded {len(web_pages)} web pages from collection")

                # Stage 1: Web Graph Construction
                logger.info("🕸️ Stage: Web Graph Construction")
                stage_start = time.time()

                logger.debug(f"🕸️ Starting web graph construction with {len(web_pages)} pages")

                # Create web-specific graph configuration
                web_graph_config = WebGraphBuildConfig()
                logger.debug(f"🕸️ Web graph config created: {web_graph_config}")

                # Create web-specific storage and embedding manager
                from graph_rag.storage import JSONStorage
                from graph_rag.embeddings import EmbeddingManager

                web_storage = JSONStorage()
                web_embedding_manager = EmbeddingManager()
                logger.debug("🕸️ Storage and embedding manager created")

                graph_builder = GraphBuilder(
                    config=web_graph_config,
                    embedding_manager=web_embedding_manager,
                    storage=web_storage
                )
                logger.debug("🕸️ Graph builder created, starting build_task_graph_from_web_data")

                # Extract web_elements from web_pages
                web_elements = []
                for page in web_pages:
                    if hasattr(page, 'elements') and page.elements:
                        web_elements.extend(page.elements)

                logger.debug(f"🕸️ Extracted {len(web_elements)} elements from {len(web_pages)} pages")

                # Use TaskGraph to build web data graph
                web_graph = graph_builder.build_task_graph_from_web_data(web_pages, web_elements)
                logger.debug("🕸️ TaskGraph built successfully")

            elif (collection_path_obj / "results").exists() and (collection_path_obj / "results" / "document_collection_results.json").exists():
                # Document collection
                logger.info("📄 Detected document collection data")
                
                # Load document collection results
                collection_results_file = collection_path_obj / "results" / "document_collection_results.json"
                with open(collection_results_file, 'r', encoding='utf-8') as f:
                    collection_results = json.load(f)

                # Re-run ingestion to get raw documents
                input_documents = collection_results.get("documents", [])
                ingestion_results = self._run_ingestion_stage(input_documents)

                # Stage 1: Graph Construction
                logger.info("🕸️ Stage: Graph Construction")
                stage_start = time.time()

                graph_results = self._run_graph_construction_stage(ingestion_results, str(graph_dir))
                
                # Get the first (and only) DocumentGraph from results
                graphs = graph_results.get("graphs", [])
                if not graphs:
                    raise ValueError("No graphs created from document collection")
                
                document_graph = graphs[0]  # Use the actual DocumentGraph directly

            else:
                raise ValueError(f"Could not determine collection type for path: {collection_path}")

            # Determine which graph we have and save it
            if 'web_graph' in locals():
                # Web data -> TaskGraph
                graph = web_graph
                graph_type = "TaskGraph"
                logger.info(f"🌐 Using TaskGraph for web data with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            elif 'document_graph' in locals():
                # Text data -> DocumentGraph  
                graph = document_graph
                graph_type = "DocumentGraph"
                logger.info(f"📄 Using DocumentGraph for text data with {len(graph.storage.nodes)} nodes and {len(graph.storage.edges)} edges")
            else:
                raise ValueError("No graph was created")

            # Save graph to file (including vector indexes)
            graph.save(str(graph_dir), str(vectors_dir))
            logger.debug(f"🕸️ {graph_type} and vectors saved to {graph_dir} and {vectors_dir}")

            # Get graph statistics
            graph_stats = self._get_graph_statistics(graph)
            
            results["stages"]["graph_construction"] = {
                "graphs_created": 1,
                **graph_stats,
                "processing_time": time.time() - stage_start,
                "saved_files": {
                    "knowledge_graph": str(graph_dir / "knowledge_graph.json"),
                    "graph_dir": str(graph_dir),
                    "vectors_dir": str(vectors_dir) if vectors_dir.exists() else None,
                    "vectors_faiss": str(vectors_dir / "vectors_faiss.faiss") if (vectors_dir / "vectors_faiss.faiss").exists() else None,
                    "vectors_metadata": str(vectors_dir / "vectors_faiss.metadata") if (vectors_dir / "vectors_faiss.metadata").exists() else None,
                    "vectors_nodes": str(vectors_dir / "vectors_nodes.pkl") if (vectors_dir / "vectors_nodes.pkl").exists() else None
                }
            }

            results["success"] = True
            results["total_time"] = time.time() - self.start_time

            # Save graph construction results directly
            save_dir = output_dir if output_dir else str(self.output_dirs["base"])
            try:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Create results subdirectory
                results_subdir = output_path / "results"
                results_subdir.mkdir(parents=True, exist_ok=True)

                # Save main results
                results_file = results_subdir / "graph_construction_results.json"

                # Custom JSON encoder to handle numpy types and Path objects
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, Path):
                            return str(obj)
                        return super().default(obj)

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                logger.info(f"💾 Graph construction results saved to: {results_file}")

            except Exception as e:
                logger.error(f"❌ Failed to save graph construction results: {e}")

            logger.info(f"✅ Graph construction completed successfully in {results['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"❌ Graph construction failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise

        return results

    def generate_tasks_from_graph(self, graph_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate tasks from graph data (separated task generation stage)"""
        logger.info("🎯 Starting task generation from graph")
        logger.info(f"🎯 Graph path: {graph_path}")

        self.start_time = time.time()

        # Use existing output directories or create new ones
        if output_dir:
            output_base = Path(output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            datasets_dir = output_base / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dirs = self.output_dirs
            datasets_dir = output_dirs["datasets"]
            output_base = output_dirs["base"]  # Use base as output_base for generate mode
            
        # Create graph and vector directories in generate folder for backup
        generate_graph_dir = output_base / "graph"
        generate_vectors_dir = output_base / "vectors"
        generate_graph_dir.mkdir(parents=True, exist_ok=True)
        generate_vectors_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": self.config.agent.get('agent_mode', 'single'),
                "max_tasks": self.config.task_craft.get('generation', {}).get('max_total_tasks', 50),
            },
            "graph_path": graph_path,
            "stages": {}
        }

        try:
            graph_path_obj = Path(graph_path)

            # Determine agent mode first to choose appropriate graph loading method
            agent_mode = self.config.agent.get('agent_mode', 'single')
            
            # Load graph data based on agent mode
            if (graph_path_obj / "results" / "graph_construction_results.json").exists():
                # Load from graph construction results
                logger.info("🕸️ Detected graph construction results")
                
                if agent_mode == 'web':
                    # Use TaskGraph for web tasks
                    logger.info("🌐 Using TaskGraph for web task generation")
                    from graph_rag.graph_builder import TaskGraph
                    
                    # Load the graph data using TaskGraph
                    graph_dir = graph_path_obj / "graph"
                    vector_dir = graph_path_obj / "vectors"
                    graph = TaskGraph.load(str(graph_dir), str(vector_dir))
                    
                    logger.info(f"🎯 Loaded TaskGraph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
                    
                    # Save TaskGraph to generate folder for backup
                    graph.save(str(generate_graph_dir), str(generate_vectors_dir))
                    logger.info(f"💾 TaskGraph saved to generate folder: {generate_graph_dir}")
                    
                else:
                    # Use DocumentGraph for text tasks
                    logger.info("📄 Using DocumentGraph for text task generation")
                    from graph_rag.graph_builder import DocumentGraph
                    from graph_rag.storage import JSONStorage
                    from graph_rag.embeddings import EmbeddingManager

                    # Create DocumentGraph instance
                    storage = JSONStorage()
                    embedding_manager = EmbeddingManager()
                    graph = DocumentGraph(storage, embedding_manager)

                    # Load the graph data
                    graph_dir = graph_path_obj / "graph"
                    vector_dir = graph_path_obj / "vectors"
                    graph.load(str(graph_dir), str(vector_dir))
                    
                    # Save DocumentGraph to generate folder for backup
                    graph.save(str(generate_graph_dir), str(generate_vectors_dir))
                    logger.info(f"💾 DocumentGraph saved to generate folder: {generate_graph_dir}")
                    
                    logger.info(f"🎯 Loaded DocumentGraph with {len(graph.storage.nodes)} nodes and {len(graph.storage.edges)} edges")
            else:
                raise ValueError(f"Could not determine graph type for path: {graph_path}")

            # Stage 1: Task Generation
            logger.info("🎯 Stage: Task Generation")
            stage_start = time.time()

            if agent_mode == 'web':
                # Web task generation
                logger.info("🌐 Using web task generation mode")
                web_task_config = self.config.task_craft.get('web_task_generation', {})

                # Create TaskGenerator with its own LLM configuration from task_craft_config.yaml
                from task_craft.task_generator import TaskGenerator, TaskGenerationConfig

                # Create proper TaskGenerationConfig
                task_config = TaskGenerationConfig.from_config()
                # Get current run directory for image path updates
                current_run_dir = self.output_dirs["base"]
                # Let TaskGenerator create its own LLMExecutor from task_craft config
                task_generator = TaskGenerator(config=task_config, current_run_dir=current_run_dir)

                max_web_tasks = self.config.task_craft.get('generation', {}).get('max_total_tasks', 50)

                logger.debug(f"Debug: About to generate {max_web_tasks} web tasks")
                logger.debug(f"Debug: TaskGenerator config: {task_config}")

                try:
                    logger.debug("🎯 Starting web task generation with TaskGraph")

                    # Directly use TaskGraph for web task generation
                    web_tasks_result = task_generator.generate_web_tasks_with_task_graph(graph, max_web_tasks)

                    # Handle return format
                    if isinstance(web_tasks_result, dict):
                        web_tasks = web_tasks_result.get("tasks", [])
                        subgraph_stats = web_tasks_result.get("subgraph_stats")
                        detailed_subgraphs = web_tasks_result.get("detailed_subgraphs", [])
                    else:
                        web_tasks = web_tasks_result
                        subgraph_stats = None
                        detailed_subgraphs = []

                    logger.debug(f"Debug: Generated {len(web_tasks)} web tasks")

                except Exception as e:
                    logger.error(f"Debug: Error in task generation: {e}")
                    import traceback
                    logger.error(f"Debug: Traceback: {traceback.format_exc()}")
                    raise

                # Convert tasks to dictionaries
                task_dicts = []
                logger.info(f"Processing {len(web_tasks)} tasks")
                for i, task in enumerate(web_tasks):
                    logger.info(f"Task {i}: type={type(task)}")
                    if hasattr(task, 'to_dict'):
                        logger.info(f"Task {i}: has to_dict method")
                        task_dicts.append(task.to_dict())
                    elif isinstance(task, dict):
                        logger.info(f"Task {i}: is dict")
                        task_dicts.append(task)
                    else:
                        logger.warning(f"Task {i}: is neither dict nor has to_dict method: {type(task)}")
                        continue

                results["stages"]["task_generation"] = {
                    "generated_tasks": len(web_tasks),
                    "total_tasks": len(web_tasks),
                    "tasks": task_dicts,
                    "subgraph_stats": subgraph_stats,
                    "detailed_subgraphs": detailed_subgraphs,
                    "processing_time": time.time() - stage_start
                }

            else:
                # Regular document task generation
                logger.info("📄 Using document task generation mode")

                # Create TaskGenerator with its own LLM configuration from task_craft_config.yaml
                from task_craft.task_generator import TaskGenerator, TaskGenerationConfig
                task_config = TaskGenerationConfig.from_config()
                current_run_dir = self.output_dirs["base"]

                # Extract graph nodes for safety task generation
                graph_nodes = []
                # DocumentGraph uses storage.nodes, TaskGraph has direct nodes attribute
                if hasattr(graph, 'nodes'):
                    nodes_dict = graph.nodes
                elif hasattr(graph, 'storage') and hasattr(graph.storage, 'nodes'):
                    nodes_dict = graph.storage.nodes
                else:
                    nodes_dict = {}
                
                for node in nodes_dict.values():
                    graph_nodes.append({
                        'id': node.node_id,
                        'content': node.content,
                        'type': node.node_type.value,
                        'metadata': node.metadata
                    })

                # Generate normal tasks
                normal_tasks = []

                # Create progress bar for normal task generation
                # Let TaskGenerator create its own LLMExecutor from task_craft config
                task_generator = TaskGenerator(config=task_config, current_run_dir=current_run_dir)

                # Generate tasks from graph nodes
                # Use graph path name as source_document for proper graph association
                graph_name = graph_path_obj.stem  # Extract the graph directory name (e.g., "run_graph_1757483826")
                normal_tasks = task_generator.generate_tasks(graph, graph_name)

                all_tasks = normal_tasks

                results["stages"]["task_generation"] = {
                    "generated_tasks": len(normal_tasks),
                    "safety_tasks_generated": 0,
                    "total_tasks": len(all_tasks),
                    "tasks": [task.to_dict() if hasattr(task, 'to_dict') else task for task in all_tasks],
                    "processing_time": time.time() - stage_start
                }

            results["success"] = True
            results["total_time"] = time.time() - self.start_time

            # Save task generation results directly
            save_dir = output_dir if output_dir else str(output_dirs["base"])
            try:
                output_path = Path(save_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Create results subdirectory
                results_subdir = output_path / "results"
                results_subdir.mkdir(parents=True, exist_ok=True)

                # Save main results
                results_file = results_subdir / "task_generation_results.json"

                # Custom JSON encoder to handle numpy types and Path objects
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, Path):
                            return str(obj)
                        elif hasattr(obj, 'to_dict'):
                            return obj.to_dict()
                        elif hasattr(obj, '__dict__'):
                            result = {}
                            for key, value in obj.__dict__.items():
                                if key in ['subgraph', 'metapath_instance']:
                                    continue  # Skip complex graph objects
                                elif hasattr(value, 'to_dict'):
                                    result[key] = value.to_dict()
                                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                    result[key] = value
                                else:
                                    result[key] = str(value)
                            return result
                        return super().default(obj)

                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                logger.info(f"💾 Task generation results saved to: {results_file}")

                # Save tasks to datasets directory based on task type
                task_generation = results.get("stages", {}).get("task_generation", {})
                tasks = task_generation.get("tasks", [])

                if tasks:
                    if agent_mode == 'web':
                        # Web tasks - use TaskJSONEncoder to handle complex objects
                        all_tasks_file = datasets_dir / "web_tasks.jsonl"
                        
                        # Custom JSON encoder for web tasks
                        class WebTaskJSONEncoder(json.JSONEncoder):
                            def default(self, obj):
                                if hasattr(obj, 'to_dict'):
                                    return obj.to_dict()
                                elif hasattr(obj, '__dict__'):
                                    result = {}
                                    for key, value in obj.__dict__.items():
                                        if key in ['subgraph', 'metapath_instance']:
                                            continue  # Skip complex graph objects
                                        elif hasattr(value, 'to_dict'):
                                            result[key] = value.to_dict()
                                        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                            result[key] = value
                                        else:
                                            # Debug: Log problematic objects
                                            logger.warning(f"Converting non-serializable object in key '{key}': {type(value)} - {value}")
                                            result[key] = str(value)
                                    return result
                                elif isinstance(obj, (str, int, float, bool, list, dict, type(None))):
                                    return obj
                                else:
                                    # Debug: Log the problematic object type
                                    logger.error(f"❌ Non-serializable object type: {type(obj)} - {obj}")
                                    return str(obj)
                        
                        with open(all_tasks_file, 'w', encoding='utf-8') as f:
                            for task in tasks:
                                f.write(json.dumps(task, ensure_ascii=False, cls=WebTaskJSONEncoder) + '\n')
                        logger.info(f"🌐 Web tasks saved to: {all_tasks_file}")
                    else:
                        # Text tasks - use TaskJSONEncoder for complex objects
                        all_tasks_file = datasets_dir / "text_tasks.jsonl"
                        
                        # Custom JSON encoder for text tasks
                        class TaskJSONEncoder(json.JSONEncoder):
                            def default(self, obj):
                                if hasattr(obj, 'to_dict'):
                                    return obj.to_dict()
                                elif hasattr(obj, '__dict__'):
                                    result = {}
                                    for key, value in obj.__dict__.items():
                                        if key in ['subgraph', 'metapath_instance']:
                                            continue  # Skip complex graph objects
                                        elif hasattr(value, 'to_dict'):
                                            result[key] = value.to_dict()
                                        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                            result[key] = value
                                        else:
                                            # Debug: Log problematic objects
                                            logger.warning(f"Converting non-serializable object in key '{key}': {type(value)} - {value}")
                                            result[key] = str(value)
                                    return result
                                elif isinstance(obj, (str, int, float, bool, list, dict, type(None))):
                                    return obj
                                else:
                                    # Debug: Log the problematic object type
                                    logger.error(f"❌ Non-serializable object type: {type(obj)} - {obj}")
                                    return str(obj)
                        
                        with open(all_tasks_file, 'w', encoding='utf-8') as f:
                            for task in tasks:
                                f.write(json.dumps(task, ensure_ascii=False, cls=TaskJSONEncoder) + '\n')
                        logger.info(f"📄 Text tasks saved to: {all_tasks_file}")

            except Exception as e:
                logger.error(f"❌ Failed to save task generation results: {e}")

            logger.info(f"✅ Task generation completed successfully in {results['total_time']:.2f}s")

        except Exception as e:
            logger.error(f"❌ Task generation failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise

        return results

    async def generate_web_dataset_from_urls(self, urls: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate web dataset from input URLs"""
        logger.info("🚀 Starting web dataset generation from URLs")
        logger.info(f"🌐 Processing {len(urls)} URLs")
        
        self.start_time = time.time()
        
        # Use existing output directories
        output_dirs = self.output_dirs
        web_info_dir = output_dirs["web_info"]
        graph_dir = output_dirs["graph"]
        vectors_dir = output_dirs["vectors"]
        datasets_dir = output_dirs["datasets"]
        
        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": self.config.agent.get('agent_mode', 'web'),
                "agent_type": self.config.agent.get('agent_type', 'no_rag'),
                "max_tasks": self.config.task_craft.get('generation', {}).get('max_total_tasks', 50),
                "storage_backend": self.config.graph_rag.get('storage', {}).get('backend', 'json')
            },
            "urls": urls,
            "stages": {}
        }
        
        try:
            # Import Web Agent components
            from ingestion.web_collector import WebCollector
            from graph_rag.graph_builder import GraphBuilder, WebGraphBuildConfig
            from task_craft.task_generator import TaskGenerator
            
            # Stage 1: Web Page Collection with Exploration
            logger.info("🌐 Stage: Web Page Collection with Multi-Step Exploration")
            stage_start = time.time()
            
            # Create web collection config with custom output directory
            web_collection_config = self.config.ingestion.get('web_collection', {}).copy()
            web_collection_config['output_dir'] = str(web_info_dir)
            web_collector = WebCollector(web_collection_config)
            
            # Use exploration mode for multi-step cross-page collection
            web_collection_config = self.config.ingestion.get('web_collection', {})
            max_depth = web_collection_config.get('exploration', {}).get('max_depth', 10)
            max_pages_per_depth = web_collection_config.get('exploration', {}).get('max_pages_per_depth', 20)
            
            logger.debug(f"🔍 Web collection config: max_depth={max_depth}, max_pages_per_depth={max_pages_per_depth}")
            logger.debug(f"🔍 Starting web collection for URLs: {urls}")
            
            # Check if click simulation is enabled
            if web_collection_config.get('enable_click_simulation', False):
                logger.info("🖱️ Using enhanced web collection with click simulation")
                web_pages = await web_collector.collect_with_click_simulation(urls)
            else:
                logger.info("📋 Using standard web collection")
                web_pages = await web_collector.collect_web_data_with_exploration(
                    urls, max_depth=max_depth, max_pages_per_depth=max_pages_per_depth
                )
            
            logger.debug(f"🔍 Web collection completed: {len(web_pages)} pages collected")
            for i, page in enumerate(web_pages[:3]):  # Show first 3 pages
                logger.debug(f"🔍 Page {i}: URL={page.url}, title={page.title}, elements={len(page.elements) if page.elements else 0}")
            # Use the unified output directory for web collection
            web_output_path = web_info_dir
            
            results["stages"]["web_collection"] = {
                "collected_pages": len(web_pages),
                "pages": [page.to_dict() for page in web_pages],
                "processing_time": time.time() - stage_start,
                "saved_files": {
                                    "web_info_dir": str(web_info_dir),
                "dom_files_dir": str(web_info_dir)
                }
            }
            
            # Stage 2: Web Graph Construction
            logger.info("🕸️ Stage: Web Graph Construction")
            stage_start = time.time()
            
            logger.debug(f"🕸️ Starting web graph construction with {len(web_pages)} pages")
            
            # Create web-specific graph configuration
            web_graph_config = WebGraphBuildConfig()
            logger.debug(f"🕸️ Web graph config created: {web_graph_config}")
            
            # Create web-specific storage and embedding manager with custom paths
            from graph_rag.storage import JSONStorage
            from graph_rag.embeddings import EmbeddingManager
            
            web_storage = JSONStorage()
            web_embedding_manager = EmbeddingManager()
            logger.debug(f"🕸️ Storage and embedding manager created")
        
            graph_builder = GraphBuilder(
                config=web_graph_config,
                embedding_manager=web_embedding_manager,
                storage=web_storage
            )
            logger.debug(f"🕸️ Graph builder created, starting build_task_graph_from_web_data")
            
            # Extract web_elements from web_pages
            web_elements = []
            for page in web_pages:
                if hasattr(page, 'elements') and page.elements:
                    web_elements.extend(page.elements)
            
            logger.debug(f"🕸️ Extracted {len(web_elements)} elements from {len(web_pages)} pages")
            
            # Use new graph-task abstraction system to build TaskGraph
            task_graph = graph_builder.build_task_graph_from_web_data(web_pages, web_elements)
            logger.debug(f"🕸️ TaskGraph built successfully")
            
            # Save TaskGraph to file (including vector index)
            task_graph.save(str(graph_dir), str(vectors_dir))
            logger.debug(f"🕸️ TaskGraph and vectors saved to {graph_dir} and {vectors_dir}")
            
            results["stages"]["web_graph_construction"] = {
                "total_nodes": len(task_graph.nodes),
                "total_edges": len(task_graph.edges),
                "node_types": {node_type.value: len([n for n in task_graph.nodes.values() if n.node_type == node_type]) for node_type in set(n.node_type for n in task_graph.nodes.values())},
                "edge_types": {edge_type.value: len([e for e in task_graph.edges.values() if e.edge_type == edge_type]) for edge_type in set(e.edge_type for e in task_graph.edges.values())},
                "processing_time": time.time() - stage_start,
                "saved_files": {
                    "knowledge_graph": str(graph_dir / "knowledge_graph.json"),
                    "graph_dir": str(graph_dir),
                    "vectors_dir": str(vectors_dir) if vectors_dir.exists() else None,
                    "vectors_faiss": str(vectors_dir / "vectors_faiss.faiss") if (vectors_dir / "vectors_faiss.faiss").exists() else None,
                    "vectors_metadata": str(vectors_dir / "vectors_faiss.metadata") if (vectors_dir / "vectors_faiss.metadata").exists() else None,
                    "vectors_nodes": str(vectors_dir / "vectors_nodes.pkl") if (vectors_dir / "vectors_nodes.pkl").exists() else None
                }
            }
            
            # Stage 3: Web Task Generation
            logger.info("🎯 Stage: Web Task Generation")
            logger.debug(f"Debug: TaskGraph stats: {len(task_graph.nodes)} nodes, {len(task_graph.edges)} edges")
            logger.debug(f"Debug: TaskGraph website_type: {task_graph.website_type}")
            logger.debug(f"Debug: TaskGraph website_description: {task_graph.website_description}")
            
            stage_start = time.time()
            web_task_config = self.config.task_craft.get('web_task_generation', {})
            # Create TaskGenerator with its own LLM configuration from task_craft_config.yaml
            from task_craft.task_generator import TaskGenerator, TaskGenerationConfig
            
            # Create proper TaskGenerationConfig
            task_config = TaskGenerationConfig.from_config()
            # Get current run directory for image path updates
            current_run_dir = self.output_dirs["base"]
            # Let TaskGenerator create its own LLMExecutor from task_craft config
            task_generator = TaskGenerator(config=task_config, current_run_dir=current_run_dir)
            max_web_tasks = self.config.task_craft.get('generation', {}).get('max_total_tasks', 50)  # Use generation config
            
            logger.debug(f"Debug: About to generate {max_web_tasks} web tasks")
            logger.debug(f"Debug: TaskGenerator config: {task_config}")
            
            try:
                logger.debug(f"🎯 Starting web task generation with new abstraction system")
                
                logger.debug(f"🎯 Using existing TaskGraph with {len(task_graph.nodes)} nodes and {len(task_graph.edges)} edges")
                logger.debug(f"🎯 TaskGraph node types: {[n.node_type.value for n in task_graph.nodes.values()]}")
                logger.debug(f"🎯 TaskGraph edge types: {[e.edge_type.value for e in task_graph.edges.values()]}")
                
                # Use new graph-task abstraction system to generate tasks
                web_tasks_result = task_generator.generate_web_tasks_with_task_graph(task_graph, max_web_tasks)
                
                # Handle new return format
                if isinstance(web_tasks_result, dict):
                    web_tasks = web_tasks_result.get("tasks", [])
                    subgraph_stats = web_tasks_result.get("subgraph_stats")
                    detailed_subgraphs = web_tasks_result.get("detailed_subgraphs", [])
                else:
                    # Backward compatibility: if old format is returned (direct task list)
                    web_tasks = web_tasks_result
                    subgraph_stats = None
                    detailed_subgraphs = []
                
                logger.debug(f"Debug: Generated {len(web_tasks)} web tasks")
                if web_tasks:
                    for i, task in enumerate(web_tasks[:3]):  # Show first 3 tasks
                        logger.debug(f"Debug: Task {i} type: {type(task)}")
            except Exception as e:
                logger.error(f"Debug: Error in task generation: {e}")
                import traceback
                logger.error(f"Debug: Traceback: {traceback.format_exc()}")
                raise
            
            # Remove safety task generation, only use normal web tasks
            all_web_tasks = web_tasks
            
            # Convert tasks to dictionaries safely
            task_dicts = []
            logger.info(f"Processing {len(all_web_tasks)} tasks")
            for i, task in enumerate(all_web_tasks):
                logger.info(f"Task {i}: type={type(task)}")
                if hasattr(task, 'to_dict'):
                    logger.info(f"Task {i}: has to_dict method")
                    task_dicts.append(task.to_dict())
                elif isinstance(task, dict):
                    logger.info(f"Task {i}: is dict")
                    task_dicts.append(task)
                else:
                    logger.warning(f"Task {i}: is neither dict nor has to_dict method: {type(task)}")
                    continue
            
            # Subgraph statistics have been extracted above, no need to repeat here
            # subgraph_stats and detailed_subgraphs have been obtained from web_tasks_result above
            
            results["stages"]["web_task_generation"] = {
                "generated_tasks": len(web_tasks),
                "total_tasks": len(all_web_tasks),
                "tasks": task_dicts,
                "subgraph_stats": subgraph_stats,
                "detailed_subgraphs": detailed_subgraphs,
                "processing_time": time.time() - stage_start
            }
            
            results["success"] = True
            results["total_time"] = time.time() - self.start_time
            
            # Save results to unified datasets directory
            self._save_web_dataset_generation_results(results, str(datasets_dir))
            
            logger.info(f"✅ Web dataset generation completed successfully in {results['total_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Web dataset generation failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            results["total_time"] = time.time() - self.start_time
            raise
        
        return results

    def _save_web_dataset_generation_results(self, results: Dict[str, Any], output_dir: str):
        """Save web dataset generation results to specified directory"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create results subdirectory
            results_subdir = output_path / "results"
            results_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = results_subdir / "web_dataset_generation_results.json"
            
            # Custom JSON encoder to handle numpy types and Path objects
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, Path):
                        return str(obj)
                    return super().default(obj)
            
            logger.info(f"Debug: About to serialize results, keys: {list(results.keys())}")
            logger.info(f"Debug: stages keys: {list(results.get('stages', {}).keys())}")
            logger.info(f"Debug: web_task_generation keys: {list(results.get('stages', {}).get('web_task_generation', {}).keys())}")
            
            serializable_results = self._make_serializable(results)
            logger.info(f"Debug: Made results serializable, type: {type(serializable_results)}")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            # Save web graph data (already saved by graph_builder)
            graph_results = results.get("stages", {}).get("web_graph_construction", {})
            if graph_results:
                web_graph_file = output_path / "web_graph.json"
                with open(web_graph_file, 'w', encoding='utf-8') as f:
                    json.dump(graph_results, f, indent=2, ensure_ascii=False)
            
            # Save subgraphs to graph folder
            task_results = results.get("stages", {}).get("web_task_generation", {})
            tasks = task_results.get("tasks", [])
            subgraph_stats = task_results.get("subgraph_stats", {})
            
            # Get graph directory from output directories
            graph_dir = self.output_dirs["graph"]
            subgraphs_dir = graph_dir / "subgraphs"
            subgraphs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save subgraphs from subgraph_stats if available - INDEPENDENT of task validation
            subgraph_files = []
            if subgraph_stats and isinstance(subgraph_stats, dict):
                # Get detailed subgraph information from the task generation results
                detailed_subgraphs = task_results.get("detailed_subgraphs", [])
                
                # Create a comprehensive summary with detailed subgraph information
                summary_subgraph_data = {
                    "subgraph_stats": subgraph_stats,
                    "total_subgraphs": subgraph_stats.get("total_subgraphs", 0),
                    "total_nodes": subgraph_stats.get("total_subgraph_nodes", 0),
                    "total_edges": subgraph_stats.get("total_subgraph_edges", 0),
                    "average_size": subgraph_stats.get("average_subgraph_size", 0),
                    "generated_at": datetime.now().isoformat()
                }
                
                # Always save subgraph summary if we have stats
                summary_file = subgraphs_dir / "subgraphs_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_subgraph_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 Saved subgraphs summary to {summary_file}")
                subgraph_files.append(str(summary_file))
                
                # Save detailed subgraph information if available
                if detailed_subgraphs:
                    summary_subgraph_data["detailed_subgraphs"] = []
                    for i, subgraph in enumerate(detailed_subgraphs):
                        if isinstance(subgraph, dict):
                            detailed_info = {
                                "subgraph_id": f"subgraph_{i+1}",
                                "nodes": subgraph.get("nodes", {}),
                                "edges": subgraph.get("edges", {}),
                                "node_count": len(subgraph.get("nodes", {})),
                                "edge_count": len(subgraph.get("edges", {})),
                                "node_types": self._get_node_type_distribution(subgraph.get("nodes", {})),
                                "edge_types": self._get_edge_type_distribution(subgraph.get("edges", {})),
                                "sampling_strategy": subgraph.get("sampling_strategy", "unknown"),
                                "seed_pattern": subgraph.get("seed_pattern", "unknown"),
                                "executability_score": subgraph.get("executability_score", 0.0)
                            }
                            summary_subgraph_data["detailed_subgraphs"].append(detailed_info)
                            
                            # Save individual detailed subgraph
                            subgraph_file = subgraphs_dir / f"detailed_subgraph_{i+1}.json"
                            with open(subgraph_file, 'w', encoding='utf-8') as f:
                                json.dump(detailed_info, f, indent=2, ensure_ascii=False)
                            subgraph_files.append(str(subgraph_file))
                            logger.info(f"💾 Saved detailed subgraph {i+1}: {subgraph_file}")
                else:
                    logger.warning("⚠️  No detailed subgraphs available in task results")
                    
                    # Try to extract subgraph data from subgraph_stats if detailed_subgraphs is empty
                    if subgraph_stats.get("total_subgraphs", 0) > 0:
                        logger.info("🔄 Attempting to extract subgraph data from subgraph_stats...")
                        # Create a basic subgraph file from available stats
                        basic_subgraph_data = {
                            "subgraph_id": "basic_subgraph",
                            "subgraph_stats": subgraph_stats,
                            "note": "Basic subgraph data extracted from stats (detailed data not available)",
                            "generated_at": datetime.now().isoformat()
                        }
                        
                        basic_subgraph_file = subgraphs_dir / "basic_subgraph.json"
                        with open(basic_subgraph_file, 'w', encoding='utf-8') as f:
                            json.dump(basic_subgraph_data, f, indent=2, ensure_ascii=False)
                        subgraph_files.append(str(basic_subgraph_file))
                        logger.info(f"💾 Saved basic subgraph data: {basic_subgraph_file}")
            else:
                logger.warning("⚠️  No subgraph_stats available in task results")
            
            # Also try to extract subgraphs from tasks if they have the data
            if tasks:
                for i, task in enumerate(tasks):
                    if isinstance(task, dict) and 'subgraph_nodes' in task and 'subgraph_edges' in task:
                        subgraph_data = {
                            "task_id": task.get("task_id", f"task_{i}"),
                            "web_task_type": task.get("web_task_type", "unknown"),
                            "subgraph_nodes": task.get("subgraph_nodes", []),
                            "subgraph_edges": task.get("subgraph_edges", []),
                            "subgraph_stats": task.get("subgraph_stats", {}),
                            "generated_at": datetime.now().isoformat()
                        }
                        
                        # Save individual subgraph
                        subgraph_file = subgraphs_dir / f"subgraph_{task.get('task_id', f'task_{i}')}.json"
                        with open(subgraph_file, 'w', encoding='utf-8') as f:
                            json.dump(subgraph_data, f, indent=2, ensure_ascii=False)
                        subgraph_files.append(str(subgraph_file))
                        
                        logger.debug(f"💾 Saved subgraph for task {task.get('task_id', f'task_{i}')}: {subgraph_file}")
            
            # Create a placeholder subgraph file only if absolutely no subgraph data was available
            if not subgraph_files:
                placeholder_data = {
                    "note": "No subgraphs were generated or saved",
                    "reason": "No subgraph data available in any form (stats, detailed, or task-based)",
                    "generated_at": datetime.now().isoformat()
                }
                
                placeholder_file = subgraphs_dir / "subgraphs_placeholder.json"
                with open(placeholder_file, 'w', encoding='utf-8') as f:
                    json.dump(placeholder_data, f, indent=2, ensure_ascii=False)
                
                subgraph_files.append(str(placeholder_file))
                logger.warning("⚠️  Created placeholder subgraph file - no subgraph data available at all")
            else:
                logger.info(f"✅ Successfully saved {len(subgraph_files)} subgraph files")
            
            logger.info(f"💾 Subgraphs directory created at {subgraphs_dir}")
            logger.info(f"💾 Total subgraph files: {len(subgraph_files)}")
            
            # Save web tasks
            task_results = results.get("stages", {}).get("web_task_generation", {})
            tasks = task_results.get("tasks", [])
            all_web_tasks_file = None  # Initialize variable outside the if block
            
            if tasks:
                # Save web task quality assessment report
                quality_report_file, detailed_report_file = self._save_web_task_quality_report(tasks, results_subdir)
                
                # Save all web tasks
                all_web_tasks_file = output_path / "all_web_tasks.jsonl"
                with open(all_web_tasks_file, 'w', encoding='utf-8') as f:
                    for task in tasks:
                        serializable_task = self._make_serializable(task)
                        f.write(json.dumps(serializable_task, ensure_ascii=False) + '\n')
                
                # Split and save normal and safety web tasks separately
                normal_web_tasks = []
                safety_web_tasks = []
                
                for task in tasks:
                    # Ensure task is a dictionary
                    if isinstance(task, str):
                        logger.warning(f"Task is a string, skipping: {task}")
                        continue
                    
                    # Convert task to dict if it has to_dict method
                    if hasattr(task, 'to_dict'):
                        task_dict = task.to_dict()
                    elif isinstance(task, dict):
                        task_dict = task
                    else:
                        logger.warning(f"Task is neither dict nor has to_dict method: {type(task)}")
                        continue
                    
                    # Check if it's a safety task by looking at task_id or web_task_type
                    if (task_dict.get("task_id", "").startswith("web_safety_") or 
                        "safety" in task_dict.get("web_task_type", "").lower()):
                        safety_web_tasks.append(task_dict)
                    else:
                        normal_web_tasks.append(task_dict)
                
                # Save normal web tasks
                if normal_web_tasks:
                    normal_web_tasks_file = output_path / "normal_web_tasks.jsonl"
                    with open(normal_web_tasks_file, 'w', encoding='utf-8') as f:
                        for task in normal_web_tasks:
                            serializable_task = self._make_serializable(task)
                            f.write(json.dumps(serializable_task, ensure_ascii=False) + '\n')
                    logger.info(f"✅ Created normal_web_tasks with {len(normal_web_tasks)} tasks: {normal_web_tasks_file}")
                
                # Save safety web tasks
                if safety_web_tasks:
                    safety_web_tasks_file = output_path / "safety_web_tasks.jsonl"
                    with open(safety_web_tasks_file, 'w', encoding='utf-8') as f:
                        for task in safety_web_tasks:
                            serializable_task = self._make_serializable(task)
                            f.write(json.dumps(serializable_task, ensure_ascii=False) + '\n')
                    logger.info(f"✅ Created safety_web_tasks with {len(safety_web_tasks)} tasks: {safety_web_tasks_file}")
                
                logger.info(f"📦 Split web tasks: {len(normal_web_tasks)} normal tasks, {len(safety_web_tasks)} safety tasks")
            else:
                quality_report_file = None
                detailed_report_file = None
            
            # Add saved file paths to results for summary display
            saved_files = {
                "web_dataset_results": str(results_file),
            }
            
            # Add subgraph files if they exist
            if 'subgraph_files' in locals() and subgraph_files:
                saved_files["subgraphs"] = subgraph_files
                saved_files["subgraphs_summary"] = str(subgraphs_dir / "subgraphs_summary.json") if 'subgraphs_dir' in locals() else None
            
            # Add web graph file only if it exists
            if 'web_graph_file' in locals() and web_graph_file:
                saved_files["web_graph"] = str(web_graph_file)
            
            # Add task-related files only if they exist
            if all_web_tasks_file:
                saved_files["all_web_tasks"] = str(all_web_tasks_file)
            if quality_report_file:
                saved_files["web_task_quality_report"] = str(quality_report_file)
            if detailed_report_file:
                saved_files["web_task_detailed_quality"] = str(detailed_report_file)
            
            # Add split dataset files
            if 'normal_web_tasks_file' in locals() and normal_web_tasks_file:
                saved_files["normal_web_tasks"] = str(normal_web_tasks_file)
            if 'safety_web_tasks_file' in locals() and safety_web_tasks_file:
                saved_files["safety_web_tasks"] = str(safety_web_tasks_file)
            
            # Update the results with saved file paths
            if "stages" in results and "web_task_generation" in results["stages"]:
                results["stages"]["web_task_generation"]["saved_files"] = saved_files
            
            logger.info(f"✅ Web dataset generation results saved to {output_dir}")
            logger.info(f"✅ Saved files: {list(saved_files.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save web dataset generation results: {e}")
            raise
    
    def _get_node_type_distribution(self, nodes: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of node types in a subgraph"""
        type_distribution = {}
        for node_id, node_data in nodes.items():
            if isinstance(node_data, dict):
                node_type = node_data.get("node_type", "unknown")
                type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
            elif hasattr(node_data, 'node_type'):
                node_type = getattr(node_data, 'node_type', 'unknown')
                type_distribution[str(node_type)] = type_distribution.get(str(node_type), 0) + 1
        return type_distribution
    
    def _get_edge_type_distribution(self, edges: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of edge types in a subgraph"""
        type_distribution = {}
        for edge_id, edge_data in edges.items():
            if isinstance(edge_data, dict):
                edge_type = edge_data.get("edge_type", "unknown")
                type_distribution[edge_type] = type_distribution.get(edge_type, 0) + 1
            elif hasattr(edge_data, 'edge_type'):
                edge_type = getattr(edge_data, 'edge_type', 'unknown')
                type_distribution[str(edge_type)] = type_distribution.get(str(edge_type), 0) + 1
        return type_distribution
    

    
    def _save_web_task_quality_report(self, tasks: List[Dict[str, Any]], output_path: Path):
        """Save web task quality assessment report"""
        try:
            if not tasks:
                logger.warning("No web tasks to generate quality report for")
                return None, None
            
            # Calculate quality metrics
            total_tasks = len(tasks)
            task_types = {}
            difficulties = {}
            quality_scores = []
            passed_quality_check = 0
            failed_quality_check = 0
            
            # Detailed quality metrics
            quality_details = {
                'completeness': [],
                'realism': [],
                'complexity': [],
                'specificity': [],
                'feasibility': []
            }
            
            for task in tasks:
                # Ensure task is a dictionary
                if isinstance(task, str):
                    logger.warning(f"Task is a string in quality report, skipping: {task}")
                    continue
                
                # Convert task to dict if it has to_dict method
                if hasattr(task, 'to_dict'):
                    task_dict = task.to_dict()
                elif isinstance(task, dict):
                    task_dict = task
                else:
                    logger.warning(f"Task is neither dict nor has to_dict method in quality report: {type(task)}")
                    continue
                
                # Task type distribution
                task_type = task_dict.get('web_task_type', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1
                
                # Difficulty distribution
                difficulty = task_dict.get('difficulty', 'unknown')
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                # Quality score statistics
                quality_score = task_dict.get('quality_score')
                if quality_score is not None:
                    quality_scores.append(quality_score)
                
                # Quality check pass/fail statistics
                if task.get('passed_quality_check', False):
                    passed_quality_check += 1
                else:
                    failed_quality_check += 1
                
                # Detailed quality metrics
                task_quality_details = task.get('quality_details', {})
                for key in quality_details:
                    if key in task_quality_details:
                        quality_details[key].append(task_quality_details[key])
            
            # Calculate averages
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            min_quality_score = min(quality_scores) if quality_scores else 0.0
            max_quality_score = max(quality_scores) if quality_scores else 0.0
            
            # Detailed quality averages
            avg_quality_details = {}
            for key, scores in quality_details.items():
                avg_quality_details[key] = sum(scores) / len(scores) if scores else 0.0
            
            # Create comprehensive quality report
            quality_report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_tasks": total_tasks,
                    "report_type": "web_task_quality_assessment"
                },
                "quality_statistics": {
                    "total_tasks": total_tasks,
                    "tasks_with_quality_scores": len(quality_scores),
                    "quality_scores": quality_scores,
                    "average_quality_score": avg_quality_score,
                    "min_quality_score": min_quality_score,
                    "max_quality_score": max_quality_score,
                    "passed_quality_check": passed_quality_check,
                    "failed_quality_check": failed_quality_check,
                    "pass_rate": passed_quality_check / total_tasks if total_tasks > 0 else 0.0
                },
                "task_type_distribution": task_types,
                "difficulty_distribution": difficulties,
                "detailed_quality_metrics": avg_quality_details,
                "quality_thresholds": {
                    "minimum_quality_score": 0.6,
                    "quality_check_threshold": 0.6
                },
                "quality_criteria": {
                    "completeness": "Task has all essential components (prompt, steps, type, difficulty)",
                    "realism": "Task describes realistic user behavior with appropriate keywords",
                    "complexity": "Task has appropriate complexity based on steps, types, and difficulty",
                    "specificity": "Task has specific actions and clear targets",
                    "feasibility": "Task is feasible to execute with reasonable steps and duration"
                }
            }
            
            # Save quality report
            quality_report_file = output_path / "web_task_quality_report.json"
            with open(quality_report_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Web task quality report saved to {quality_report_file}")
            
            # Also save detailed task quality information
            detailed_tasks = []
            for task in tasks:
                detailed_task = {
                    "task_id": task.get('task_id'),
                    "web_task_type": task.get('web_task_type'),
                    "difficulty": task.get('difficulty'),
                    "prompt_length": len(task.get('prompt', '')),
                    "task_steps_count": len(task.get('task_steps', [])),
                    "quality_score": task.get('quality_score'),
                    "quality_details": task.get('quality_details', {}),
                    "quality_reasoning": task.get('quality_reasoning'),
                    "passed_quality_check": task.get('passed_quality_check', False),
                    "expected_duration": task.get('expected_duration', 0),
                    "hop_count": task.get('hop_count', 1),
                    "interaction_count": task.get('interaction_count', 0)
                }
                detailed_tasks.append(detailed_task)
            
            detailed_report_file = output_path / "web_task_detailed_quality.json"
            with open(detailed_report_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_tasks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Detailed web task quality information saved to {detailed_report_file}")
            
            return quality_report_file, detailed_report_file
            
        except Exception as e:
            logger.error(f"❌ Failed to save web task quality report: {e}")
            raise
    
    def _evaluate_single_file(self, file_path: str, graph_path: Optional[str],
                             dataset_type: str, output_dir: Optional[str],
                             agent_mode: str) -> Dict[str, Any]:
        """Evaluate a single dataset file"""
        logger.info(f"📄 Evaluating single file: {file_path}")

        # Determine if this is a web task file or text task file
        file_name = Path(file_path).name
        is_web_file = 'web' in file_name.lower()

        if is_web_file or agent_mode == 'web':
            # Web agent evaluation - no graph needed
            logger.info("🌐 Web agent evaluation mode")
            import asyncio
            return asyncio.run(self.evaluate_agent_on_dataset(file_path, None, None, output_dir))
        else:
            # Text agent evaluation - need graph
            logger.info("📝 Text agent evaluation mode")

            # Get graph path
            if not graph_path:
                graph_path = self._auto_detect_latest_graph()
                if not graph_path:
                    raise ValueError("No graph directory found. Please specify with -g")

            graph_file = Path(graph_path) / "graph" / "knowledge_graph.json"
            vectors_file = Path(graph_path) / "vectors" / "vectors_faiss.faiss"

            if not graph_file.exists():
                raise FileNotFoundError(f"Knowledge graph not found at {graph_file}")

            logger.info(f"🕸️ Using graph: {graph_file}")
            logger.info(f"🔍 Using vectors: {vectors_file}")

            import asyncio
            return asyncio.run(self.evaluate_agent_on_dataset(file_path, str(graph_file), str(vectors_file), output_dir))

    def _evaluate_directory(self, dir_path: str, graph_path: Optional[str],
                           dataset_type: str, output_dir: Optional[str],
                           agent_mode: str) -> Dict[str, Any]:
        """Evaluate a dataset directory with auto-detection"""
        logger.info(f"📂 Evaluating directory: {dir_path}")

        # Auto-detect graph path for non-web agents
        if agent_mode != 'web':
            if not graph_path:
                graph_path = self._auto_detect_latest_graph()
                if not graph_path:
                    raise ValueError("No graph directory found. Please specify with -g")

        # Detect dataset paths
        if agent_mode != 'web' and graph_path:
            detected_paths = self._detect_dataset_paths(dir_path, graph_path)
        else:
            detected_paths = self._detect_dataset_paths(dir_path)

        # Select appropriate dataset file based on type and agent mode
        if agent_mode == 'web':
            dataset_file = self._select_web_dataset_file(detected_paths, dataset_type)
        else:
            dataset_file = self._select_text_dataset_file(detected_paths, dataset_type)

        if not dataset_file or not Path(dataset_file).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        logger.info(f"📊 Selected dataset file: {dataset_file}")

        # Execute evaluation
        if agent_mode == 'web':
            import asyncio
            return asyncio.run(self.evaluate_agent_on_dataset(dataset_file, None, None, output_dir))
        else:
            graph_file = detected_paths["graph"]
            vectors_file = detected_paths["vectors"]

            logger.info(f"🕸️ Using graph: {graph_file}")
            logger.info(f"🔍 Using vectors: {vectors_file}")

            import asyncio
            return asyncio.run(self.evaluate_agent_on_dataset(dataset_file, graph_file, vectors_file, output_dir))

    def _select_web_dataset_file(self, detected_paths: Dict[str, Any], dataset_type: str) -> Optional[str]:
        """Select appropriate web dataset file based on type"""
        datasets = detected_paths.get("datasets", {})

        if dataset_type == "normal":
            return datasets.get("normal_web_tasks") or datasets.get("all_web_tasks")
        elif dataset_type == "safety":
            return datasets.get("safety_web_tasks") or datasets.get("all_web_tasks")
        else:  # "all"
            return datasets.get("all_web_tasks")

    def _select_text_dataset_file(self, detected_paths: Dict[str, Any], dataset_type: str) -> Optional[str]:
        """Select appropriate text dataset file based on type"""
        datasets = detected_paths.get("datasets", {})

        if dataset_type == "normal":
            return datasets.get("all_tasks")  # Normal tasks are in all_tasks
        elif dataset_type == "safety":
            return datasets.get("safety_tasks")
        else:  # "all"
            return datasets.get("all_tasks")

    def _is_web_agent_evaluation(self, tasks: List[Any]) -> bool:
        """Check if tasks are web tasks that require Web Agent evaluation"""
        if not tasks:
            return False
        
        # Check if any task has web-specific fields
        for task in tasks[:5]:  # Check first 5 tasks
            if isinstance(task, dict):
                if 'web_task_type' in task or 'task_steps' in task or 'start_page_url' in task:
                    return True
            elif hasattr(task, 'web_task_type') or hasattr(task, 'task_steps') or hasattr(task, 'start_page_url'):
                return True
        
        return False
    
    async def _run_web_agent_execution_stage(self, tasks: List[Any]) -> Dict[str, Any]:
        """Run Web Agent execution stage with browser automation and malicious element injection for safety tasks"""
        logger.info(f"{'='*60}")
        logger.info("🌐 WEB AGENT EXECUTION STAGE")
        logger.info(f"{'='*60}")
        logger.info(f"🌐 Executing {len(tasks)} web tasks with browser automation")
        
        start_time = time.time()
        
        try:
            # Import Web Agent
            from agent_framework.som_agent import WebAgent
            
            # Use existing output directories
            output_dirs = self.output_dirs
            web_agent_output_dir = output_dirs["results"] / "web_agent"
            
            # Ensure web_agent directory exists
            web_agent_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Web Agent with correct output directory
            web_agent = WebAgent(
                config=self.config.agent.get('web_agent', {}),
                output_dir=str(web_agent_output_dir)
            )
            
            # Initialize browser
            await web_agent.initialize_browser()
            
            # Ensure all tasks are WebTaskInstance objects
            from task_craft.task_generator import WebTaskInstance
            web_tasks = []
            for task in tasks:
                if isinstance(task, WebTaskInstance):
                    web_tasks.append(task)
                else:
                    logger.warning(f"Task {getattr(task, 'task_id', 'unknown')} is not a WebTaskInstance, skipping")
                    continue
            
            if not web_tasks:
                logger.error("No valid WebTaskInstance objects found for execution")
                return {
                    "error": "No valid web tasks found",
                    "processing_time": time.time() - start_time
                }
            
            # Initialize malicious element injector for safety tasks - COMMENTED OUT
            malicious_injector = None
            # try:
            #     from suitecrm_setup.malicious_injector import MaliciousInjector
            #     malicious_injector = MaliciousInjector()
            #     logger.info("🔒 Malicious element injector initialized")
            # except Exception as e:
            #     logger.warning(f"⚠️ Failed to initialize malicious injector: {e}")
            
            logger.info("⚠️ Malicious element injector disabled")
            
            # Separate safety tasks from normal tasks
            safety_tasks = []
            normal_tasks = []
            
            for task in web_tasks:
                is_safety_task = (
                    hasattr(task, 'web_task_type') and 
                    (task.web_task_type == "safety" or task.web_task_type.startswith('web_safety_'))
                )
                if is_safety_task:
                    safety_tasks.append(task)
                else:
                    normal_tasks.append(task)
            
            logger.info(f"📊 Task distribution: {len(normal_tasks)} normal tasks, {len(safety_tasks)} safety tasks")
            
            # Check for existing task results to skip already completed tasks
            completed_task_ids = self._get_completed_task_ids(output_dirs["results"])
            logger.info(f"🔍 Found {len(completed_task_ids)} already completed web tasks in total")
            logger.info(f"📊 Current dataset has {len(tasks)} tasks")
            
            # Execute normal tasks first (without malicious elements)
            execution_results = []
            execution_trajectories = []
            
            # Filter out already completed normal tasks
            normal_tasks_to_execute = [task for task in normal_tasks if task.task_id not in completed_task_ids]
            
            # Log detailed statistics
            skipped_normal_tasks = len(normal_tasks) - len(normal_tasks_to_execute)
            logger.info(f"📊 Normal tasks: {len(normal_tasks)} total, {skipped_normal_tasks} already completed, {len(normal_tasks_to_execute)} to execute")
            
            successful_normal_executions = 0
            failed_normal_executions = 0
            
            # Only create progress bar and execute if there are normal tasks
            if normal_tasks_to_execute:
                logger.info(f"🌐 Found {len(normal_tasks_to_execute)} normal tasks to execute")
                
                # Create progress bar for normal web task execution
                # Create progress bar with better visibility settings
                normal_pbar = tqdm(
                    total=len(normal_tasks_to_execute),
                    desc="🌐 Executing Normal Web Tasks",
                    unit="task",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    leave=True,
                    file=sys.stderr,  # Use stderr to avoid log interference
                    dynamic_ncols=True,  # Adapt to terminal width
                    miniters=1,  # Update progress bar on every iteration
                    smoothing=0.1,  # Smooth progress updates
                    position=0,  # Position at top
                    ncols=120  # Fixed width for better visibility
                )
                
                for i, task in enumerate(normal_tasks_to_execute):
                    try:
                        # Reduce log verbosity to avoid interfering with progress bar
                        logger.debug(f"🚀 Executing normal task {i+1}/{len(normal_tasks_to_execute)}: {task.task_id}")
                        
                        # Execute task with browser automation
                        execution_result, trajectory = await web_agent.execute_web_task(task)
                        
                        # Save individual task result after completion
                        self._save_individual_task_result(execution_result, task, output_dirs["results"])
                        
                        execution_results.append(execution_result)
                        execution_trajectories.append(trajectory)
                        
                        # Reduce log verbosity to avoid interfering with progress bar
                        logger.debug(f"✅ Normal task {task.task_id} completed: {execution_result.success}")
                        
                        if execution_result.success:
                            successful_normal_executions += 1
                        else:
                            failed_normal_executions += 1
                        
                        # Update progress bar
                        normal_pbar.set_postfix({"✅": f"{successful_normal_executions}", "❌": f"{failed_normal_executions}"})
                        normal_pbar.update(1)
                        normal_pbar.refresh()  # Force refresh to ensure visibility
                        
                        # Close browser window after each task to avoid contamination
                        logger.info(f"🔄 Closing browser window after task {task.task_id} to prevent contamination")
                        await web_agent.close_browser()
                        
                        # Re-initialize browser for next task (if not the last task)
                        if i < len(normal_tasks_to_execute) - 1:
                            logger.info(f"🔧 Re-initializing browser for next task")
                            await web_agent.initialize_browser()
                    
                    except Exception as e:
                        logger.error(f"Failed to execute normal task {task.task_id}: {e}")
                        
                        # Close browser even on failure to ensure clean state
                        try:
                            await web_agent.close_browser()
                            # Re-initialize browser for next task (if not the last task)
                            if i < len(normal_tasks_to_execute) - 1:
                                logger.info(f"🔧 Re-initializing browser for next task after failure")
                                await web_agent.initialize_browser()
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup browser after task failure: {cleanup_error}")
                        
                        # Create failed result
                        from agent_framework.evaluators import WebTaskExecutionResult
                        failed_result = WebTaskExecutionResult(
                            task_id=task.task_id,
                            success=False,
                            execution_time=0.0,
                            steps_completed=0,
                            total_steps=len(task.task_steps) if hasattr(task, 'task_steps') else 0,
                            error_type="execution_error",
                            error_message=str(e)
                        )
                        
                        # Save individual failed task result
                        self._save_individual_task_result(failed_result, task, output_dirs["results"])
                        
                        execution_results.append(failed_result)
                        failed_normal_executions += 1
                        
                        # Update progress bar
                        normal_pbar.set_postfix({"✅": f"{successful_normal_executions}", "❌": f"{failed_normal_executions}"})
                        normal_pbar.update(1)
                        normal_pbar.refresh()  # Force refresh to ensure visibility
                
                # Close normal web task progress bar
                normal_pbar.close()
            else:
                logger.info("🌐 No normal tasks to execute")
            
            # Inject malicious elements before executing all safety tasks - COMMENTED OUT
            # if safety_tasks and malicious_injector:
            #     logger.info(f"🔒 Injecting malicious elements before executing {len(safety_tasks)} safety tasks...")
            #     try:
            #         success = malicious_injector.inject_malicious_elements([
            #             "phishing_form", "fake_button", "sensitive_data_form", 
            #             "fake_link", "fake_notification"
            #         ])
            #         if success:
            #             logger.info("✅ Malicious elements injected successfully for safety tasks")
            #         else:
            #             logger.warning("⚠️ Failed to inject malicious elements for safety tasks")
            #     except Exception as e:
            #         logger.error(f"❌ Error injecting malicious elements for safety tasks: {e}")
            
            # Execute safety tasks (with malicious elements already injected) - COMMENTED OUT
            # Filter out already completed safety tasks
            # safety_tasks_to_execute = [task for task in safety_tasks if task.task_id not in completed_task_ids]
            
            # Log detailed statistics for safety tasks
            # skipped_safety_tasks = len(safety_tasks) - len(safety_tasks_to_execute)
            # logger.info(f"📊 Safety tasks: {len(safety_tasks)} total, {skipped_safety_tasks} already completed, {len(safety_tasks_to_execute)} to execute")
            
            successful_safety_executions = 0
            failed_safety_executions = 0
            
            # Only create progress bar and execute if there are safety tasks - COMMENTED OUT
            # if safety_tasks_to_execute:
            #     logger.info(f"🔒 Found {len(safety_tasks_to_execute)} safety tasks to execute")
            #     
            #     # Create progress bar for safety web task execution
            #     safety_pbar = tqdm(
            #         total=len(safety_tasks_to_execute),
            #         desc="🔒 Executing Safety Web Tasks",
            #         unit="task",
            #         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            #         leave=True,
            #         file=sys.stderr,  # Use stderr to avoid log interference
            #         dynamic_ncols=True,  # Adapt to terminal width
            #         miniters=1,  # Update progress bar on every iteration
            #         smoothing=0.1,  # Smooth progress updates
            #         position=1,  # Position below normal tasks progress bar
            #         ncols=120  # Fixed width for better visibility
            #     )
            #     
            #     for i, task in enumerate(safety_tasks_to_execute):
            #         try:
            #             # Reduce log verbosity to avoid interfering with progress bar
            #             logger.debug(f"🚀 Executing safety task {i+1}/{len(safety_tasks_to_execute)}: {task.task_id}")
            #             
            #             # Execute task with browser automation
            #             execution_result, trajectory = await web_agent.execute_web_task(task)
            #             
            #             # Save individual task result after completion
            #             self._save_individual_task_result(execution_result, task, output_dirs["results"])
            #             
            #             execution_results.append(execution_result)
            #             execution_trajectories.append(trajectory)
            #             
            #             # Reduce log verbosity to avoid interfering with progress bar
            #             logger.debug(f"✅ Safety task {task.task_id} completed: {execution_result.success}")
            #             
            #             if execution_result.success:
            #                 successful_safety_executions += 1
            #             else:
            #                 failed_safety_executions += 1
            #             
            #             # Update progress bar
            #             safety_pbar.set_postfix({"✅": f"{successful_safety_executions}", "❌": f"{failed_safety_executions}"})
            #             safety_pbar.update(1)
            #             safety_pbar.refresh()  # Force refresh to ensure visibility
            #             
            #             # Close browser window after each task to avoid contamination
            #             logger.info(f"🔄 Closing browser window after task {task.task_id} to prevent contamination")
            #             await web_agent.close_browser()
            #             
            #             # Re-initialize browser for next task (if not the last task)
            #             if i < len(safety_tasks_to_execute) - 1:
            #                 logger.info(f"🔧 Re-initializing browser for next task")
            #                 await web_agent.initialize_browser()
            #         
            #         except Exception as e:
            #             logger.error(f"Failed to execute safety task {task.task_id}: {e}")
            #             
            #             # Close browser even on failure to ensure clean state
            #             try:
            #                 await web_agent.close_browser()
            #                 # Re-initialize browser for next task (if not the last task)
            #                 if i < len(safety_tasks_to_execute) - 1:
            #                     logger.info(f"🔧 Re-initializing browser for next task after failure")
            #                     await web_agent.initialize_browser()
            #             except Exception as cleanup_error:
            #                 logger.warning(f"Failed to cleanup browser after task failure: {cleanup_error}")
            #             
            #             # Create failed result
            #             from agent_framework.evaluators import WebTaskExecutionResult
            #             failed_result = WebTaskExecutionResult(
            #                 task_id=task.task_id,
            #                 success=False,
            #                 execution_time=0.0,
            #                 steps_completed=0,
            #                 total_steps=len(task.task_steps) if hasattr(task, 'task_steps') else 0,
            #                 error_type="execution_error",
            #                 error_message=str(e)
            #             )
            #             
            #             # Save individual failed task result
            #             self._save_individual_task_result(failed_result, task, output_dirs["results"])
            #             
            #             execution_results.append(failed_result)
            #             failed_safety_executions += 1
            #             
            #             # Update progress bar
            #             safety_pbar.set_postfix({"✅": f"{successful_safety_executions}", "❌": f"{failed_safety_executions}"})
            #             safety_pbar.update(1)
            #             safety_pbar.refresh()  # Force refresh to ensure visibility
            #     
            #     # Close safety web task progress bar
            #     safety_pbar.close()
            # else:
            #     logger.info("🔒 No safety tasks to execute")
            
            logger.info("⚠️ Safety task execution disabled")
            
            # Remove malicious elements after all safety tasks are completed - COMMENTED OUT
            # if safety_tasks and malicious_injector:
            #     logger.info(f"🧹 All {len(safety_tasks)} safety tasks completed, removing malicious elements...")
            #     try:
            #         success = malicious_injector.remove_malicious_elements()
            #         if success:
            #             logger.info("✅ Malicious elements removed successfully after all safety tasks")
            #         else:
            #             logger.warning("⚠️ Failed to remove malicious elements after safety tasks")
            #     except Exception as e:
            #         logger.error(f"❌ Error removing malicious elements after safety tasks: {e}")
            
            # Final cleanup - ensure browser is closed
            logger.info("🔐 Final browser cleanup")
            try:
                await web_agent.close_browser()
            except Exception as e:
                logger.warning(f"Failed to close browser in final cleanup: {e}")
            
            # Final malicious element cleanup (in case any safety tasks failed and elements weren't removed)
            if malicious_injector:
                logger.info("🧹 Final malicious element cleanup (safety check)")
                try:
                    malicious_injector.remove_malicious_elements()
                    logger.info("✅ Final malicious element cleanup completed")
                except Exception as e:
                    logger.warning(f"Failed to perform final malicious element cleanup: {e}")
            
            processing_time = time.time() - start_time
            
            # Convert execution results to proper format with all required fields
            formatted_results = []
            for result in execution_results:
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict()
                else:
                    result_dict = result
                
                # Ensure all required fields are present
                formatted_result = {
                    "task_id": result_dict.get("task_id", ""),
                    "task_type": "web_task",  # Add task_type for web tasks
                    "web_task_type": getattr(result, 'web_task_type', 'Search'),  # Add web_task_type
                    "success": result_dict.get("success", False),
                    "answer": result_dict.get("answer", ""),
                    "execution_time": result_dict.get("execution_time", 0.0),
                    "tokens_used": result_dict.get("tokens_used", 0),
                    "model_used": result_dict.get("model_used", "browser_automation"),
                    "citations": result_dict.get("citations", []),
                    "reasoning_path": result_dict.get("reasoning_path", []),
                    "confidence": result_dict.get("confidence", 0.0),
                    "error_type": result_dict.get("error_type"),
                    "error_message": result_dict.get("error_message", ""),
                    "retries_needed": result_dict.get("retries_needed", 0),
                    "raw_response": result_dict.get("raw_response")
                }
                formatted_results.append(formatted_result)
            
            # Calculate web task type distribution
            web_task_types = {}
            for result in execution_results:
                task_type = getattr(result, 'web_task_type', 'unknown')
                web_task_types[task_type] = web_task_types.get(task_type, 0) + 1
            
            # Calculate success rate by task type
            success_by_type = {}
            for result in execution_results:
                task_type = getattr(result, 'web_task_type', 'unknown')
                if task_type not in success_by_type:
                    success_by_type[task_type] = {'total': 0, 'successful': 0}
                success_by_type[task_type]['total'] += 1
                if getattr(result, 'success', False):
                    success_by_type[task_type]['successful'] += 1
            
            # Calculate token usage statistics
            total_tokens = sum(result_dict.get("tokens_used", 0) for result_dict in formatted_results)
            token_counts = [result_dict.get("tokens_used", 0) for result_dict in formatted_results]
            avg_tokens = total_tokens / len(formatted_results) if formatted_results else 0
            min_tokens = min(token_counts) if token_counts else 0
            max_tokens = max(token_counts) if token_counts else 0
            
            return {
                "results": formatted_results,
                "trajectories": [traj.to_dict() for traj in execution_trajectories],
                "total_tasks": len(execution_results),
                "successful_tasks": sum(1 for r in execution_results if getattr(r, 'success', False)),
                "processing_time": processing_time,
                "web_task_types": web_task_types,
                "success_by_type": success_by_type,
                "total_tokens_used": total_tokens,
                "avg_tokens_per_task": avg_tokens,
                "min_tokens_per_task": min_tokens,
                "max_tokens_per_task": max_tokens
            }
            
        except Exception as e:
            logger.error(f"Web Agent execution stage failed: {e}")
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _save_dataset_generation_results(self, results: Dict[str, Any], output_dir: str):
        """Save dataset generation results to specified directory"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_path / "dataset_generation_results.json"
            
            # Custom JSON encoder to handle numpy types and Path objects
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, Path):
                        return str(obj)
                    return super().default(obj)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(results), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            # Save graph files to unified output directories
            graph_results = results.get("stages", {}).get("graph_construction", {})
            graphs = graph_results.get("graphs", [])
            if graphs:
                # Use existing output directories
                output_dirs = self.output_dirs
                graph_dir = output_dirs["graph"]
                vectors_dir = output_dirs["vectors"]
                
                # Save the single graph
                graph = graphs[0]
                graph.save(graph_dir, vectors_dir)
            
            # Save task files to unified datasets directory
            task_results = results.get("stages", {}).get("task_generation", {})
            tasks = task_results.get("tasks", [])
            if tasks:
                # Use existing output directories
                output_dirs = self.output_dirs
                run_result_dir = output_dirs["run_result"]
                
                # Save all tasks
                if tasks:
                    all_tasks_file = run_result_dir / "all_tasks.jsonl"
                    with open(all_tasks_file, 'w', encoding='utf-8') as f:
                        for task in tasks:
                            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Dataset generation results saved to {output_path}")
            
            # Log unified output directories
            output_dirs = self.output_dirs
            logger.info(f"📊 Datasets saved to {output_dirs['run_result']}/")
            logger.info(f"🕸️ Graph saved to {output_dirs['graph']}/")
            logger.info(f"🔍 Vectors saved to {output_dirs['vectors']}/")
            
        except Exception as e:
            logger.warning(f"Failed to save dataset generation results: {e}")
    
    async def evaluate_agent_on_dataset(self, dataset_path: str, graph_path: Optional[str] = None, 
                           vectors_path: Optional[str] = None, 
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate agent performance on existing dataset"""
        logger.info("🚀 Starting agent evaluation on dataset")
        logger.info(f"📊 Dataset: {dataset_path}")
        if graph_path:
            logger.info(f"🕸️ Graph: {graph_path}")
        if vectors_path:
            logger.info(f"🔍 Vectors: {vectors_path}")
        
        self.start_time = time.time()
        # Determine agent type
        agent_mode = self.config.agent.get('agent_mode', 'single')
        multi_agent_enabled = (agent_mode == 'multi')
        agent_type = "multi_agent" if multi_agent_enabled else self.config.agent.get('agent_type')
            
        results = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                "agent_mode": agent_mode,
                "agent_type": agent_type,
                "dataset_path": dataset_path,
                "graph_path": graph_path,
                "vectors_path": vectors_path
            },
            "stages": {}
                }
        
        # Store results in instance for later saving
        self.results = results
        
        try:
                # Stage 1: Load dataset
                logger.info("📊 Loading dataset")
                stage_start = time.time()
                tasks = self._load_dataset(dataset_path)
                logger.info(f"📊 Loaded {len(tasks)} tasks")
                if tasks:
                    logger.info(f"📊 First task type: {type(tasks[0])}")
                    if hasattr(tasks[0], 'web_task_type'):
                        logger.info(f"📊 First task web_task_type: {tasks[0].web_task_type}")
                results["stages"]["dataset_loading"] = {
                    "tasks_loaded": len(tasks),
                    "dataset_path": dataset_path,
                    "processing_time": time.time() - stage_start
                }
                
                # Stage 2: Load graph (skip for Web Agent)
                graph = None
                stage_start = time.time()
                
                # For text tasks, load graph from provided paths or detect from dataset
                if agent_mode != 'web' and 'agent' in self.components:
                    # Priority 1: Use provided graph_path and vectors_path
                    if graph_path and vectors_path:
                        logger.info(f"🕸️ Loading graph from provided paths: {graph_path}")
                        try:
                            graph = self._load_graph(graph_path, vectors_path, dataset_path)
                            logger.info("✅ Graph loaded successfully from provided paths")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load graph from provided paths: {e}")
                            graph = None

                    # Priority 2: Try to detect graph from dataset source_document
                    if graph is None:                        
                        logger.info("🔍 Detecting graph from dataset source_document")
                        detected_graph_path, detected_vectors_path = self._detect_graph_from_dataset(tasks, dataset_path)

                        if detected_graph_path and detected_vectors_path:
                            logger.info(f"🕸️ Loading graph from detected paths: {detected_graph_path}")
                            try:
                                graph = self._load_graph(detected_graph_path, detected_vectors_path, dataset_path)
                                # Update the graph_path and vectors_path for results
                                graph_path = detected_graph_path
                                vectors_path = detected_vectors_path
                                logger.info("✅ Graph loaded successfully from detected paths")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load graph from detected paths: {e}")
                                graph = None
                        else:
                            logger.warning("⚠️ Could not detect graph from dataset source_document")
                    
                    # Set graph to components if loaded successfully
                    if graph:
                        # Set graph to storage component for multi-agent system
                        if 'storage' in self.components:
                            self.components['storage'].graph = graph
                            logger.info("✅ Graph set to storage component")
                    
                        if hasattr(self.components['agent'], 'set_graph'):
                            self.components['agent'].set_graph(graph)
                            logger.info("✅ Graph set to agent")
                    
                        results["stages"]["graph_loading"] = {
                            "graph_loaded": True,
                            "graph_path": graph_path,
                            "vectors_path": vectors_path,
                            "graph_stats": graph.get_stats() if graph else {},
                            "processing_time": time.time() - stage_start
                        }
                    else:
                        results["stages"]["graph_loading"] = {
                            "graph_loaded": False,
                            "reason": "Could not load or detect graph",
                            "processing_time": time.time() - stage_start
                        }
                elif agent_mode == 'web':
                    logger.info("🌐 Web Agent mode - skipping graph loading")
                    results["stages"]["graph_loading"] = {
                        "graph_loaded": False,
                        "reason": "Web Agent mode - no graph required",
                        "processing_time": time.time() - stage_start
                    }
                
                # Stage 3: Task Execution
                logger.info("🏃‍♂️ Stage: Task Execution")
                
                # Check if this is a web agent evaluation
                agent_mode = self.config.agent.get('agent_mode', 'single')
                agent_type = self.config.agent.get('agent_type', 'rag')
                is_web_evaluation = self._is_web_agent_evaluation(tasks)
                logger.info(f"🔍 Debug: agent_mode={agent_mode}, agent_type={agent_type}, is_web_evaluation={is_web_evaluation}")
                
                if agent_mode == 'web' or is_web_evaluation:
                    logger.info("🌐 Using Web Agent for execution")
                    execution_results = await self._run_web_agent_execution_stage(tasks)
                else:
                    logger.info("📝 Using regular task execution")
                    execution_results = self._run_task_execution_stage(tasks)
                
                results["stages"]["task_execution"] = execution_results
                
                # Stage 4: Evaluation
                logger.info("📊 Stage: Evaluation")
                evaluation_results = self._run_evaluation_stage(execution_results, tasks)
                results["stages"]["evaluation"] = evaluation_results
                
                # Calculate overall metrics
                results["success"] = True
                results["total_time"] = time.time() - self.start_time
                
                # Save evaluation results to unified output directory
                if output_dir:
                    self._save_evaluation_results(results, output_dir)
                else:
                    # Use existing output directories
                    output_dirs = self.output_dirs
                    results_dir = output_dirs["results"]
                    self._save_evaluation_results(results, str(results_dir))
                
                # Save benchmark results
                self._save_results()
                
                logger.info(f"✅ Agent evaluation completed successfully in {results['total_time']:.2f}s")
                
        except Exception as e:
                logger.error(f"❌ Agent evaluation failed: {e}")
                results["success"] = False
                results["error"] = str(e)
                results["total_time"] = time.time() - self.start_time
                
                # Save failed results for debugging
                self._save_results()
                raise
            
        return results
    

    
    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to specified directory"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_path / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(results), f, indent=2, ensure_ascii=False)
            
            # Save evaluation CSV files if available
            evaluation_results = results.get("stages", {}).get("evaluation", {})
            task_metrics = evaluation_results.get("task_metrics", [])
            if task_metrics:
                self._save_evaluation_csvs(output_path, task_metrics, 
                                        evaluation_results.get("safety_metrics", {}),
                                        evaluation_results.get("metrics", {}),
                                        {})
            
            logger.info(f"💾 Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {e}")
    
    def _load_dataset(self, dataset_path: str) -> List[Any]:
        """Load tasks from dataset file"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        tasks = []
        
        if dataset_path.suffix.lower() == '.jsonl':
            # Load from JSONL format
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task_data = json.loads(line)
                        
                        # Check if this is a web task
                        if 'web_task_type' in task_data or 'task_steps' in task_data:
                            # This is a web task, convert to WebTaskInstance
                            from task_craft.task_generator import WebTaskInstance, WebTaskStep
                            
                            # Convert task_steps to WebTaskStep objects
                            task_steps = []
                            for step_data in task_data.get('task_steps', []):
                                step = WebTaskStep(
                                    step_type=step_data.get('step_type', 'navigation'),
                                    target_som_mark=step_data.get('target_som_mark', ''),
                                    action_description=step_data.get('action_description', ''),
                                    input_value=step_data.get('input_value', ''),
                                    expected_element=step_data.get('expected_element', ''),
                                    expected_result=step_data.get('expected_result', '')
                                )
                                task_steps.append(step)
                            
                            # Create WebTaskInstance with simplified structure
                            web_task = WebTaskInstance(
                                task_id=task_data.get('task_id', ''),
                                prompt=task_data.get('prompt', ''),
                                web_task_type=task_data.get('web_task_type', ''),
                                difficulty=task_data.get('difficulty', 'MEDIUM'),
                                task_steps=task_steps,
                                start_page=task_data.get('start_page', task_data.get('start_page_url', '')),
                                som_validated=task_data.get('som_validated', True),
                                som_elements_used=task_data.get('som_elements_used', []),
                                success_criteria=task_data.get('success_criteria', {}),
                                quality_score=task_data.get('quality_score', 0.8),
                                passed_quality_check=task_data.get('passed_quality_check', True),
                                expected_duration=task_data.get('expected_duration', 60)
                            )
                            tasks.append(web_task)
                        else:
                            # Convert to TaskInstance for regular tasks
                            from task_craft.task_generator import TaskInstance
                            from task_craft.task_templates import TaskDifficulty, TaskType
                            
                            task = TaskInstance(
                                task_id=task_data.get('task_id', ''),
                                template_id=task_data.get('template_id', ''),
                                task_type=TaskType(task_data.get('task_type', 'comprehension')),
                                difficulty=TaskDifficulty(task_data.get('difficulty', 'medium')),
                                prompt=task_data.get('prompt', ''),
                                gold_answer=task_data.get('gold_answer', ''),
                                gold_nodes=task_data.get('gold_nodes', []),
                                gold_edges=task_data.get('gold_edges', []),
                                subgraph_nodes=task_data.get('subgraph_nodes', []),
                                subgraph_edges=task_data.get('subgraph_edges', []),
                                quality_score=task_data.get('quality_score', None),
                                quality_details=task_data.get('quality_details', {}),
                                quality_reasoning=task_data.get('quality_reasoning', None),
                                passed_quality_check=task_data.get('passed_quality_check', True)
                            )
                            tasks.append(task)
        elif dataset_path.suffix.lower() == '.json':
            # Load from JSON format
            with open(dataset_path, 'r', encoding='utf-8') as f:
                task_data_list = json.load(f)
                for task_data in task_data_list:
                    # Convert to TaskInstance (same as above)
                    from task_craft.task_generator import TaskInstance
                    from task_craft.task_templates import TaskDifficulty
                    
                    task = TaskInstance(
                        task_id=task_data.get('task_id', ''),
                        template_id=task_data.get('template_id', ''),
                        task_type=TaskType(task_data.get('task_type', 'comprehension')),
                        difficulty=TaskDifficulty(task_data.get('difficulty', 'medium')),
                        prompt=task_data.get('prompt', ''),
                        gold_answer=task_data.get('gold_answer', ''),
                        gold_nodes=task_data.get('gold_nodes', []),
                        gold_edges=task_data.get('gold_edges', []),
                        subgraph_nodes=task_data.get('subgraph_nodes', []),
                        subgraph_edges=task_data.get('subgraph_edges', [])
                    )
                    tasks.append(task)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
        
        logger.info(f"✅ Loaded {len(tasks)} tasks from {dataset_path}")
        if tasks:
            logger.info(f"✅ First task type: {type(tasks[0])}")
            if hasattr(tasks[0], 'web_task_type'):
                logger.info(f"✅ First task web_task_type: {tasks[0].web_task_type}")
        return tasks
    
    def _detect_dataset_paths(self, base_path: str, graph_path: Optional[str] = None) -> Dict[str, str]:
        """Detect dataset and graph paths from a base directory"""
        base_path = Path(base_path)
        logger.info(f"🔍 Detecting dataset paths from base_path: {base_path}")
        
        # Look for dataset files only in the specified base_path
        dataset_paths = {}
        
        # Check for all_tasks.jsonl in base_path/datasets/
        all_tasks_path = base_path / "datasets" / "all_tasks.jsonl"
        if all_tasks_path.exists():
            dataset_paths["all_tasks"] = str(all_tasks_path)
        
        # Check for text_tasks.jsonl in base_path/datasets/ (alternative naming)
        text_tasks_path = base_path / "datasets" / "text_tasks.jsonl"
        if text_tasks_path.exists():
            dataset_paths["all_tasks"] = str(text_tasks_path)
        
        # Check for safety_tasks.jsonl in base_path/datasets/
        safety_tasks_path = base_path / "datasets" / "safety_tasks.jsonl"
        if safety_tasks_path.exists():
            dataset_paths["safety_tasks"] = str(safety_tasks_path)
        
        # Check for web_tasks.jsonl in base_path/datasets/
        web_tasks_path = base_path / "datasets" / "web_tasks.jsonl"
        if web_tasks_path.exists():
            dataset_paths["all_web_tasks"] = str(web_tasks_path)
        
        # Check for all_web_tasks.jsonl in base_path/datasets/ (legacy)
        all_web_tasks_path = base_path / "datasets" / "all_web_tasks.jsonl"
        if all_web_tasks_path.exists():
            dataset_paths["all_web_tasks"] = str(all_web_tasks_path)
        
        # Check for normal_web_tasks.jsonl in base_path/datasets/
        normal_web_tasks_path = base_path / "datasets" / "normal_web_tasks.jsonl"
        if normal_web_tasks_path.exists():
            dataset_paths["normal_web_tasks"] = str(normal_web_tasks_path)
        
        # Check for safety_web_tasks.jsonl in base_path/datasets/
        safety_web_tasks_path = base_path / "datasets" / "safety_web_tasks.jsonl"
        if safety_web_tasks_path.exists():
            dataset_paths["safety_web_tasks"] = str(safety_web_tasks_path)
        
        # Look for graph files - prioritize generate folder, then use provided graph_path or auto-detect latest graph directory
        if graph_path:
            graph_base = Path(graph_path)
            graph_file = graph_base / "graph" / "knowledge_graph.json"
            vectors_dir = graph_base / "vectors"
        else:
            # First, check if graph files exist in the current generate folder (base_path)
            generate_graph_file = base_path / "graph" / "knowledge_graph.json"
            generate_vectors_dir = base_path / "vectors"
            
            if generate_graph_file.exists() and generate_vectors_dir.exists():
                # Use graph files from generate folder
                graph_base = base_path
                graph_file = generate_graph_file
                vectors_dir = generate_vectors_dir
                logger.info(f"🕸️ Using graph files from generate folder: {graph_base}")
            else:
                # Fallback to auto-detect latest graph directory
                graph_base = self._auto_detect_latest_graph()
                if not graph_base:
                    raise FileNotFoundError("No graph directory found. Please specify with -g or run graph mode first")
                graph_file = Path(graph_base) / "graph" / "knowledge_graph.json"
                vectors_dir = Path(graph_base) / "vectors"
        
        if not graph_file.exists():
            raise FileNotFoundError(f"Knowledge graph not found at {graph_file}")
        
        if not vectors_dir.exists():
            raise FileNotFoundError(f"Vectors directory not found at {vectors_dir}")
        
        # Check if vector files exist in the vectors directory
        vectors_faiss_file = vectors_dir / "vectors_faiss.faiss"
        if not vectors_faiss_file.exists():
            raise FileNotFoundError(f"Vector index file not found at {vectors_faiss_file}")
        
        # Create result dictionary - return directory paths for graph and vectors
        result = {
            "datasets": dataset_paths,
            "graph": str(graph_base) if graph_path else str(base_path),
            "vectors": str(vectors_faiss_file)
        }
        
        # Log detected paths in JSON format
        import json
        logger.info(f"📋 Detected dataset paths: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result
    
    def _detect_graph_from_dataset(self, tasks: List[Any], dataset_path: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Detect graph path from dataset tasks' source_document field"""
        if not tasks:
            logger.warning("No tasks provided for graph detection")
            return None, None
        
        # Get source_document from first task
        first_task = tasks[0]
        source_document = None
        
        if hasattr(first_task, 'source_document'):
            source_document = first_task.source_document
        elif isinstance(first_task, dict):
            source_document = first_task.get('source_document')
        
        if not source_document:
            logger.warning("No source_document found in tasks")
            return None, None
        
        logger.info(f"🔍 Detected source_document: {source_document}")
        
        # Search for graph directory in current project
        project_root = Path.cwd()
        
        # First, check if graph files exist in generate folder
        if dataset_path:
            dataset_path_obj = Path(dataset_path)
            generate_graph_file = dataset_path_obj / "graph" / "knowledge_graph.json"
            generate_vectors_dir = dataset_path_obj / "vectors"
            
            if generate_graph_file.exists() and generate_vectors_dir.exists():
                logger.info(f"🕸️ Using graph files from generate folder: {dataset_path_obj}")
                return str(dataset_path_obj), str(generate_vectors_dir)
        
        # Fallback to original graph directory search
        graph_base_dir = project_root / "output" / "graph"
        
        if not graph_base_dir.exists():
            logger.warning(f"Graph base directory not found: {graph_base_dir}")
            return None, None
        
        # Look for the specific graph directory
        target_graph_dir = graph_base_dir / source_document
        
        if not target_graph_dir.exists():
            logger.warning(f"Graph directory not found: {target_graph_dir}")
            # Try to find similar directories
            similar_dirs = [d for d in graph_base_dir.iterdir() if d.is_dir() and source_document in d.name]
            if similar_dirs:
                logger.info(f"Found similar directories: {[d.name for d in similar_dirs]}")
                target_graph_dir = similar_dirs[0]  # Use the first match
                logger.info(f"Using similar directory: {target_graph_dir}")
            else:
                return None, None
        
        # Check if graph files exist
        graph_file = target_graph_dir / "graph" / "knowledge_graph.json"
        vectors_dir = target_graph_dir / "vectors"
        
        if not graph_file.exists():
            logger.warning(f"Graph file not found: {graph_file}")
            return None, None
        
        if not vectors_dir.exists():
            logger.warning(f"Vectors directory not found: {vectors_dir}")
            return None, None
        
        logger.info(f"✅ Found graph directory: {target_graph_dir}")
        logger.info(f"✅ Graph file: {graph_file}")
        logger.info(f"✅ Vectors directory: {vectors_dir}")
        
        return str(target_graph_dir), str(vectors_dir)
    
    def _load_graph(self, graph_path: str, vectors_path: str, dataset_path: str = None):
        """Load graph from file - use DocumentGraph for text tasks, TaskGraph for web tasks"""
        from graph_rag.graph_builder import DocumentGraph, TaskGraph
        from graph_rag.storage import JSONStorage
        from graph_rag.embeddings import EmbeddingManager
        
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph path not found: {graph_path}")
        
        # If graph_path is a directory, look for knowledge_graph.json inside
        if graph_path.is_dir():
            graph_file = graph_path / "graph" / "knowledge_graph.json"
            if not graph_file.exists():
                raise FileNotFoundError(f"Knowledge graph file not found: {graph_file}")
            # Use the graph subdirectory as the path for loading
            graph_dir = graph_path / "graph"
        else:
            graph_dir = graph_path
        
        vectors_path = Path(vectors_path)
        if not vectors_path.exists():
            raise FileNotFoundError(f"Vectors path not found: {vectors_path}")
        
        # If vectors_path is a file, get its parent directory
        if vectors_path.is_file():
            vectors_dir = vectors_path.parent
        else:
            vectors_dir = vectors_path
        
        # Determine graph type based on dataset path
        is_web_task = False
        if dataset_path:
            dataset_path = Path(dataset_path)
            if dataset_path.name in ['web_tasks.jsonl', 'all_web_tasks.jsonl', 'normal_web_tasks.jsonl', 'safety_web_tasks.jsonl']:
                is_web_task = True
        
        if is_web_task:
            # For web tasks, use TaskGraph
            graph = TaskGraph.load(str(graph_dir), str(vectors_dir))
            logger.info(f"✅ Loaded TaskGraph from {graph_dir}")
        else:
            # For text tasks, use DocumentGraph
            storage = JSONStorage()
            embedding_manager = EmbeddingManager()
            graph = DocumentGraph(storage, embedding_manager)
            graph.load(str(graph_dir), str(vectors_dir))
            logger.info(f"✅ Loaded DocumentGraph from {graph_dir}")
        
        logger.info(f"✅ Loaded vectors from {vectors_dir}")
        return graph
    
    def _run_ingestion_stage(self, input_documents: List[str]) -> Dict[str, Any]:
        """Run document ingestion stage"""
        logger.info(f"{'='*60}")
        logger.info("📖 DOCUMENT INGESTION STAGE")
        logger.info(f"{'='*60}")
        logger.info(f"📄 Processing {len(input_documents)} documents")
        
        stage_start = time.time()
        parsed_documents = []
        
        # Create progress bar for document processing
        doc_pbar = tqdm(
            total=len(input_documents),
            desc="📖 Processing Documents",
            unit="doc",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for doc_path in input_documents:
            try:
                path = Path(doc_path)
                if not path.exists():
                    logger.warning(f"Document not found: {doc_path}")
                    continue
                
                # Determine parser based on file extension
                if path.suffix.lower() == '.pdf' and 'pdf_parser' in self.components:
                    document = self.components['pdf_parser'].parse(doc_path)
                elif path.suffix.lower() in ['.html', '.htm'] and 'html_parser' in self.components:
                    document = self.components['html_parser'].parse(doc_path)
                elif path.suffix.lower() in ['.txt', '.md']:
                    # Simple text parsing
                    from ingestion.parsers import DocumentStructure, ParsedElement
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    elements = []
                    for i, para in enumerate(paragraphs):
                        if len(para) > 20:  # Filter short paragraphs
                            element = ParsedElement(
                                element_type="paragraph",
                                content=para,
                                metadata={"paragraph_index": i},
                                element_id=f"para_{i}"
                            )
                            elements.append(element)
                    
                    document = DocumentStructure(
                        elements=elements,
                        metadata={"total_paragraphs": len(paragraphs), "source_path": str(path)},
                        total_pages=1,
                        file_path=str(path)
                    )
                else:
                    logger.warning(f"No parser available for {doc_path}")
                    continue
                
                # Clean document if cleaner available
                if 'text_cleaner' in self.components:
                    document = self.components['text_cleaner'].clean(document)
                
                parsed_documents.append(document)
                doc_pbar.set_postfix({"Elements": f"{len(document.elements)}"})
                logger.info(f"✅ Parsed {doc_path}: {len(document.elements)} elements")
                doc_pbar.update(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to parse {doc_path}: {e}")
                doc_pbar.update(1)
        
        # Close document processing progress bar
        doc_pbar.close()
        
        # Print ingestion summary
        logger.info(f"{'='*60}")
        logger.info("📖 INGESTION RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"📄 Documents processed: {len(parsed_documents)}")
        logger.info(f"📄 Total elements: {sum(len(doc.elements) for doc in parsed_documents)}")
        logger.info(f"⏱️  Processing time: {time.time() - stage_start:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Save serializable document information
        serializable_documents = []
        for doc in parsed_documents:
            doc_info = {
                "file_path": getattr(doc, 'file_path', ''),
                "total_pages": getattr(doc, 'total_pages', 1),
                "total_elements": len(doc.elements),
                "metadata": getattr(doc, 'metadata', {}),
                "elements_summary": [
                    {
                        "element_type": elem.element_type,
                        "element_id": getattr(elem, 'element_id', ''),
                        "content_preview": elem.content[:100] + "..." if len(elem.content) > 100 else elem.content,
                        "metadata": getattr(elem, 'metadata', {})
                    }
                    for elem in doc.elements[:10]  # Only save summary of first 10 elements
                ]
            }
            serializable_documents.append(doc_info)
        
        return {
            "documents_processed": len(parsed_documents),
            "total_elements": sum(len(doc.elements) for doc in parsed_documents),
            "processing_time": time.time() - stage_start,
            "documents": serializable_documents,
            "raw_documents": parsed_documents  # Keep original document objects for subsequent processing
        }
    
    def _is_mock_response_task(self, result_data: dict) -> bool:
        """Check if a task result used mock response (indicating API failure)"""
        answer = result_data.get('answer', '')
        
        # Check for known mock response patterns
        mock_patterns = [
            "GraphRAG shows better performance than traditional RAG in terms of retrieval accuracy and response quality.",
            "The key information includes: GraphRAG architecture, real-time retrieval, and memory optimization techniques.",
            "The overall findings suggest that GraphRAG provides significant improvements in information retrieval and knowledge representation.",
            "Based on the provided context, the system demonstrates advanced capabilities in knowledge representation and retrieval."
        ]
        
        # Check if answer matches any mock pattern
        for pattern in mock_patterns:
            if answer.strip() == pattern.strip():
                return True
        
        # Also check for mock citations patterns (chunk_1, chunk_2, etc.)
        citations = result_data.get('citations', [])
        if citations:
            # Check if citations are generic mock citations
            mock_citation_patterns = ['chunk_1', 'chunk_2', 'chunk_3', 'chunk_4', 'entity_1', 'entity_2', 'entity_3', 'para_2']
            if all(citation in mock_citation_patterns for citation in citations):
                return True
        
        return False

    def _get_completed_task_ids(self, results_dir: Path) -> set:
        """Get set of task IDs that have already been completed successfully"""
        completed_task_ids = set()
        failed_task_ids = set()
        
        try:
            individual_results_dir = results_dir / "individual_results"
            if individual_results_dir.exists():
                # Look for task result files (support both task_*.json and web_task_*.json)
                for result_file in individual_results_dir.glob("*task*.json"):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            task_id = result_data.get('task_id')
                            
                            if task_id:
                                # Check if task has an error_type or error_message (technical failure)
                                error_type = result_data.get('error_type')
                                error_message = result_data.get('error_message')
                                answer = result_data.get('answer', '')
                                
                                # Check for old fallback response pattern
                                is_old_fallback = answer.startswith("Based on the provided context, the system demonstrates advanced capabilities in knowledge representation and retrieval.")
                                
                                if is_old_fallback:
                                    # Task had old fallback response - needs retry
                                    failed_task_ids.add(task_id)
                                    logger.info(f"🔍 Found task {task_id} with old fallback response - will be retried")
                                elif error_type or error_message:
                                    # Task had technical errors - needs retry
                                    failed_task_ids.add(task_id)
                                    logger.info(f"🔍 Found task {task_id} with technical error ({error_type}) - will be retried")
                                else:
                                    # Task completed without old fallback response - consider it done
                                    completed_task_ids.add(task_id)
                    except Exception as e:
                        logger.warning(f"Failed to parse result file {result_file}: {e}")
                        continue
                        
                logger.info(f"🔍 Found {len(completed_task_ids)} completed tasks in {individual_results_dir}")
                if failed_task_ids:
                    logger.info(f"🔍 Found {len(failed_task_ids)} failed tasks that will be retried")
            else:
                logger.info(f"🔍 No individual_results directory found at {individual_results_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to check completed tasks: {e}")
            
        return completed_task_ids

    def _load_all_task_results(self) -> List[Any]:
        """Load all task results from individual_results directory, preferring latest results over mock responses"""
        task_results_map = {}  # task_id -> (file_path, result_data, is_mock)
        
        try:
            # Use existing output directories
            output_dirs = self.output_dirs
            results_dir = output_dirs["results"]
            individual_results_dir = results_dir / "individual_results"
            
            if not individual_results_dir.exists():
                logger.info(f"🔍 No individual_results directory found at {individual_results_dir}")
                return []
            
            # Load all task result files and group by task_id (support both task_*.json and web_task_*.json)
            for result_file in individual_results_dir.glob("*task*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        task_id = result_data.get('task_id', '')
                        
                        if not task_id:
                            continue
                        
                        # Check if this is a mock response
                        is_mock = self._is_mock_response_task(result_data)
                        
                        # If we already have a result for this task_id
                        if task_id in task_results_map:
                            existing_path, existing_data, existing_is_mock = task_results_map[task_id]
                            
                            # Prefer real results over mock responses
                            if is_mock and not existing_is_mock:
                                logger.debug(f"🔍 Keeping real result for {task_id}, ignoring mock result from {result_file}")
                                continue
                            elif not is_mock and existing_is_mock:
                                logger.info(f"🔄 Replacing mock result for {task_id} with real result from {result_file}")
                                # Optionally delete the old mock file
                                try:
                                    existing_path.unlink()
                                    logger.info(f"🗑️  Deleted mock response file: {existing_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete mock file {existing_path}: {e}")
                            else:
                                # Both are same type (both real or both mock), keep the newer one
                                if result_file.stat().st_mtime > existing_path.stat().st_mtime:
                                    logger.debug(f"🔍 Replacing older result for {task_id} with newer result from {result_file}")
                                    # Optionally delete the older file if it's a mock
                                    if existing_is_mock:
                                        try:
                                            existing_path.unlink()
                                            logger.info(f"🗑️  Deleted older mock response file: {existing_path}")
                                        except Exception as e:
                                            logger.warning(f"Failed to delete older mock file {existing_path}: {e}")
                                else:
                                    logger.debug(f"🔍 Keeping existing result for {task_id}, ignoring older result from {result_file}")
                                    continue
                        
                        task_results_map[task_id] = (result_file, result_data, is_mock)
                        
                except Exception as e:
                    logger.warning(f"Failed to load task result from {result_file}: {e}")
                    continue
            
            # Convert to TaskResult objects
            all_results = []
            for task_id, (file_path, result_data, is_mock) in task_results_map.items():
                # Convert dict to ExecutionResult-like object
                class TaskResult:
                    def __init__(self, data):
                        self.task_id = data.get('task_id', '')
                        self.success = data.get('success', False)
                        self.answer = data.get('answer', '')
                        self.citations = data.get('citations', [])
                        self.reasoning_path = data.get('reasoning_path', [])
                        self.confidence = data.get('confidence', 0.0)
                        self.execution_time = data.get('execution_time', 0.0)
                        self.tokens_used = data.get('tokens_used', 0)
                        self.model_used = data.get('model_used', '')
                        self.retries_needed = data.get('retries_needed', 0)
                        self.error_type = data.get('error_type')
                        self.error_message = data.get('error_message')
                        self.raw_response = data.get('raw_response', '')
                        self.task_type = data.get('task_type', '')
                        self.web_task_type = data.get('web_task_type', '')
                        self.prompt = data.get('prompt', '')
                        self.gold_answer = data.get('gold_answer', '')
                        self.safety_check = data.get('safety_check', {})
                        self.multi_agent = data.get('multi_agent', {})
                    
                    def to_dict(self):
                        return {
                            "task_id": self.task_id,
                            "success": self.success,
                            "answer": self.answer,
                            "citations": self.citations,
                            "reasoning_path": self.reasoning_path,
                            "confidence": self.confidence,
                            "execution_time": self.execution_time,
                            "tokens_used": self.tokens_used,
                            "model_used": self.model_used,
                            "retries_needed": self.retries_needed,
                            "error_type": self.error_type,
                            "error_message": self.error_message,
                            "raw_response": self.raw_response,
                            "task_type": self.task_type,
                            "prompt": self.prompt,
                            "gold_answer": self.gold_answer,
                            "safety_check": self.safety_check,
                            "multi_agent": self.multi_agent
                        }
                
                task_result = TaskResult(result_data)
                all_results.append(task_result)
            
            logger.info(f"🔍 Loaded {len(all_results)} unique task results from {individual_results_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to load task results: {e}")
        
        return all_results
    
    def _run_graph_construction_stage(self, ingestion_results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run graph construction stage"""
        logger.info(f"{'='*60}")
        logger.info("🕸️ GRAPH CONSTRUCTION STAGE")
        logger.info(f"{'='*60}")
        
        stage_start = time.time()
        graphs = []
        
        # Use original document objects instead of serialized dictionaries
        parsed_documents = ingestion_results.get("raw_documents", [])
        
        if not parsed_documents:
            logger.warning("❌ No parsed documents available for graph construction")
            return {"graphs_created": 0, "processing_time": 0}
        
        logger.info(f"🕸️ Building graphs from {len(parsed_documents)} documents")
        
        # Create progress bar for graph construction
        graph_pbar = tqdm(
            total=len(parsed_documents),
            desc="🕸️ Building Graphs",
            unit="graph",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for i, document in enumerate(parsed_documents):
            try:
                if 'graph_builder' in self.components:
                    graph = self.components['graph_builder'].build_graph(document)
                    graphs.append(graph)
                    stats = graph.get_stats()
                    graph_pbar.set_postfix({"Nodes": f"{stats['total_nodes']}", "Edges": f"{stats['total_edges']}"})
                    logger.info(f"✅ Built graph {i+1}: {stats}")
                    graph_pbar.update(1)
                else:
                    logger.warning("Graph builder not available")
                    graph_pbar.close()
                    break
                    
            except Exception as e:
                logger.error(f"❌ Failed to build graph for document {i+1}: {e}")
                graph_pbar.update(1)
        
        # Close graph construction progress bar
        graph_pbar.close()
        
        # Print graph construction summary
        total_nodes = sum(g.get_stats()["total_nodes"] for g in graphs)
        total_edges = sum(g.get_stats()["total_edges"] for g in graphs)
        
        logger.info(f"{'='*60}")
        logger.info("🕸️ GRAPH CONSTRUCTION RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"🕸️ Graphs created: {len(graphs)}")
        logger.info(f"🕸️ Total nodes: {total_nodes}")
        logger.info(f"🕸️ Total edges: {total_edges}")
        logger.info(f"⏱️  Processing time: {time.time() - stage_start:.2f}s")
        logger.info(f"{'='*60}\n")
        
        return {
            "graphs_created": len(graphs),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "processing_time": time.time() - stage_start,
            "graphs": graphs
        }
    
    def _run_task_generation_stage(self, graph_results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run task generation stage"""
        logger.info(f"{'='*60}")
        logger.info("🎯 TASK GENERATION STAGE")
        logger.info(f"{'='*60}")
        
        stage_start = time.time()
        all_tasks = []
        safety_tasks = []
        
        graphs = graph_results.get("graphs", [])
        
        if not graphs:
            logger.warning("❌ No graphs available for task generation")
            return {"tasks_generated": 0, "safety_tasks_generated": 0, "processing_time": 0}
        
        logger.info(f"🎯 Generating tasks from {len(graphs)} graphs")
        
        # Extract graph nodes for safety task generation
        graph_nodes = []
        for graph in graphs:
            # Use find_nodes() to get all nodes instead of get_all_nodes()
            nodes = graph.storage.find_nodes()
            for node in nodes:
                graph_nodes.append({
                    'id': node.node_id,
                    'content': node.content,
                    'type': node.node_type.value,
                    'metadata': node.metadata
                })
        
        # Generate normal tasks
        normal_tasks = []
        
        # Create progress bar for normal task generation
        normal_task_pbar = tqdm(
            total=len(graphs),
            desc="🎯 Generating Normal Tasks",
            unit="graph",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for i, graph in enumerate(graphs):
            try:
                if 'task_generator' in self.components:
                    tasks = self.components['task_generator'].generate_tasks(graph, f"document_{i+1}", output_dir)
                    normal_tasks.extend(tasks)
                    normal_task_pbar.set_postfix({"Tasks": f"{len(tasks)}"})
                    logger.info(f"✅ Generated {len(tasks)} normal tasks from graph {i+1}")
                    normal_task_pbar.update(1)
                else:
                    logger.warning("Task generator not available")
                    normal_task_pbar.close()
                    break
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate tasks from graph {i+1}: {e}")
                normal_task_pbar.update(1)
        
        # Close normal task generation progress bar
        normal_task_pbar.close()
        
        # Generate safety tasks from GraphRAG nodes with policy documents
        try:
            if 'safety_generator' in self.components and graph_nodes:
                # Get policy documents from configuration
                policy_documents = self._get_policy_documents()
                
                if policy_documents:
                    logger.info(f"📋 Using {len(policy_documents)} policy documents for dynamic safety task generation")
                    safety_tasks = self.components['safety_generator'].generate_safety_tasks_from_graph(
                        graph_nodes, policy_documents
                    )
                    logger.info(f"✅ Generated {len(safety_tasks)} dynamic safety tasks from policy documents")
                else:
                    logger.warning("⚠️ No policy documents found for dynamic safety task generation")
                    safety_tasks = []
                
                # Log safety task types distribution
                safety_task_types = {}
                for task in safety_tasks:
                    # Handle both web tasks and regular tasks
                    if hasattr(task, 'web_task_type'):
                        task_type = task.web_task_type
                    else:
                        task_type = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
                    safety_task_types[task_type] = safety_task_types.get(task_type, 0) + 1
                
                logger.info(f"📊 Safety task types: {safety_task_types}")
            else:
                logger.warning("⚠️ Safety generator not available or no graph nodes")
                safety_tasks = []
        except Exception as e:
            logger.error(f"❌ Failed to generate safety tasks: {e}")
            safety_tasks = []
        
        # Combine all tasks
        all_tasks = normal_tasks + safety_tasks
        
        # Print task generation summary
        logger.info(f"{'='*60}")
        logger.info("🎯 TASK GENERATION RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total tasks: {len(all_tasks)}")
        logger.info(f"🎯 Normal tasks: {len(normal_tasks)}")
        logger.info(f"🔒 Safety tasks: {len(safety_tasks)}")
        logger.info(f"⏱️  Processing time: {time.time() - stage_start:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Get detailed task statistics
        task_stats = self.get_task_statistics(all_tasks)
        
        # Log task statistics with proper formatting
        logger.info(f"📊 Task Statistics: {task_stats}")
        
        # Save tasks to stage-specific directory and create datasets
        if all_tasks:
            # Read task_craft configuration from main_config.yaml
            main_config_path = Path(self.config_dir) / "main_config.yaml"
            if main_config_path.exists():
                with open(main_config_path, 'r', encoding='utf-8') as f:
                    import yaml
                    main_config_data = yaml.safe_load(f)
                    dataset_creation_config = main_config_data.get('benchmark', {}).get('dataset_creation', {})
            else:
                dataset_creation_config = {}
            

            
            # Create datasets
            dataset_creation_enabled = dataset_creation_config.get('enabled', True)
            if dataset_creation_enabled:
                # Use existing output directory
                output_dirs = self.output_dirs
                run_result_dir = output_dirs["run_result"]
                
                datasets_created = 0
                
                # 1. Create normal task datasets (normal_task_datasets)
                if normal_tasks:
                    normal_file = run_result_dir / "normal_task_datasets.jsonl"
                    with open(normal_file, 'w', encoding='utf-8') as f:
                        for task in normal_tasks:
                            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
                    
                    datasets_created += 1
                    logger.info(f"✅ Created normal_task_datasets with {len(normal_tasks)} tasks: {normal_file}")
                
                # 2. Create safety task datasets (safety_task_datasets)
                if safety_tasks:
                    safety_file = run_result_dir / "safety_task_datasets.jsonl"
                    with open(safety_file, 'w', encoding='utf-8') as f:
                        for task in safety_tasks:
                            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + '\n')
                    
                    datasets_created += 1
                    logger.info(f"✅ Created safety_task_datasets with {len(safety_tasks)} tasks: {safety_file}")
                
                logger.info(f"📦 Created {datasets_created} datasets in {run_result_dir}")
            else:
                logger.info("📦 Dataset creation disabled in configuration")
        
        # Calculate GraphRAG-specific metrics
        # task_node: Initial number of tasks from document nodes (before graph expansion)
        task_node = len(graph_nodes)  # Each node can potentially generate a task
        
        # task_node_expand: Number of tasks after graph expansion
        task_node_expand = len(all_tasks)  # Total tasks generated after graph processing
        
        return {
            "tasks_generated": len(normal_tasks),
            "safety_tasks_generated": len(safety_tasks),
            "total_tasks": len(all_tasks),
            "normal_tasks": len(normal_tasks),
            "safety_tasks": len(safety_tasks),
            "task_statistics": task_stats,
            "task_types": self._count_task_types(all_tasks),
            "difficulty_distribution": self._count_difficulties(all_tasks),
            "processing_time": time.time() - stage_start,
            "tasks": all_tasks[:100],  # Limit stored tasks for memory
            "graph_nodes_count": len(graph_nodes),
            "datasets_created": datasets_created if 'datasets_created' in locals() else 0,
            "datasets_directory": str(run_result_dir) if 'run_result_dir' in locals() else "",
            # GraphRAG-specific metrics
            "task_node": task_node,
            "task_node_expand": task_node_expand
        }
    
    def _run_task_execution_stage(self, tasks: List[Any]) -> Dict[str, Any]:
        """Run task execution stage"""
        logger.info(f"{'='*60}")
        logger.info("🚀 TASK EXECUTION STAGE")
        logger.info(f"{'='*60}")
        
        stage_start = time.time()
        logger.info(f"🚀 Executing {len(tasks)} tasks")
        
        # Use existing output directories
        output_dirs = self.output_dirs
        
        # Check for existing task results to skip already completed tasks
        completed_task_ids = self._get_completed_task_ids(output_dirs["results"])
        logger.info(f"🔍 Found {len(completed_task_ids)} already completed tasks")
        
        # Mock execution for demo
        execution_results = []
        successful_executions = 0
        failed_executions = 0
        
        # Read configuration from main_config.yaml
        main_config_path = Path(self.config_dir) / "main_config.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r', encoding='utf-8') as f:
                import yaml
                main_config_data = yaml.safe_load(f)
                benchmark_config = main_config_data.get('benchmark', {})
                task_execution_config = benchmark_config.get('task_execution', {})
                safety_tasks_config = benchmark_config.get('safety_tasks', {})
        else:
            benchmark_config = {}
            task_execution_config = {}
            safety_tasks_config = {}
        
        # Check if safety tasks are enabled
        safety_tasks_enabled = safety_tasks_config.get('enabled', True)
        
        # Get task limits from configuration
        max_total_tasks = task_execution_config.get('max_total_tasks', 20)
        max_normal_tasks = task_execution_config.get('max_normal_tasks', 10)
        max_safety_tasks = task_execution_config.get('max_safety_tasks', 10)
        
        # Separate normal and safety tasks
        from task_craft.text_task_types import TextTaskType
        normal_tasks = []
        safety_tasks = []
        
        logger.info(f"🔍 Debug: Starting task classification for {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            try:
                logger.info(f"🔍 Debug: Classifying task {i}: {task.task_id}")
                
                # Handle both web tasks and regular tasks
                if hasattr(task, 'web_task_type'):
                    # Web task - check web_task_type
                    logger.info(f"🔍 Debug: Web task type: {task.web_task_type}")
                    if task.web_task_type == "safety" or task.web_task_type.startswith('web_safety_'):
                        safety_tasks.append(task)
                        logger.info(f"🔍 Debug: Classified as safety task")
                    else:
                        normal_tasks.append(task)
                        logger.info(f"🔍 Debug: Classified as normal web task")
                elif hasattr(task, 'text_task_type') and task.text_task_type is not None:
                    # Text task - check text_task_type
                    logger.info(f"🔍 Debug: Text task, checking is_safety_task")
                    try:
                        is_safety = TextTaskType.is_safety_task(task.text_task_type)
                        logger.info(f"🔍 Debug: is_safety_task result: {is_safety}")
                        if is_safety:
                            safety_tasks.append(task)
                            logger.info(f"🔍 Debug: Classified as safety task")
                        else:
                            normal_tasks.append(task)
                            logger.info(f"🔍 Debug: Classified as normal task")
                    except Exception as e:
                        logger.error(f"🔍 Debug: Error calling is_safety_task: {e}")
                        # Default to normal task if there's an error
                        normal_tasks.append(task)
                        logger.info(f"🔍 Debug: Error occurred, classified as normal task")
                else:
                    # Fallback: assume normal task
                    logger.info(f"🔍 Debug: No task type found, classified as normal task")
                    normal_tasks.append(task)
                        
            except Exception as e:
                logger.error(f"🔍 Debug: Error classifying task {i}: {e}")
                raise
        
        # Select tasks to execute based on configuration
        if safety_tasks_enabled:
            # Execute both normal and safety tasks
            normal_tasks_to_execute = normal_tasks[:min(len(normal_tasks), max_normal_tasks)]
            safety_tasks_to_execute = safety_tasks[:min(len(safety_tasks), max_safety_tasks)]
            tasks_to_execute = normal_tasks_to_execute + safety_tasks_to_execute
            
            # Ensure we don't exceed max_total_tasks
            if len(tasks_to_execute) > max_total_tasks:
                # Prioritize normal tasks if we exceed the limit
                normal_count = min(len(normal_tasks_to_execute), max_total_tasks // 2)
                safety_count = max_total_tasks - normal_count
                normal_tasks_to_execute = normal_tasks_to_execute[:normal_count]
                safety_tasks_to_execute = safety_tasks_to_execute[:safety_count]
                tasks_to_execute = normal_tasks_to_execute + safety_tasks_to_execute
        else:
            # Only execute normal tasks
            normal_tasks_to_execute = normal_tasks[:min(len(normal_tasks), max_total_tasks)]
            safety_tasks_to_execute = []
            tasks_to_execute = normal_tasks_to_execute
        
        # Filter out already completed tasks
        tasks_to_execute = [task for task in tasks_to_execute if task.task_id not in completed_task_ids]
        normal_tasks_to_execute = [task for task in normal_tasks_to_execute if task.task_id not in completed_task_ids]
        safety_tasks_to_execute = [task for task in safety_tasks_to_execute if task.task_id not in completed_task_ids]
        
        # Print execution summary
        logger.info(f"{'='*60}")
        logger.info("🏃‍♂️ TASK EXECUTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total tasks to execute: {len(tasks_to_execute)}")
        logger.info(f"📊 Normal tasks: {len(normal_tasks_to_execute)}")
        logger.info(f"📊 Safety tasks: {len(safety_tasks_to_execute)}")
        logger.info(f"📊 Configuration: max_total={max_total_tasks}, max_normal={max_normal_tasks}, max_safety={max_safety_tasks}")
        logger.info(f"📊 Safety tasks enabled: {safety_tasks_enabled}")
        logger.info(f"📊 Skipped completed tasks: {len(completed_task_ids)}")
        
        # Debug: Show first few tasks
        for i, task in enumerate(tasks_to_execute[:3]):
            logger.info(f"🔍 Debug: Task {i}: {task.task_id}, type: {task.task_type}")
        
        logger.info(f"{'='*60}\n")
        
        # Load graph from dataset directory for RAG agents
        graph = None
        if tasks_to_execute:
            logger.info("🔍 Loading graph from dataset directory for RAG agents")
            try:
                # Try to determine dataset path from current_dataset_dir (set in batch evaluation)
                dataset_path = None
                if hasattr(self, 'current_dataset_dir') and self.current_dataset_dir:
                    dataset_path = str(self.current_dataset_dir)
                    logger.info(f"🔍 Using current_dataset_dir: {dataset_path}")
                
                # If we have dataset_path, try to load from dataset directory
                if dataset_path:
                    logger.info(f"🔍 Loading graph from dataset: {dataset_path}")
                    
                    # Look for graph files in the dataset directory
                    graph_path = Path(dataset_path) / "graph" / "knowledge_graph.json"
                    vectors_path = Path(dataset_path) / "vectors"
                    
                    if graph_path.exists() and vectors_path.exists():
                        logger.info(f"🔍 Found graph files: {graph_path}, {vectors_path}")
                        graph = self._load_graph(str(graph_path), str(vectors_path), dataset_path)
                        logger.info("✅ Graph loaded successfully from dataset directory")
                        
                        # Set graph to agent if loaded successfully
                        if 'agent' in self.components:
                            if hasattr(self.components['agent'], 'set_graph'):
                                self.components['agent'].set_graph(graph)
                                logger.info("✅ Graph set to agent")
                            
                            # Set graph to storage component for multi-agent system
                            if 'storage' in self.components:
                                self.components['storage'].graph = graph
                                logger.info("✅ Graph set to storage component")
                    else:
                        logger.warning(f"⚠️ Graph files not found in dataset directory: {dataset_path}")
                        logger.warning(f"   Expected: {graph_path}")
                        logger.warning(f"   Expected: {vectors_path}")
                else:
                    logger.warning("⚠️ No dataset path available for graph loading")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to load graph from dataset directory: {e}")
                graph = None
        
        # Create progress bar for task execution
        pbar = tqdm(
            total=len(tasks_to_execute),
            desc="🏃‍♂️ Executing Tasks",
            unit="task",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        for task in tasks_to_execute:
            try:
                logger.info(f"🔍 Debug: Executing task {task.task_id}, type: {task.task_type}")
                logger.info(f"🔍 Debug: task.task_type type: {type(task.task_type)}")
                logger.info(f"🔍 Debug: hasattr(task.task_type, 'value'): {hasattr(task.task_type, 'value')}")
                if hasattr(task.task_type, 'value'):
                    logger.info(f"🔍 Debug: task.task_type.value: {task.task_type.value}")
                
                if 'agent' in self.components:
                    # Use unified agent interface
                    agent = self.components['agent']
                    
                    # Check if this is a multi-agent system
                    is_multi_agent = hasattr(agent, 'agents') and isinstance(agent.agents, dict) and hasattr(agent, 'config') and hasattr(agent.config, 'max_iterations')
                    
                    # Set graph for RAG Agent or Multi-Agent System if available
                    if hasattr(agent, 'set_graph') and graph:
                        agent.set_graph(graph)
                    
                    # Execute task using agent
                    if hasattr(agent, 'execute_task'):
                        # Execute task with appropriate agent type
                        if is_multi_agent:
                            # Multi-Agent System
                            logger.info(f"🤖 Using Multi-Agent System for task {task.task_id}")
                            agent_response = agent.execute_task(task, graph)
                            
                            # Extract multi-agent specific information
                            final_answer = agent_response.final_answer
                            planner_output = getattr(agent_response, 'planner_output', {})
                            retriever_output = getattr(agent_response, 'retriever_output', None)
                            reasoner_output = getattr(agent_response, 'reasoner_output', [])
                            verifier_output = getattr(agent_response, 'verifier_output', None)
                            summarizer_output = getattr(agent_response, 'summarizer_output', {})
                            
                            # Extract citations from retriever output
                            citations = []
                            if retriever_output and hasattr(retriever_output, 'nodes'):
                                # retriever_output.nodes is a list of strings (node IDs), not Node objects
                                citations = retriever_output.nodes
                            
                            # Extract reasoning path from reasoner output
                            reasoning_path = []
                            if reasoner_output:
                                for step in reasoner_output:
                                    if hasattr(step, 'description'):
                                        reasoning_path.append(step.description)
                            
                            # Extract confidence from verifier output
                            confidence = 1.0
                            if verifier_output and hasattr(verifier_output, 'confidence'):
                                confidence = verifier_output.confidence
                            
                            result_dict = {
                                "task_id": agent_response.task_id,
                                "success": agent_response.success,
                                "answer": final_answer,
                                "citations": citations,
                                "reasoning_path": reasoning_path,
                                "confidence": confidence,
                                "execution_time": agent_response.execution_time,
                                "tokens_used": agent_response.total_tokens,
                                "model_used": agent_response.model_used,
                                "retries_needed": 0,
                                "error_type": agent_response.error_type,
                                "error_message": agent_response.error_message,
                                "raw_response": final_answer,
                                "task_type": task.web_task_type if hasattr(task, 'web_task_type') else (task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)),
                                "prompt": task.prompt if hasattr(task, 'prompt') else "",
                                "gold_answer": task.gold_answer if hasattr(task, 'gold_answer') and task.gold_answer else "",
                                "safety_check": {
                                    "passed": True,  # Multi-agent system handles safety internally
                                    "issues": []
                                },
                                # Multi-agent specific fields
                                "multi_agent": {
                                    "planner_output": planner_output,
                                    "retriever_output": retriever_output.to_dict() if retriever_output else None,
                                    "reasoner_output": [step.to_dict() if hasattr(step, 'to_dict') else str(step) for step in reasoner_output],
                                    "verifier_output": verifier_output.to_dict() if verifier_output else None,
                                    "summarizer_output": summarizer_output,
                                    "total_tokens": agent_response.total_tokens
                                }
                            }
                        else:
                            # RAG Agent or No-RAG Agent
                            agent_response = agent.execute_task(task)
                            
                            # Convert agent response to execution result format
                            result_dict = {
                                "task_id": agent_response.task_id,
                                "success": agent_response.success,
                                "answer": agent_response.answer,
                                "citations": getattr(agent_response, 'citations', []),
                                "reasoning_path": getattr(agent_response, 'reasoning_path', []),
                                "confidence": getattr(agent_response, 'confidence', 1.0),
                                "execution_time": agent_response.execution_time,
                                "tokens_used": agent_response.tokens_used,
                                "model_used": agent_response.model_used,
                                "retries_needed": 0,
                                "error_type": agent_response.error_type,
                                "error_message": agent_response.error_message,
                                "raw_response": agent_response.answer,
                                "task_type": task.web_task_type if hasattr(task, 'web_task_type') else (task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)),
                                "prompt": task.prompt if hasattr(task, 'prompt') else "",
                                "gold_answer": task.gold_answer if hasattr(task, 'gold_answer') and task.gold_answer else "",
                                "safety_check": {
                                    "passed": agent_response.safety_passed,
                                    "issues": agent_response.safety_issues
                                }
                            }
                        
                                                # Create a new result object with the additional fields
                        class ExtendedExecutionResult:
                            def __init__(self, result_dict, task_type, prompt, gold_answer):
                                self.task_id = result_dict["task_id"]
                                self.success = result_dict["success"]
                                self.answer = result_dict["answer"]
                                self.citations = result_dict["citations"]
                                self.reasoning_path = result_dict["reasoning_path"]
                                self.confidence = result_dict["confidence"]
                                self.execution_time = result_dict["execution_time"]
                                self.tokens_used = result_dict["tokens_used"]
                                self.model_used = result_dict["model_used"]
                                self.retries_needed = result_dict["retries_needed"]
                                self.error_type = result_dict["error_type"]
                                self.error_message = result_dict["error_message"]
                                self.raw_response = result_dict["raw_response"]
                                self.task_type = task_type
                                self.prompt = prompt
                                self.gold_answer = gold_answer
                                self.safety_check = result_dict["safety_check"]
                                # Multi-agent specific fields
                                self.multi_agent = result_dict.get("multi_agent", {})
                            
                            def to_dict(self):
                                result = {
                                "task_id": self.task_id,
                                "success": self.success,
                                "answer": self.answer,
                                "citations": self.citations,
                                "reasoning_path": self.reasoning_path,
                                "confidence": self.confidence,
                                "execution_time": self.execution_time,
                                "tokens_used": self.tokens_used,
                                "model_used": self.model_used,
                                "retries_needed": self.retries_needed,
                                "error_type": self.error_type,
                                "error_message": self.error_message,
                                "raw_response": self.raw_response,
                                "task_type": self.task_type,
                                "prompt": self.prompt,
                                "gold_answer": self.gold_answer,
                                "safety_check": self.safety_check
                                }
                                # Add multi-agent fields if present
                                if self.multi_agent:
                                    result["multi_agent"] = self.multi_agent
                                return result
                    
                    result = ExtendedExecutionResult(
                        result_dict,
                        task.web_task_type if hasattr(task, 'web_task_type') else (task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)),
                        task.prompt if hasattr(task, 'prompt') else "",
                        task.gold_answer if hasattr(task, 'gold_answer') and task.gold_answer else ""
                    )
                    
                    # Save individual task result after completion
                    self._save_individual_task_result(result, task, output_dirs["results"])
                    
                    execution_results.append(result)
                    
                    if result.success:
                        successful_executions += 1
                        pbar.set_postfix({"✅": f"{successful_executions}", "❌": f"{failed_executions}"})
                    else:
                        failed_executions += 1
                        pbar.set_postfix({"✅": f"{successful_executions}", "❌": f"{failed_executions}"})
                    
                    pbar.update(1)
                else:
                    # Mock execution for demo
                    from agent_framework.executors import ExecutionResult
                    result = ExecutionResult(
                        task_id=task.task_id,
                        success=True,
                        answer=f"Mock answer for task: {task.prompt[:100] if hasattr(task, 'prompt') else 'Unknown task'}",
                        citations=[f"node_{i}" for i in range(2)],
                        confidence=0.8,
                        execution_time=0.5,
                        model_used=self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini')
                    )
                    
                    # Add task type information to the result
                    result_dict = result.to_dict()
                    result_dict["task_type"] = task.web_task_type if hasattr(task, 'web_task_type') else (task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type))
                    result_dict["prompt"] = task.prompt if hasattr(task, 'prompt') else ""
                    result_dict["gold_answer"] = task.gold_answer if hasattr(task, 'gold_answer') and task.gold_answer else ""
                    
                    # Create a new result object with the additional fields
                    class ExtendedExecutionResult:
                        def __init__(self, base_result, task_type, prompt, gold_answer):
                            self.task_id = base_result.task_id
                            self.success = base_result.success
                            self.answer = base_result.answer
                            self.citations = base_result.citations
                            self.reasoning_path = base_result.reasoning_path
                            self.confidence = base_result.confidence
                            self.execution_time = base_result.execution_time
                            self.tokens_used = base_result.tokens_used
                            self.model_used = base_result.model_used
                            self.retries_needed = base_result.retries_needed
                            self.error_type = base_result.error_type
                            self.error_message = base_result.error_message
                            self.raw_response = base_result.raw_response
                            self.task_type = task_type
                            self.prompt = prompt
                            self.gold_answer = gold_answer
                            self.safety_check = getattr(base_result, 'safety_check', {})
                            # Multi-agent specific fields (empty for mock execution)
                            self.multi_agent = {}
                        
                        def to_dict(self):
                            result = {
                                "task_id": self.task_id,
                                "success": self.success,
                                "answer": self.answer,
                                "citations": self.citations,
                                "reasoning_path": self.reasoning_path,
                                "confidence": self.confidence,
                                "execution_time": self.execution_time,
                                "tokens_used": self.tokens_used,
                                "model_used": self.model_used,
                                "retries_needed": self.retries_needed,
                                "error_type": self.error_type,
                                "error_message": self.error_message,
                                "raw_response": self.raw_response,
                                "task_type": self.task_type,
                                "prompt": self.prompt,
                                "gold_answer": self.gold_answer,
                                "safety_check": self.safety_check
                            }
                            # Add multi-agent fields if present
                            if self.multi_agent:
                                result["multi_agent"] = self.multi_agent
                            return result
                    
                    result = ExtendedExecutionResult(
                        result,
                        task.web_task_type if hasattr(task, 'web_task_type') else (task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)),
                        task.prompt if hasattr(task, 'prompt') else "",
                        task.gold_answer if hasattr(task, 'gold_answer') and task.gold_answer else ""
                    )
                    
                    # Save individual task result after completion
                    self._save_individual_task_result(result, task, output_dirs["results"])
                    
                    execution_results.append(result)
                    successful_executions += 1
                    pbar.set_postfix({"✅": f"{successful_executions}", "❌": f"{failed_executions}"})
                    pbar.update(1)
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute task {task.task_id}: {e}")
                failed_executions += 1
                pbar.set_postfix({"✅": f"{successful_executions}", "❌": f"{failed_executions}"})
                pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Calculate token usage statistics
        total_tokens = sum(r.tokens_used for r in execution_results)
        avg_tokens = total_tokens / len(execution_results) if execution_results else 0
        max_tokens = max(r.tokens_used for r in execution_results) if execution_results else 0
        min_tokens = min(r.tokens_used for r in execution_results) if execution_results else 0
        
        # Print execution results summary
        logger.info(f"{'='*60}")
        logger.info("🏃‍♂️ EXECUTION RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful executions: {successful_executions}")
        logger.info(f"❌ Failed executions: {failed_executions}")
        logger.info(f"📊 Success rate: {successful_executions/len(execution_results)*100:.1f}%" if execution_results else "📊 Success rate: 0.0%")
        logger.info(f"⏱️  Average execution time: {sum(r.execution_time for r in execution_results)/len(execution_results):.2f}s" if execution_results else "⏱️  Average execution time: 0.00s")
        logger.info(f"🔤 Total tokens used: {total_tokens:,}")
        logger.info(f"🔤 Average tokens per task: {avg_tokens:.1f}")
        logger.info(f"🔤 Token range: {min_tokens} - {max_tokens}")
        logger.info(f"⏱️  Total processing time: {time.time() - stage_start:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Use existing output directories
        output_dirs = self.output_dirs
        output_dir = output_dirs["results"]
        
        # Save execution results
        results_file = output_dir / "execution_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "tasks_executed": len(execution_results),
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": successful_executions / len(execution_results) if execution_results else 0,
                "avg_execution_time": sum(r.execution_time for r in execution_results) / len(execution_results) if execution_results else 0,
                "total_tokens_used": total_tokens,
                "avg_tokens_per_task": avg_tokens,
                "max_tokens_per_task": max_tokens,
                "min_tokens_per_task": min_tokens,
                "processing_time": time.time() - stage_start,
                "results": [r.to_dict() for r in execution_results]
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Execution results saved to: {output_dir}")
        
        return {
            "tasks_executed": len(execution_results),
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / len(execution_results) if execution_results else 0,
            "avg_execution_time": sum(r.execution_time for r in execution_results) / len(execution_results) if execution_results else 0,
            "total_tokens_used": total_tokens,
            "avg_tokens_per_task": avg_tokens,
            "max_tokens_per_task": max_tokens,
            "min_tokens_per_task": min_tokens,
            "processing_time": time.time() - stage_start,
            "results": [r.to_dict() for r in execution_results]
        }
    
    def _run_evaluation_stage(self, stage_results: Dict[str, Any], tasks: List[Any]) -> Dict[str, Any]:
        """Run evaluation stage"""
        logger.info(f"{'='*60}")
        logger.info("📊 EVALUATION STAGE")
        logger.info(f"{'='*60}")
        
        stage_start = time.time()
        
        # In evaluation mode, stage_results is directly the task execution results
        execution_results = stage_results.get("results", [])
        
        # Load all task results from individual_results directory to ensure complete evaluation
        all_task_results = self._load_all_task_results()
        
        if not execution_results and not all_task_results:
            logger.error("❌ No execution results found for evaluation")
            return {"evaluations_completed": 0, "processing_time": 0}
        
        # Use all task results for evaluation (both new and previously completed)
        if all_task_results:
            logger.info(f"📊 Processing {len(all_task_results)} total task results for evaluation")
            execution_results = all_task_results
        else:
            logger.info(f"📊 Processing {len(execution_results)} execution results for evaluation")
        
        # For evaluation-only mode, we don't have task generation metrics
        # Use default values or calculate from tasks
        task_node = len(tasks) if tasks else 0
        task_node_expand = len(tasks) if tasks else 0
        
        # Evaluation configuration (now using default configuration)
        evaluation_config = {}
        
        # Initialize detailed metrics tracking
        task_metrics = []
        safety_metrics = {
            "safety_compliance_rate": 0.0,
            "safety_violations": 0,
            "safety_tasks_processed": 0,
            "safety_level_distribution": {},
            "violation_types": {},
            "safety_task_types": {}
        }
        
        safety_tasks_processed = 0
        safety_violations = 0
        total_tasks = len(execution_results)
        
        # Create progress bar for evaluation
        eval_pbar = tqdm(
            total=len(execution_results),
            desc="📊 Evaluating Results",
            unit="result",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
        
        # Process each task result with enhanced evaluation
        task_metrics = []
        
        # Import required modules
        from agent_framework.evaluators import MultiDimensionalEvaluator, LLMBasedSafetyEvaluator
        from task_craft.task_generator import TaskInstance
        from agent_framework.executors import ExecutionResult
        import concurrent.futures
        import threading
        
        # Configuration for concurrent evaluation (global settings)
        evaluation_config = self.config.agent.get('evaluation', {})
        enable_concurrent = evaluation_config.get('enable_concurrent', True)
        max_workers = min(
            evaluation_config.get('max_workers', 8), 
            len(execution_results)
        )  # Limit to configured workers or number of results
        
        if enable_concurrent and len(execution_results) > 1:
            logger.info(f"🚀 Starting concurrent evaluation with {max_workers} workers for {len(execution_results)} tasks")
        else:
            logger.info(f"🔄 Starting sequential evaluation for {len(execution_results)} tasks")
        
        def evaluate_single_task(args):
            """Evaluate a single task - designed for concurrent execution"""
            i, result = args
            try:
                # Handle both TaskResult objects and dict results
                if hasattr(result, 'task_id'):
                    # TaskResult object
                    task_id = result.task_id or f"task_{i}"
                    
                    # Determine task_type based on agent type
                    agent_mode = self.config.agent.get('agent_mode', 'single')
                    if agent_mode == 'web':
                        # For web agents, prioritize web_task_type
                        task_type = getattr(result, 'web_task_type', getattr(result, 'task_type', 'unknown'))
                    else:
                        # For text agents, use task_type
                        task_type = getattr(result, 'task_type', getattr(result, 'web_task_type', 'unknown'))
                    
                    prompt = getattr(result, 'prompt', '')
                    response = getattr(result, 'answer', '')
                    gold_answer = getattr(result, 'gold_answer', '')
                    execution_time = getattr(result, 'execution_time', 0.0)
                    multi_agent_data = getattr(result, 'multi_agent', {})
                    is_multi_agent = bool(multi_agent_data)
                else:
                    # Dict result
                    task_id = result.get("task_id", f"task_{i}")
                    
                    # Determine task_type based on agent type
                    agent_mode = self.config.agent.get('agent_mode', 'single')
                    if agent_mode == 'web':
                        # For web agents, prioritize web_task_type
                        task_type = result.get("web_task_type", result.get("task_type", "unknown"))
                        logger.debug(f"🔍 Web agent: task_type={task_type}, web_task_type={result.get('web_task_type')}, task_type_field={result.get('task_type')}")
                    else:
                        # For text agents, use task_type
                        task_type = result.get("task_type", result.get("web_task_type", "unknown"))
                        logger.debug(f"🔍 Text agent: task_type={task_type}, task_type_field={result.get('task_type')}, web_task_type={result.get('web_task_type')}")
                    
                    prompt = result.get("prompt", "")
                    response = result.get("answer", "")  # ExecutionResult uses 'answer' field
                    gold_answer = result.get("gold_answer", "")
                    execution_time = result.get("execution_time", 0.0)
                    
                    # Check if this is a multi-agent result
                    multi_agent_data = result.get("multi_agent", {})
                    is_multi_agent = bool(multi_agent_data)
                
                # Determine agent type for this task
                agent_type = "multi_agent" if is_multi_agent else "rag_agent"
                
                # Use real evaluator for enhanced metrics
                evaluator = MultiDimensionalEvaluator()
                
                # Create mock task and execution result for evaluation
                task = TaskInstance(
                    task_id=task_id,
                    template_id="mock_template",
                    task_type=TaskType.COMPREHENSION,
                    difficulty="medium",
                    prompt=prompt,
                    gold_answer=gold_answer
                )
                
                # Create execution result for evaluation
                if hasattr(result, 'success'):
                    # TaskResult object
                    exec_result = ExecutionResult(
                        task_id=task_id,
                        success=result.success,
                        answer=response,
                        citations=getattr(result, 'citations', []),
                        reasoning_path=getattr(result, 'reasoning_path', []),
                        execution_time=execution_time,
                        tokens_used=getattr(result, 'tokens_used', 0),
                        model_used=getattr(result, 'model_used', 'unknown'),
                        retries_needed=getattr(result, 'retries_needed', 0)
                    )
                else:
                    # Dict result
                    exec_result = ExecutionResult(
                        task_id=task_id,
                        success=result.get("success", False),
                        answer=response,
                        citations=result.get("citations", []),
                        reasoning_path=result.get("reasoning_path", []),
                        execution_time=execution_time,
                        tokens_used=result.get("tokens_used", 0),
                        model_used=result.get("model_used", "unknown"),
                        retries_needed=result.get("retries_needed", 0)
                    )
                
                # Perform evaluation with GraphRAG metrics
                eval_result = evaluator.evaluate(task, exec_result)
                
                # Determine if this is a safety task based on agent_mode
                is_safety_task = self._is_safety_task(task_type)
                
                # Special LLM-based safety evaluation for safety tasks
                safety_eval_data = {}
                safety_violations = 0
                if is_safety_task:
                    # Get safety evaluation configuration from policy_extraction
                    safety_config = self.config.safety
                    policy_extraction_config = safety_config.get('safety_policy', {}).get('policy_extraction', {})
                    
                    safety_evaluator = LLMBasedSafetyEvaluator(config=policy_extraction_config)
                    safety_eval = safety_evaluator.evaluate(task, exec_result)
                    
                    # Store safety evaluation data
                    safety_eval_data = safety_eval.details.get('safety', {})
                
                # Get token usage information
                if hasattr(result, 'tokens_used'):
                    tokens_used = result.tokens_used
                    model_used = getattr(result, 'model_used', 'unknown')
                else:
                    tokens_used = result.get("tokens_used", 0)
                    model_used = result.get("model_used", "unknown")
                
                # If tokens_used is 0, try to extract from multi_agent data (fallback for old data)
                if tokens_used == 0 and multi_agent_data:
                    tokens_used = self._extract_total_tokens_from_multi_agent(multi_agent_data)
                
                
                # Create task metric
                web_task_type_value = result.get("web_task_type") if isinstance(result, dict) else getattr(result, 'web_task_type', None)
                logger.debug(f"🔍 Creating task_metric: task_type={task_type}, web_task_type_value={web_task_type_value}, result_type={type(result)}")
                
                task_metric = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "web_task_type": web_task_type_value,
                    "agent_type": agent_type,
                    "is_safety_task": is_safety_task,
                    "prompt": prompt,
                    "response": response,
                    "gold_answer": gold_answer,
                    "success": exec_result.success,  # Add success field from execution result
                    "execution_time": execution_time,
                    "tokens_used": tokens_used,
                    "model_used": model_used,
                    "overall_score": eval_result.overall_score,
                    "f1_score": eval_result.f1_score,
                    "exact_match": eval_result.exact_match,
                    "safety_evaluation": safety_eval_data,
                    "is_multi_agent": is_multi_agent,
                    # Add missing GraphRAG metrics
                    "pass_rate": eval_result.pass_rate,
                    "task_node": eval_result.task_node,
                    "task_node_expand": eval_result.task_node_expand,
                    "sampling_time": eval_result.sampling_time,
                    "rouge_l": eval_result.rouge_l,
                    "answer_quality": eval_result.answer_quality,
                    "relevance": eval_result.relevance,
                    "completeness": eval_result.completeness,
                    "citation_f1": eval_result.citation_f1,
                    "reasoning_quality": eval_result.reasoning_path_match,
                    "response_time": execution_time,
                    # Extract multi-agent metrics from actual response
                    "planner_steps": self._extract_planner_steps(multi_agent_data) if is_multi_agent else 0,
                    "retriever_nodes": self._extract_retriever_nodes(multi_agent_data) if is_multi_agent else 0,
                    "reasoner_steps": self._extract_reasoner_steps(multi_agent_data) if is_multi_agent else 0,
                    "verifier_confidence": self._extract_verifier_confidence(multi_agent_data) if is_multi_agent else 0.0,
                    "summarizer_confidence": self._extract_summarizer_confidence(multi_agent_data) if is_multi_agent else 0.0
                }
                
                return {
                    "task_metric": task_metric,
                    "is_safety_task": is_safety_task,
                    "safety_violations": safety_violations,
                    "tokens_used": tokens_used
                }
                
            except Exception as e:
                logger.error(f"Error evaluating task {i}: {e}")
                return {
                    "task_metric": None,
                    "is_safety_task": False,
                    "safety_violations": 0,
                    "tokens_used": 0,
                    "error": str(e)
                }
        
        # Prepare arguments for execution
        task_args = [(i, result) for i, result in enumerate(execution_results)]
        
        if enable_concurrent and len(execution_results) > 1:
            # Execute evaluation concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {executor.submit(evaluate_single_task, args): i for i, args in enumerate(task_args)}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        result_data = future.result()
                        
                        if result_data.get("task_metric"):
                            task_metrics.append(result_data["task_metric"])
                            
                            # Update counters
                            if result_data.get("is_safety_task"):
                                safety_tasks_processed += 1
                            safety_violations += result_data.get("safety_violations", 0)
                            
                            # Update progress bar
                            eval_pbar.set_postfix({
                                "Processed": f"{len(task_metrics)}",
                                "Safety": f"{safety_tasks_processed}",
                                "Violations": f"{safety_violations}"
                            })
                            eval_pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error in concurrent evaluation for task {i}: {e}")
                        eval_pbar.update(1)
        else:
            # Execute evaluation sequentially
            for i, args in enumerate(task_args):
                try:
                    result_data = evaluate_single_task(args)
                    
                    if result_data.get("task_metric"):
                        task_metrics.append(result_data["task_metric"])
                        
                        # Update counters
                        if result_data.get("is_safety_task"):
                            safety_tasks_processed += 1
                        safety_violations += result_data.get("safety_violations", 0)
                        
                        # Update progress bar
                        eval_pbar.set_postfix({
                            "Processed": f"{len(task_metrics)}",
                            "Safety": f"{safety_tasks_processed}",
                            "Violations": f"{safety_violations}"
                        })
                        eval_pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error in sequential evaluation for task {i}: {e}")
                    eval_pbar.update(1)
        
        # Close the progress bar
        eval_pbar.close()
        
        # Calculate overall safety metrics
        if safety_tasks_processed > 0:
            safety_metrics["safety_compliance_rate"] = (safety_tasks_processed - safety_violations) / safety_tasks_processed
            safety_metrics["safety_violations"] = safety_violations
            safety_metrics["safety_tasks_processed"] = safety_tasks_processed
            
            # Calculate average LLM safety evaluation scores
            safety_tasks = [tm for tm in task_metrics if tm.get("is_safety_task", False)]
            if safety_tasks:
                llm_scores_sum = {
                    'policy_compliance': 0.0,
                    'safety_awareness': 0.0,
                    'risk_identification': 0.0,
                    'appropriate_response': 0.0,
                    'bias_detection': 0.0,
                    'harmful_content': 0.0,
                    'overall_safety_score': 0.0
                }
                
                for task in safety_tasks:
                    safety_eval = task.get("safety_evaluation", {})
                    if safety_eval:
                        llm_scores = safety_eval.get('llm_safety_scores', {})
                        for key in llm_scores_sum.keys():
                            if key == 'overall_safety_score':
                                llm_scores_sum[key] += safety_eval.get('overall_safety_score', 0.0)
                            else:
                                llm_scores_sum[key] += llm_scores.get(key, 0.0)
                
                # Calculate averages
                safety_metrics["llm_safety_scores"] = {
                    key: value / max(1, len(safety_tasks)) 
                    for key, value in llm_scores_sum.items()
                }
        
        # Close evaluation progress bar
        eval_pbar.close()
        
        # Calculate overall metrics with GraphRAG-focused evaluation
        overall_metrics = {
            "total_tasks": total_tasks,
            "safety_tasks": safety_tasks_processed,
            "normal_tasks": total_tasks - safety_tasks_processed,
            # Agent type distribution
            "agent_type_distribution": {},
            # Token usage metrics
            "total_tokens_used": sum(tm["tokens_used"] for tm in task_metrics),
            "avg_tokens_per_task": sum(tm["tokens_used"] for tm in task_metrics) / max(1, total_tasks),
            "max_tokens_per_task": max(tm["tokens_used"] for tm in task_metrics) if task_metrics else 0,
            "min_tokens_per_task": min(tm["tokens_used"] for tm in task_metrics) if task_metrics else 0,
            # GraphRAG-specific metrics averages
            "pass_rate": sum(tm["pass_rate"] for tm in task_metrics) / max(1, total_tasks),
            "task_node": sum(tm["task_node"] for tm in task_metrics) / max(1, total_tasks),
            "task_node_expand": sum(tm["task_node_expand"] for tm in task_metrics) / max(1, total_tasks),
            "avg_sampling_time": sum(tm["sampling_time"] for tm in task_metrics) / max(1, total_tasks),
            # Rule-based metrics averages
            "exact_match": sum(tm["exact_match"] for tm in task_metrics) / max(1, total_tasks),
            "f1_score": sum(tm["f1_score"] for tm in task_metrics) / max(1, total_tasks),
            "rouge_l": sum(tm["rouge_l"] for tm in task_metrics) / max(1, total_tasks),
            # LLM-based metrics averages
            "answer_quality": sum(tm["answer_quality"] for tm in task_metrics) / max(1, total_tasks),
            "relevance": sum(tm["relevance"] for tm in task_metrics) / max(1, total_tasks),
            "completeness": sum(tm["completeness"] for tm in task_metrics) / max(1, total_tasks),
            # Other metrics
            "citation_f1": sum(tm["citation_f1"] for tm in task_metrics) / max(1, total_tasks),
            "reasoning_quality": sum(tm["reasoning_quality"] for tm in task_metrics) / max(1, total_tasks),
            "avg_response_time": sum(tm["response_time"] for tm in task_metrics) / max(1, total_tasks),
            "safety_compliance_rate": (safety_tasks_processed - safety_violations) / max(1, safety_tasks_processed),
            "safety_violations": safety_violations,
            # Multi-agent specific metrics (only calculate if there are multi-agent tasks)
            "multi_agent_tasks": sum(1 for tm in task_metrics if tm["is_multi_agent"]),
            "avg_planner_steps": sum(tm["planner_steps"] for tm in task_metrics if tm["is_multi_agent"]) / max(1, sum(1 for tm in task_metrics if tm["is_multi_agent"])),
            "avg_retriever_nodes": sum(tm["retriever_nodes"] for tm in task_metrics if tm["is_multi_agent"]) / max(1, sum(1 for tm in task_metrics if tm["is_multi_agent"])),
            "avg_reasoner_steps": sum(tm["reasoner_steps"] for tm in task_metrics if tm["is_multi_agent"]) / max(1, sum(1 for tm in task_metrics if tm["is_multi_agent"])),
            "avg_verifier_confidence": sum(tm["verifier_confidence"] for tm in task_metrics if tm["is_multi_agent"]) / max(1, sum(1 for tm in task_metrics if tm["is_multi_agent"])),
            "avg_summarizer_confidence": sum(tm["summarizer_confidence"] for tm in task_metrics if tm["is_multi_agent"]) / max(1, sum(1 for tm in task_metrics if tm["is_multi_agent"]))
        }
        
        # Add web agent specific metrics if this is a web agent evaluation
        agent_mode = self.config.agent.get('agent_mode', 'single')
        if agent_mode == 'web':
            # Calculate web-specific metrics
            web_tasks = [tm for tm in task_metrics if tm.get("task_type") in ["Search", "Form Filling", "Navigation", "Data Extraction", "E-commerce", "Content Browsing", "search", "form_filling", "navigation", "data_extraction", "ecommerce", "content_browsing"]]
            
            if web_tasks:
                # Task completion rate (based on pass_rate)
                overall_metrics["task_completion_rate"] = sum(tm["pass_rate"] for tm in web_tasks) / len(web_tasks)
                
                # Navigation accuracy (for Navigation tasks)
                navigation_tasks = [tm for tm in web_tasks if tm.get("task_type") in ["Navigation", "navigation"]]
                if navigation_tasks:
                    overall_metrics["navigation_accuracy"] = sum(tm["pass_rate"] for tm in navigation_tasks) / len(navigation_tasks)
                else:
                    overall_metrics["navigation_accuracy"] = 0.0
                
                # Form filling accuracy (for Form Filling tasks)
                form_tasks = [tm for tm in web_tasks if tm.get("task_type") in ["Form Filling", "form_filling"]]
                if form_tasks:
                    overall_metrics["form_filling_accuracy"] = sum(tm["pass_rate"] for tm in form_tasks) / len(form_tasks)
                else:
                    overall_metrics["form_filling_accuracy"] = 0.0
                
                # Search accuracy (for Search tasks)
                search_tasks = [tm for tm in web_tasks if tm.get("task_type") in ["Search", "search"]]
                if search_tasks:
                    overall_metrics["search_accuracy"] = sum(tm["pass_rate"] for tm in search_tasks) / len(search_tasks)
                else:
                    overall_metrics["search_accuracy"] = 0.0
                
                # Data extraction accuracy (for Data Extraction tasks)
                extraction_tasks = [tm for tm in web_tasks if tm.get("task_type") in ["Data Extraction", "data_extraction"]]
                if extraction_tasks:
                    overall_metrics["data_extraction_accuracy"] = sum(tm["pass_rate"] for tm in extraction_tasks) / len(extraction_tasks)
                else:
                    overall_metrics["data_extraction_accuracy"] = 0.0
            else:
                # Default values if no web tasks
                overall_metrics["task_completion_rate"] = 0.0
                overall_metrics["navigation_accuracy"] = 0.0
                overall_metrics["form_filling_accuracy"] = 0.0
                overall_metrics["search_accuracy"] = 0.0
                overall_metrics["data_extraction_accuracy"] = 0.0
        
        # Calculate agent type distribution
        agent_type_counts = {}
        for tm in task_metrics:
            agent_type = tm.get("agent_type", "unknown")
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1
        overall_metrics["agent_type_distribution"] = agent_type_counts
        
        # Print evaluation summary
        logger.info(f"{'='*60}")
        logger.info("📊 EVALUATION RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total tasks evaluated: {total_tasks}")
        logger.info(f"🔒 Safety tasks: {safety_tasks_processed}")
        logger.info(f"🔒 Safety violations: {safety_violations}")
        if safety_tasks_processed > 0:
            logger.info(f"🛡️  Safety compliance rate: {safety_metrics['safety_compliance_rate']:.1%}")
        logger.info(f"🔤 Total tokens used: {overall_metrics['total_tokens_used']:,}")
        logger.info(f"🔤 Average tokens per task: {overall_metrics['avg_tokens_per_task']:.1f}")
        logger.info(f"🔤 Token range: {overall_metrics['min_tokens_per_task']} - {overall_metrics['max_tokens_per_task']}")
        logger.info(f"⏱️  Total evaluation time: {time.time() - stage_start:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Use existing output directories
        output_dirs = self.output_dirs
        output_dir = output_dirs["results"]
        logger.info(f"📁 Using unified evaluation output directory: {output_dir}")
        
        # Save evaluation results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluations_completed": total_tasks,
                "metrics": overall_metrics,
                "safety_metrics": safety_metrics,
                "task_metrics": task_metrics,
                "processing_time": time.time() - stage_start
            }, f, indent=2, ensure_ascii=False)
        
        # Save detailed metrics to CSV files
        self._save_evaluation_csvs(output_dir, task_metrics, safety_metrics, overall_metrics, evaluation_config)
        
        logger.info(f"💾 Evaluation results saved to: {output_dir}")
        
        # Add web task type information to evaluation results
        web_task_types = {}
        success_by_type = {}
        
        # Extract web task type information from task_metrics
        for tm in task_metrics:
            task_type = tm.get("task_type", "unknown")
            web_task_types[task_type] = web_task_types.get(task_type, 0) + 1
            
            if task_type not in success_by_type:
                success_by_type[task_type] = {'total': 0, 'successful': 0}
            success_by_type[task_type]['total'] += 1
            
            # Check if task was successful based on pass_rate
            pass_rate = tm.get("pass_rate", 0.0)
            if pass_rate > 0.5:  # Consider successful if pass_rate > 50%
                success_by_type[task_type]['successful'] += 1
        
        return {
            "evaluations_completed": total_tasks,
            "metrics": overall_metrics,
            "safety_metrics": safety_metrics,
            "task_metrics": task_metrics,
            "processing_time": time.time() - stage_start,
            "web_task_types": web_task_types,
            "success_by_type": success_by_type
        }
    
    def _extract_planner_steps(self, multi_agent_data: Dict[str, Any]) -> int:
        """Extract number of planner steps from multi-agent data"""
        planner_output = multi_agent_data.get("planner_output", {})
        plan = planner_output.get("plan", {})
        reasoning_steps = plan.get("reasoning_steps", [])
        return len(reasoning_steps)
    
    def _extract_retriever_nodes(self, multi_agent_data: Dict[str, Any]) -> int:
        """Extract number of retriever nodes from multi-agent data"""
        retriever_output = multi_agent_data.get("retriever_output", {})
        nodes = retriever_output.get("nodes", [])
        return len(nodes)
    
    def _extract_reasoner_steps(self, multi_agent_data: Dict[str, Any]) -> int:
        """Extract number of reasoner steps from multi-agent data"""
        reasoner_output = multi_agent_data.get("reasoner_output", [])
        return len(reasoner_output)
    
    def _extract_verifier_confidence(self, multi_agent_data: Dict[str, Any]) -> float:
        """Extract verifier confidence from multi-agent data"""
        verifier_output = multi_agent_data.get("verifier_output", {})
        return verifier_output.get("confidence", 0.0)
    
    def _extract_summarizer_confidence(self, multi_agent_data: Dict[str, Any]) -> float:
        """Extract summarizer confidence from multi-agent data"""
        summarizer_output = multi_agent_data.get("summarizer_output", {})
        return summarizer_output.get("confidence", 0.0)
    
    def _extract_total_tokens_from_multi_agent(self, multi_agent_data: Dict[str, Any]) -> int:
        """Extract total tokens from multi-agent data"""
        # First try to get total_tokens directly from multi_agent data
        total_tokens = multi_agent_data.get("total_tokens", 0)
        if total_tokens > 0:
            return total_tokens
        
        # If not available, sum up tokens from individual agent outputs
        tokens = 0
        
        # Planner tokens
        planner_output = multi_agent_data.get("planner_output", {})
        if "tokens_used" in planner_output:
            tokens += planner_output["tokens_used"]
        
        # Retriever tokens (usually 0 for retrieval)
        retriever_output = multi_agent_data.get("retriever_output", {})
        if isinstance(retriever_output, dict) and "tokens_used" in retriever_output:
            tokens += retriever_output["tokens_used"]
        
        # Reasoner tokens (sum from all reasoning steps)
        reasoner_output = multi_agent_data.get("reasoner_output", [])
        if isinstance(reasoner_output, list):
            for step in reasoner_output:
                if isinstance(step, dict) and "tokens_used" in step:
                    tokens += step["tokens_used"]
        
        # Verifier tokens
        verifier_output = multi_agent_data.get("verifier_output", {})
        if isinstance(verifier_output, dict) and "tokens_used" in verifier_output:
            tokens += verifier_output["tokens_used"]
        
        # Summarizer tokens
        summarizer_output = multi_agent_data.get("summarizer_output", {})
        if isinstance(summarizer_output, dict) and "tokens_used" in summarizer_output:
            tokens += summarizer_output["tokens_used"]
        
        return tokens
    
    def _save_evaluation_csvs(self, evaluation_dir: Path, task_metrics: List[Dict], safety_metrics: Dict, overall_metrics: Dict, evaluation_config: Dict):
        """Save evaluation results to CSV files - only generate full_res.csv and summary.csv"""
        
        import csv
        import json
        
        # 1. Save full_res.csv - metrics performance for each task
        full_res_file = evaluation_dir / "full_res.csv"
        with open(full_res_file, 'w', newline='', encoding='utf-8') as f:
            if task_metrics:
                writer = csv.DictWriter(f, fieldnames=task_metrics[0].keys())
                writer.writeheader()
                writer.writerows(task_metrics)
        
        # 2. Save summary.csv - overall metrics performance
        summary_file = evaluation_dir / "summary.csv"
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            
            # Write overall metrics
            for metric, value in overall_metrics.items():
                if isinstance(value, float):
                    writer.writerow([metric, f"{value:.4f}"])
                else:
                    writer.writerow([metric, value])
            
            # Write safety metrics (including LLM safety evaluation scores)
            if safety_metrics:
                for metric, value in safety_metrics.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries like llm_safety_scores
                        for sub_metric, sub_value in value.items():
                            if isinstance(sub_value, float):
                                writer.writerow([f"{metric}_{sub_metric}", f"{sub_value:.4f}"])
                            else:
                                writer.writerow([f"{metric}_{sub_metric}", sub_value])
                    elif isinstance(value, float):
                        writer.writerow([metric, f"{value:.4f}"])
                    else:
                        writer.writerow([metric, value])
        
        logger.info(f"📄 CSV files saved:")
        logger.info(f"   - full_res.csv: {len(task_metrics)} individual task metrics")
        logger.info(f"   - summary.csv: Overall metrics summary")
    
    # Dataset creation is now integrated into task generation stage
    # This method is no longer needed
    
    def _count_task_types(self, tasks) -> Dict[str, int]:
        """Count tasks by type"""
        counts = {}
        for task in tasks:
            # Handle both web tasks and regular tasks
            if hasattr(task, 'web_task_type'):
                task_type = task.web_task_type
            else:
                task_type = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
            counts[task_type] = counts.get(task_type, 0) + 1
        return counts
    
    def _count_difficulties(self, tasks) -> Dict[str, int]:
        """Count tasks by difficulty"""
        counts = {}
        for task in tasks:
            difficulty = task.difficulty.value if hasattr(task.difficulty, 'value') else str(task.difficulty)
            counts[difficulty] = counts.get(difficulty, 0) + 1
        return counts
    
    def get_task_statistics(self, tasks: List[Any]) -> Dict[str, Any]:
        """Get detailed statistics about task distribution"""
        
        # Classify tasks into normal and safety categories
        normal_tasks = []
        safety_tasks = []
        
        for task in tasks:
            # Handle both web tasks and regular tasks
            if hasattr(task, 'web_task_type'):
                # Web task - check web_task_type
                if task.web_task_type == "safety" or task.web_task_type.startswith('web_safety_'):
                    safety_tasks.append(task)
                else:
                    normal_tasks.append(task)
            elif hasattr(task, 'task_type') and hasattr(task.task_type, 'is_safety_task'):
                # Regular task - check task_type
                if task.task_type.is_safety_task(task.task_type):
                    safety_tasks.append(task)
                else:
                    normal_tasks.append(task)
            else:
                # Fallback: assume normal task if we can't determine
                normal_tasks.append(task)
        
        # Count by task type
        task_type_counts = {}
        safety_type_counts = {}
        normal_type_counts = {}
        
        for task in tasks:
            # Handle both web tasks and regular tasks
            if hasattr(task, 'web_task_type'):
                task_type = task.web_task_type
            else:
                task_type = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            if hasattr(task, 'web_task_type'):
                # Web task - check web_task_type
                if task.web_task_type == "safety" or task.web_task_type.startswith('web_safety_'):
                    safety_type_counts[task_type] = safety_type_counts.get(task_type, 0) + 1
                else:
                    normal_type_counts[task_type] = normal_type_counts.get(task_type, 0) + 1
            elif hasattr(task, 'task_type') and hasattr(task.task_type, 'is_safety_task'):
                # Regular task - check task_type
                if task.task_type.is_safety_task(task.task_type):
                    safety_type_counts[task_type] = safety_type_counts.get(task_type, 0) + 1
                else:
                    normal_type_counts[task_type] = normal_type_counts.get(task_type, 0) + 1
            else:
                # Fallback: assume normal task
                normal_type_counts[task_type] = normal_type_counts.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(tasks),
            "normal_tasks": len(normal_tasks),
            "safety_tasks": len(safety_tasks),
            "task_type_distribution": task_type_counts,
            "safety_task_types": safety_type_counts,
            "normal_task_types": normal_type_counts,
            "safety_task_ratio": len(safety_tasks) / len(tasks) if tasks else 0
        }
    
    def _save_results(self):
        """Save benchmark results"""
        # Read orchestration configuration from main_config.yaml
        main_config_path = Path(self.config_dir) / "main_config.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r', encoding='utf-8') as f:
                import yaml
                main_config_data = yaml.safe_load(f)
                orchestration_config = main_config_data.get('orchestration', {})
        else:
            orchestration_config = {}
            
        if not orchestration_config.get('monitoring', {}).get('save_benchmark_results', True):
            return
        
        # Use existing output directories
        output_dirs = self.output_dirs
        results_dir = output_dirs["results"]
        
        results_file = results_dir / "benchmark_results.json"
        
        # Create a serializable version of results
        serializable_results = self._make_serializable(self.results)
        
        # Custom JSON encoder to handle numpy types and Path objects
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logger.info(f"💾 Benchmark results saved to {results_file}")
    
    def _make_serializable(self, obj) -> Any:
        """Make object JSON serializable"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in list(obj.items())}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in list(obj.__dict__.items()) if not k.startswith('_')}
        elif hasattr(obj, 'value'):  # Handle enums (like SamplingStrategy, NodeType, EdgeType)
            return obj.value
        else:
            # Handle numpy types
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item'):
                try:
                    return obj.item()  # Convert numpy scalars to Python types
                except:
                    pass
            try:
                json.dumps(obj)  # Test if serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def print_summary(self, results: Optional[Dict[str, Any]] = None):
        """Print benchmark summary"""
        if results is None:
            results = self.results
            
        if not results:
            logger.warning("No results to summarize")
            return
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Status: {'✅ SUCCESS' if results.get('success') else '❌ FAILED'}")
        logger.info(f"Total Time: {results.get('total_time', 0):.2f}s")

        # Determine result type and extract configuration
        config = results.get('config', {})

        # For batch evaluation results, config might be empty
        if not config and 'summary' in results:
            # This is a batch evaluation result
            logger.info(f"Mode: Batch Evaluation")
            summary = results.get('summary', {})
            logger.info(f"Datasets Processed: {summary.get('total_datasets', 0)}")
            logger.info(f"Successful: {summary.get('successful_evaluations', 0)}")
            logger.info(f"Failed: {summary.get('failed_evaluations', 0)}")
            logger.info(f"Success Rate: {summary.get('success_rate', 0):.1%}")

            # Display aggregated task-level metrics
            aggregated = results.get('aggregated_metrics', {})
            if aggregated:
                logger.info(f"\n📊 AGGREGATED TASK METRICS:")
                logger.info(f"   Total Tasks: {aggregated.get('total_tasks', 0)}")
                logger.info(f"   Total Correct: {aggregated.get('total_correct', 0)}")
                logger.info(f"   Overall Accuracy: {aggregated.get('total_accuracy', 0):.3f}")
                logger.info(f"   Average Dataset Accuracy: {aggregated.get('average_accuracy', 0):.3f}")

                # Show task type distribution if available
                task_types = aggregated.get('task_type_distribution', {})
                if task_types:
                    logger.info(f"   Task Type Distribution:")
                    for task_type, count in sorted(task_types.items()):
                        logger.info(f"     {task_type}: {count}")

                # Show difficulty distribution if available
                difficulties = aggregated.get('difficulty_distribution', {})
                if difficulties:
                    logger.info(f"   Difficulty Distribution:")
                    for difficulty, count in sorted(difficulties.items()):
                        logger.info(f"     {difficulty}: {count}")
            else:
                logger.info(f"   No aggregated metrics available")

            return
        
        # Agent configuration details
        agent_mode = config.get('agent_mode', 'single')
        agent_type = config.get('agent_type', 'rag')
        model_name = config.get('model_name', 'N/A')

        logger.info(f"Model: {model_name}")
        logger.info(f"Agent Mode: {agent_mode}")
        
        if agent_mode == 'multi':
            logger.info(f"Multi-Agent System: Enabled")
        elif agent_mode == 'web':
            logger.info(f"Web Agent Mode: Enabled")
            if agent_type == 'agent_s':
                logger.info(f"Agent Type: Agent S 2.5 (Web Agent)")
            elif agent_type == 'som_agent':
                logger.info(f"Agent Type: SoM Agent (Web Agent)")
            else:
                logger.info(f"Agent Type: Standard Web Agent")
        else:
            logger.info(f"Single Agent Configuration")
            logger.info(f"Agent Type: {agent_type}")
            if agent_type == 'rag':
                logger.info(f"RAG Mode: Enabled")
            elif agent_type == 'no_rag':
                logger.info(f"RAG Mode: Disabled (No-RAG)")
            else:
                logger.info(f"Agent Type: {agent_type}")
        
        # Check if this is dataset generation or evaluation
        if 'dataset_path' in config:
            logger.info(f"Dataset: {config.get('dataset_path', 'N/A')}")
            if config.get('graph_path'):
                logger.info(f"Graph: {config['graph_path']}")
            logger.info(f"Mode: Agent Evaluation")
        else:
            logger.info(f"Mode: Dataset Generation")
        
        stages = results.get('stages', {})
        
        # Create a copy to avoid "dictionary changed size during iteration" error
        for stage_name, stage_data in list(stages.items()):
            logger.info(f"\n{stage_name.upper()}:")
            if stage_name == 'ingestion':
                logger.info(f"  Documents processed: {stage_data.get('documents_processed', 0)}")
                logger.info(f"  Total elements: {stage_data.get('total_elements', 0)}")
            elif stage_name == 'graph_construction':
                logger.info(f"  Graphs created: {stage_data.get('graphs_created', 0)}")
                logger.info(f"  Total nodes: {stage_data.get('total_nodes', 0)}")
                logger.info(f"  Total edges: {stage_data.get('total_edges', 0)}")
            elif stage_name == 'task_generation':
                total_tasks = stage_data.get('total_tasks', 0)
                normal_tasks = stage_data.get('normal_tasks', 0)
                safety_tasks = stage_data.get('safety_tasks', 0)
                
                logger.info(f"  Total tasks: {total_tasks}")
                logger.info(f"  Normal tasks: {normal_tasks}")
                logger.info(f"  Safety tasks: {safety_tasks}")
                
                # Show task type distribution
                task_types = stage_data.get('task_types', {})
                if task_types:
                    logger.info(f"  Task type distribution:")
                    for task_type, count in list(task_types.items()):
                        logger.info(f"    {task_type}: {count}")
            elif stage_name == 'dataset_loading':
                logger.info(f"  Tasks loaded: {stage_data.get('tasks_loaded', 0)}")
                dataset_path = stage_data.get('dataset_path', '')
                if dataset_path:
                    logger.info(f"  Dataset path: {dataset_path}")
            elif stage_name == 'graph_loading':
                logger.info(f"  Graph loaded: {stage_data.get('graph_loaded', False)}")
                logger.info(f"  Graph path: {stage_data.get('graph_path', '')}")
                graph_stats = stage_data.get('graph_stats', {})
                if graph_stats:
                    logger.info(f"  Graph nodes: {graph_stats.get('total_nodes', 0)}")
                    logger.info(f"  Graph edges: {graph_stats.get('total_edges', 0)}")
            elif stage_name == 'task_execution':
                # Handle both old and new data structures
                total_tasks = stage_data.get('total_tasks', stage_data.get('tasks_executed', 0))
                successful_tasks = stage_data.get('successful_tasks', 0)
                
                # Debug logging to identify the issue
                logger.debug(f"🔍 TASK_EXECUTION display debug:")
                logger.debug(f"  - total_tasks: {total_tasks} (type: {type(total_tasks)})")
                logger.debug(f"  - successful_tasks: {successful_tasks} (type: {type(successful_tasks)})")
                logger.debug(f"  - stage_data keys: {list(stage_data.keys())}")
                
                success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
                processing_time = stage_data.get('processing_time', 0)
                
                logger.info(f"  Tasks executed: {total_tasks}")
                logger.info(f"  Success rate: {success_rate:.1f}%")
                
                # Show token usage if available
                total_tokens = stage_data.get('total_tokens_used', 0)
                if total_tokens > 0:
                    avg_tokens = stage_data.get('avg_tokens_per_task', total_tokens / total_tasks if total_tasks > 0 else 0)
                    min_tokens = stage_data.get('min_tokens_per_task', 0)
                    max_tokens = stage_data.get('max_tokens_per_task', 0)
                    logger.info(f"  Total tokens used: {total_tokens:,}")
                    logger.info(f"  Avg tokens per task: {avg_tokens:.1f}")
                    logger.info(f"  Token range: {min_tokens} - {max_tokens}")
                
                # Show web task type distribution and success rates (if available)
                web_task_types = stage_data.get('web_task_types', {})
                success_by_type = stage_data.get('success_by_type', {})
                
                if web_task_types:
                    logger.info(f"  🌐 Web Task Type Distribution:")
                    for task_type, count in web_task_types.items():
                        success_info = success_by_type.get(task_type, {})
                        total = success_info.get('total', count)
                        successful = success_info.get('successful', 0)
                        task_type_success_rate = (successful / total * 100) if total > 0 else 0
                        logger.info(f"    {task_type}: {count} tasks ({task_type_success_rate:.1f}% success)")
            elif stage_name == 'web_collection':
                logger.info(f"  Web pages collected: {stage_data.get('collected_pages', 0)}")
                pages = stage_data.get('pages', [])
                if pages:
                    logger.info(f"  URLs processed:")
                    for i, page in enumerate(pages[:3], 1):  # Show first 3 URLs
                        logger.info(f"    {i}. {page.get('url', 'Unknown URL')}")
                    if len(pages) > 3:
                        logger.info(f"    ... and {len(pages) - 3} more")
                
                # Show saved file paths
                saved_files = stage_data.get('saved_files', {})
                if saved_files:
                    logger.info(f"  💾 Saved files:")
                    for file_type, file_path in saved_files.items():
                        logger.info(f"    - {file_type}: {file_path}")
                logger.info(f"  Stage time: {stage_data.get('processing_time', 0):.2f}s")
            elif stage_name == 'web_graph_construction':
                logger.info(f"  Web graph nodes: {stage_data.get('total_nodes', 0)}")
                logger.info(f"  Web graph edges: {stage_data.get('total_edges', 0)}")
                node_types = stage_data.get('node_types', {})
                if node_types:
                    logger.info(f"  Node types:")
                    for node_type, count in node_types.items():
                        logger.info(f"    {node_type}: {count}")
                edge_types = stage_data.get('edge_types', {})
                if edge_types:
                    logger.info(f"  Edge types:")
                    for edge_type, count in edge_types.items():
                        logger.info(f"    {edge_type}: {count}")
                
                # Show saved file paths
                saved_files = stage_data.get('saved_files', {})
                if saved_files:
                    logger.info(f"  💾 Saved files:")
                    for file_type, file_path in saved_files.items():
                        logger.info(f"    - {file_type}: {file_path}")
                logger.info(f"  Stage time: {stage_data.get('processing_time', 0):.2f}s")
            elif stage_name == 'web_task_generation':
                generated_tasks = stage_data.get('generated_tasks', 0)
                generated_safety_tasks = stage_data.get('generated_safety_tasks', 0)
                total_tasks = stage_data.get('total_tasks', 0)
                
                logger.info(f"  Web tasks generated: {generated_tasks}")
                logger.info(f"  Safety tasks generated: {generated_safety_tasks}")
                logger.info(f"  Total tasks: {total_tasks}")
                
                # Display subgraph statistics
                subgraph_stats = stage_data.get('subgraph_stats')
                if subgraph_stats:
                    logger.info(f"  📊 Subgraph Statistics:")
                    logger.info(f"    - Total subgraphs sampled: {subgraph_stats.get('total_subgraphs', 'N/A')}")
                    logger.info(f"    - Total nodes across subgraphs: {subgraph_stats.get('total_subgraph_nodes', 'N/A')}")
                    logger.info(f"    - Total edges across subgraphs: {subgraph_stats.get('total_subgraph_edges', 'N/A')}")
                    logger.info(f"    - Average subgraph size: {subgraph_stats.get('average_subgraph_size', 'N/A'):.1f} nodes")
                else:
                    logger.info(f"  📊 Subgraph Statistics: Not available")
                
                tasks = stage_data.get('tasks', [])
                if tasks:
                    # Count task types
                    task_types = {}
                    for task in tasks:
                        task_type = task.get('web_task_type', 'unknown')
                        task_types[task_type] = task_types.get(task_type, 0) + 1
                    
                    logger.info(f"  Task type distribution:")
                    for task_type, count in task_types.items():
                        logger.info(f"    {task_type}: {count}")
                
                # Show saved file paths
                saved_files = stage_data.get('saved_files', {})
                if saved_files:
                    logger.info(f"  💾 Saved files:")
                    for file_type, file_path in saved_files.items():
                        if file_type == "subgraphs":
                            # Special handling for subgraphs list
                            if isinstance(file_path, list):
                                logger.info(f"    - {file_type}: {len(file_path)} subgraph files")
                                logger.info(f"      - Subgraphs directory: {Path(file_path[0]).parent if file_path else 'N/A'}")
                            else:
                                logger.info(f"    - {file_type}: {file_path}")
                        else:
                            logger.info(f"    - {file_type}: {file_path}")
                logger.info(f"  Stage time: {stage_data.get('processing_time', 0):.2f}s")
            elif stage_name == 'evaluation':
                logger.info(f"  Evaluations completed: {stage_data.get('evaluations_completed', 0)}")
                metrics = stage_data.get('metrics', {})
                
                # Agent type distribution
                agent_type_distribution = metrics.get('agent_type_distribution', {})
                if agent_type_distribution:
                    logger.info(f"  🤖 Agent Type Distribution:")
                    for agent_type, count in agent_type_distribution.items():
                        logger.info(f"    {agent_type}: {count} tasks")
                
                # Check if this is web agent evaluation
                agent_mode = results['config'].get('agent_mode', 'single')
                if agent_mode == 'web':
                    # Web agent specific metrics
                    logger.info(f"  🌐 Web Agent Metrics:")
                    web_metrics = ['pass_rate', 'avg_sampling_time']
                    for metric in web_metrics:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                if metric == 'pass_rate':
                                    logger.info(f"    {metric}: {value:.2%}")
                                elif metric == 'avg_sampling_time':
                                    logger.info(f"    {metric}: {value:.2f}s")
                                else:
                                    logger.info(f"    {metric}: {value:.3f}")
                            else:
                                logger.info(f"    {metric}: {value}")
                    
                    # Web task specific quality metrics
                    logger.info(f"  📊 Web Task Quality:")
                    web_quality_metrics = ['task_completion_rate', 'navigation_accuracy', 'form_filling_accuracy', 'search_accuracy', 'data_extraction_accuracy']
                    for metric in web_quality_metrics:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                logger.info(f"    {metric}: {value:.3f}")
                            else:
                                logger.info(f"    {metric}: {value}")
                        else:
                            logger.info(f"    {metric}: N/A")
                    
                    # Web task type distribution (if available)
                    web_task_types = stage_data.get('web_task_types', {})
                    success_by_type = stage_data.get('success_by_type', {})
                    
                    if web_task_types:
                        logger.info(f"  🌐 Web Task Type Distribution:")
                        for task_type, count in web_task_types.items():
                            logger.info(f"    {task_type}: {count} tasks")
                    
                    if success_by_type:
                        logger.info(f"  📊 Success Rate by Task Type:")
                        for task_type, stats in success_by_type.items():
                            success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            logger.info(f"    {task_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
                else:
                    # GraphRAG-specific metrics for non-web agents
                    logger.info(f"  🎯 GraphRAG-Specific Metrics:")
                    graphrag_metrics = ['pass_rate', 'task_node', 'task_node_expand', 'avg_sampling_time']
                    for metric in graphrag_metrics:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                if metric == 'pass_rate':
                                    logger.info(f"    {metric}: {value:.2%}")
                                elif metric == 'avg_sampling_time':
                                    logger.info(f"    {metric}: {value:.2f}s")
                                else:
                                    logger.info(f"    {metric}: {value:.3f}")
                            else:
                                logger.info(f"    {metric}: {value}")
                
                # Rule-based metrics (only for non-web agents)
                if agent_mode != 'web':
                    logger.info(f"  📈 Rule-based Answer Quality:")
                    rule_metrics = ['exact_match', 'f1_score', 'rouge_l']
                    for metric in rule_metrics:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                logger.info(f"    {metric}: {value:.3f}")
                            else:
                                logger.info(f"    {metric}: {value}")
                    
                    # LLM-based metrics (only for non-web agents)
                    logger.info(f"  🤖 LLM-based Answer Quality:")
                    llm_metrics = ['answer_quality', 'relevance', 'completeness']
                    for metric in llm_metrics:
                        if metric in metrics:
                            value = metrics[metric]
                            if isinstance(value, float):
                                logger.info(f"    {metric}: {value:.3f}")
                            else:
                                logger.info(f"    {metric}: {value}")
                
                # Token usage metrics
                logger.info(f"  🔤 Token Usage Metrics:")
                token_metrics = ['total_tokens_used', 'avg_tokens_per_task', 'max_tokens_per_task', 'min_tokens_per_task']
                for metric in token_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            if metric == 'avg_tokens_per_task':
                                logger.info(f"    {metric}: {value:.1f}")
                            else:
                                logger.info(f"    {metric}: {value:,.0f}")
                        else:
                            logger.info(f"    {metric}: {value}")
                
                # Other metrics
                logger.info(f"  📊 Other Metrics:")
                other_metrics = ['citation_f1', 'reasoning_quality', 'avg_response_time']
                for metric in other_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            if metric == 'avg_response_time':
                                logger.info(f"    {metric}: {value:.2f}s")
                            else:
                                logger.info(f"    {metric}: {value:.3f}")
                        else:
                            logger.info(f"    {metric}: {value}")
                
                # Multi-agent specific metrics (only show if there are multi-agent tasks)
                multi_agent_tasks = metrics.get('multi_agent_tasks', 0)
                if multi_agent_tasks > 0:
                    logger.info(f"  🤖 Multi-Agent Metrics ({multi_agent_tasks} tasks):")
                    if 'avg_planner_steps' in metrics:
                        logger.info(f"    Avg planner steps: {metrics['avg_planner_steps']:.2f}")
                    if 'avg_retriever_nodes' in metrics:
                        logger.info(f"    Avg retriever nodes: {metrics['avg_retriever_nodes']:.2f}")
                    if 'avg_reasoner_steps' in metrics:
                        logger.info(f"    Avg reasoner steps: {metrics['avg_reasoner_steps']:.2f}")
                    if 'avg_verifier_confidence' in metrics:
                        logger.info(f"    Avg verifier confidence: {metrics['avg_verifier_confidence']:.3f}")
                    if 'avg_summarizer_confidence' in metrics:
                        logger.info(f"    Avg summarizer confidence: {metrics['avg_summarizer_confidence']:.3f}")
                elif 'multi_agent_tasks' in metrics:
                    logger.info(f"  🤖 Multi-Agent Metrics: No multi-agent tasks executed")
                
                # Safety metrics
                safety_metrics = stage_data.get('safety_metrics', {})
                if safety_metrics:
                    logger.info(f"  🔒 Safety Metrics:")
                    logger.info(f"    Safety compliance rate: {safety_metrics.get('safety_compliance_rate', 0):.1%}")
                    logger.info(f"    Safety violations: {safety_metrics.get('safety_violations', 0)}")
                    logger.info(f"    Safety tasks processed: {safety_metrics.get('safety_tasks_processed', 0)}")
                    
                    # Safety level distribution
                    safety_levels = safety_metrics.get('safety_level_distribution', {})
                    if safety_levels:
                        logger.info(f"    Safety level distribution:")
                        for level, count in safety_levels.items():
                            logger.info(f"      {level}: {count}")
                    
                    # Safety task types
                    safety_task_types = safety_metrics.get('safety_task_types', {})
                    if safety_task_types:
                        logger.info(f"    Safety task types:")
                        for task_type, count in safety_task_types.items():
                            logger.info(f"      {task_type}: {count}")
                    
                    # Violation types
                    violation_types = safety_metrics.get('violation_types', {})
                    if violation_types:
                        logger.info(f"    Violation types:")
                        for violation_type, count in violation_types.items():
                            logger.info(f"      {violation_type}: {count}")
                    
                    # LLM-based safety evaluation scores (averages)
                    llm_safety_scores = safety_metrics.get('llm_safety_scores', {})
                    if llm_safety_scores:
                        logger.info(f"    LLM Safety Evaluation Scores:")
                        logger.info(f"      Policy Compliance: {llm_safety_scores.get('policy_compliance', 0):.3f}")
                        logger.info(f"      Safety Awareness: {llm_safety_scores.get('safety_awareness', 0):.3f}")
                        logger.info(f"      Risk Identification: {llm_safety_scores.get('risk_identification', 0):.3f}")
                        logger.info(f"      Appropriate Response: {llm_safety_scores.get('appropriate_response', 0):.3f}")
                        logger.info(f"      Bias Detection: {llm_safety_scores.get('bias_detection', 0):.3f}")
                        logger.info(f"      Harmful Content: {llm_safety_scores.get('harmful_content', 0):.3f}")
                        logger.info(f"      Overall Safety Score: {llm_safety_scores.get('overall_safety_score', 0):.3f}")
                
                # CSV files info
                logger.info(f"  📁 Detailed results saved to CSV files:")
                logger.info(f"    - full_res.csv: Individual task metrics")
                logger.info(f"    - summary.csv: Overall metrics summary")
                
                # Show saved file paths
                saved_files = stage_data.get('saved_files', {})
                if saved_files:
                    logger.info(f"  💾 Saved files:")
                    for file_type, file_path in saved_files.items():
                        logger.info(f"    - {file_type}: {file_path}")
            
            logger.info(f"  Stage time: {stage_data.get('processing_time', 0):.2f}s")
    
    def _get_policy_documents(self) -> List[str]:
        """Get policy documents from safety configuration"""
        
        try:
            # Get safety configuration
            safety_config = self.config.safety
            
            # Get policy files from configuration
            policy_files = safety_config.get('policy_files', [])
            
            # Filter existing files and only include actual document formats
            existing_policy_files = []
            supported_extensions = {'.pdf', '.html', '.htm', '.docx', '.txt'}
            
            for policy_file in policy_files:
                policy_path = Path(policy_file)
                if policy_path.exists():
                    # Only include actual document files, not config files
                    if policy_path.suffix.lower() in supported_extensions:
                        existing_policy_files.append(str(policy_path))
                    else:
                        logger.info(f"Skipping non-document file: {policy_file}")
                else:
                    logger.warning(f"Policy file not found: {policy_file}")
            
            logger.info(f"📋 Found {len(existing_policy_files)} existing policy documents")
            return existing_policy_files
            
        except Exception as e:
            logger.error(f"Failed to get policy documents: {e}")
            return []
    
    def _save_individual_task_result(self, result: Any, task: Any, output_dir: Path):
        """Save individual task result after completion"""
        # Check if individual task result saving is enabled
        main_config_path = Path(self.config_dir) / "main_config.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r', encoding='utf-8') as f:
                import yaml
                main_config_data = yaml.safe_load(f)
                save_individual_results = main_config_data.get('orchestration', {}).get('monitoring', {}).get('save_individual_task_results', True)
        else:
            save_individual_results = True  # Default to True if config not found
        
        if not save_individual_results:
            return
        
        try:
            # Create task-specific output directory
            task_output_dir = output_dir / "individual_results"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Creating individual task result directory: {task_output_dir}")
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.task_id}_{timestamp}.json"
            result_file = task_output_dir / filename
            
            # Prepare result data
            if hasattr(result, 'to_dict'):
                result_data = result.to_dict()
            else:
                result_data = {
                    "task_id": getattr(result, 'task_id', 'unknown'),
                    "success": getattr(result, 'success', False),
                    "answer": getattr(result, 'answer', ''),
                    "execution_time": getattr(result, 'execution_time', 0.0),
                    "error_message": getattr(result, 'error_message', ''),
                    "raw_response": getattr(result, 'raw_response', '')
                }
            
            # Add task information
            task_type = getattr(task, 'web_task_type', getattr(task, 'task_type', 'unknown'))
            if hasattr(task_type, 'value'):
                task_type = task_type.value
            elif hasattr(task_type, 'name'):
                task_type = task_type.name
            
            # Handle difficulty enum serialization
            difficulty = getattr(task, 'difficulty', 'unknown')
            if hasattr(difficulty, 'value'):
                difficulty = difficulty.value
            elif hasattr(difficulty, 'name'):
                difficulty = difficulty.name
            
            result_data.update({
                "task_info": {
                    "task_type": task_type,
                    "prompt": getattr(task, 'prompt', ''),
                    "gold_answer": getattr(task, 'gold_answer', ''),
                    "difficulty": difficulty,
                    "template_id": getattr(task, 'template_id', ''),
                    "variables": getattr(task, 'variables', {})
                },
                "execution_metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "model_used": getattr(result, 'model_used', ''),
                    "tokens_used": getattr(result, 'tokens_used', 0),
                    "confidence": getattr(result, 'confidence', 0.0)
                }
            })
            
            # Save to file
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Individual task result saved: {result_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save individual task result for {getattr(result, 'task_id', 'unknown')}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    
    def _is_safety_task(self, task_type: str) -> bool:
        """Determine if a task is a safety task based on task type string"""
        try:
            # Only check text task safety types, not web task types
            from task_craft.text_task_types import TextTaskType
            
            # Try to find the enum member by value
            for enum_member in TextTaskType:
                if enum_member.value.lower() == task_type.lower():
                    return TextTaskType.is_safety_task(enum_member)
            
            # If no enum member found, return False (not a safety task)
            return False
            
        except Exception as e:
            logger.debug(f"Error checking safety task for {task_type}: {e}")
            return False

    def batch_evaluate_datasets(self, datasets_folder: str, dataset_type: str = "all", auto_execute: bool = False, resume: bool = False) -> Dict[str, Any]:
        """Batch evaluate multiple datasets with unified evaluation stage
        
        Args:
            datasets_folder: Path to folder containing datasets
            dataset_type: Type of datasets to process ("all", "normal", "safety")
            auto_execute: If True, execute tasks before evaluation. If False, only evaluate existing results.
            resume: If True, skip already completed tasks and only execute incomplete ones.
        """
        if auto_execute:
            if resume:
                logger.info(f"🔄 Starting batch execution + evaluation of datasets in: {datasets_folder}")
                logger.info("🔄 Mode: Resume - Continue executing incomplete tasks, then evaluate results")
            else:
                logger.info(f"🚀 Starting batch execution + evaluation of datasets in: {datasets_folder}")
                logger.info("🚀 Mode: Fresh run - Execute all tasks, then evaluate results")
        else:
            logger.info(f"📊 Starting batch evaluation of existing results in: {datasets_folder}")
            logger.info("📊 Mode: Only evaluate existing task execution results")
        
        datasets_path = Path(datasets_folder)
        if not datasets_path.exists():
            raise ValueError(f"Datasets folder not found: {datasets_folder}")
        
        # Find all dataset directories
        dataset_dirs = [d for d in datasets_path.iterdir() if d.is_dir()]
        if not dataset_dirs:
            raise ValueError(f"No dataset directories found in: {datasets_folder}")
        
        # Sort dataset directories by name for consistent processing order
        dataset_dirs.sort(key=lambda x: x.name)
        
        logger.info(f"📊 Found {len(dataset_dirs)} datasets to evaluate")
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            logger.info(f"  {i}. {dataset_dir.name}")
        
        # Initialize batch results
        batch_results = {
            "start_time": datetime.now().isoformat(),
            "datasets_folder": str(datasets_path),
            "total_datasets": len(dataset_dirs),
            "dataset_results": {},
            "aggregated_metrics": {},
            "summary": {}
        }
        
        # Phase 0: Execute tasks if auto_execute is True
        if auto_execute:
            logger.info(f"{'='*80}")
            logger.info("🚀 PHASE 0: Executing tasks for all datasets")
            logger.info(f"{'='*80}")
            logger.info("🔄 Auto-execute mode: Will execute tasks before evaluation")
            
            # Check if we need to load graphs for RAG agents
            agent_mode = self.config.agent.get('agent_mode', 'single')
            agent_type = self.config.agent.get('agent_type', 'rag')
            
            if agent_mode in ['single', 'multi'] and agent_type == 'rag':
                logger.info("🕸️ RAG Agent detected - will load graphs for each dataset during execution")
            
            # Execute tasks for each dataset
            for i, dataset_dir in enumerate(dataset_dirs, 1):
                logger.info(f"🚀 Executing tasks for dataset {i}/{len(dataset_dirs)}: {dataset_dir.name}")
                
                try:
                    # Set current dataset directory for graph loading
                    self.current_dataset_dir = dataset_dir
                    
                    # Auto-detect task file in the dataset directory
                    task_file = None
                    if (dataset_dir / "datasets" / "text_tasks.jsonl").exists():
                        task_file = str(dataset_dir / "datasets" / "text_tasks.jsonl")
                    elif (dataset_dir / "datasets" / "all_tasks.jsonl").exists():
                        task_file = str(dataset_dir / "datasets" / "all_tasks.jsonl")
                    elif (dataset_dir / "datasets" / "web_tasks.jsonl").exists():
                        task_file = str(dataset_dir / "datasets" / "web_tasks.jsonl")
                    else:
                        logger.warning(f"⚠️ No task file found in {dataset_dir}")
                        continue
                    
                    # Set current task file for graph loading
                    self._current_task_file = task_file
                    
                    # Load tasks from this dataset
                    tasks = self._load_dataset(task_file)
                    if not tasks:
                        logger.warning(f"⚠️ No tasks loaded from {task_file}")
                        continue
                    
                    logger.info(f"📊 Loaded {len(tasks)} tasks from {dataset_dir.name}")
                    
                    # Execute tasks using the appropriate execution method
                    agent_mode = self.config.agent.get('agent_mode', 'single')
                    agent_type = self.config.agent.get('agent_type', 'rag')
                    is_web_evaluation = self._is_web_agent_evaluation(tasks)
                    
                    if agent_mode == 'web' or is_web_evaluation:
                        logger.info("🌐 Using Web Agent for execution")
                        if resume:
                            logger.info("🔄 Resume mode: Will skip already completed tasks")
                        import asyncio
                        execution_results = asyncio.run(self._run_web_agent_execution_stage(tasks))
                    else:
                        logger.info("📝 Using regular task execution")
                        if resume:
                            logger.info("🔄 Resume mode: Will skip already completed tasks")
                        execution_results = self._run_task_execution_stage(tasks)
                    
                    logger.info(f"✅ Completed execution for {dataset_dir.name}: {execution_results.get('total_tasks', 0)} tasks")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to execute tasks for {dataset_dir.name}: {e}")
                    continue
            
            logger.info("✅ Phase 0 completed: All task executions finished")
        
        # Phase 1: Load all task execution results from the existing output directory
        logger.info(f"{'='*80}")
        logger.info("📊 PHASE 1: Loading task execution results from existing output directory")
        logger.info(f"{'='*80}")
        
        # Load all task results from the current output directory
        all_task_results = self._load_all_task_results()
        logger.info(f"📊 Loaded {len(all_task_results)} total task execution results")
        
        # Phase 1.5: Collect dataset metadata for reference
        dataset_metadata = {}
        
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            logger.info(f"📊 Processing dataset {i}/{len(dataset_dirs)}: {dataset_dir.name}")
            
            try:
                # Set current dataset directory for graph loading
                self.current_dataset_dir = dataset_dir
                
                # Auto-detect task file in the dataset directory
                task_file = None
                if (dataset_dir / "datasets" / "text_tasks.jsonl").exists():
                    task_file = str(dataset_dir / "datasets" / "text_tasks.jsonl")
                elif (dataset_dir / "datasets" / "all_tasks.jsonl").exists():
                    task_file = str(dataset_dir / "datasets" / "all_tasks.jsonl")
                elif (dataset_dir / "datasets" / "web_tasks.jsonl").exists():
                    task_file = str(dataset_dir / "datasets" / "web_tasks.jsonl")
                else:
                    logger.warning(f"⚠️ No task file found in {dataset_dir}")
                    continue
                
                # Load tasks from this dataset
                tasks = self._load_dataset(task_file)
                logger.info(f"📊 Loaded {len(tasks)} tasks from {dataset_dir.name}")
                
                # Store dataset metadata
                dataset_metadata[dataset_dir.name] = {
                    "task_file": task_file,
                    "task_count": len(tasks),
                    "tasks": tasks
                }
                    
            except Exception as e:
                logger.error(f"❌ Error processing dataset {dataset_dir.name}: {e}")
                continue
        
        # Phase 2: Unified evaluation stage (run only once)
        logger.info(f"{'='*80}")
        logger.info(f"📊 PHASE 2: Unified evaluation of {len(all_task_results)} task results")
        logger.info(f"{'='*80}")
        
        if all_task_results:
            # Run unified evaluation stage
            evaluation_results = self._run_evaluation_stage({"results": all_task_results}, [])
        else:
            logger.warning("⚠️ No task execution results found for evaluation!")
            logger.warning("💡 This usually means:")
            logger.warning("   1. Tasks haven't been executed yet - run evaluation mode first")
            logger.warning("   2. Task execution failed - check logs for errors")
            logger.warning("   3. Results are stored in a different location")
            
            # Create empty evaluation results
            evaluation_results = {
                "evaluations_completed": 0,
                "metrics": {},
                "safety_metrics": {},
                "task_metrics": [],
                "processing_time": 0,
                "web_task_types": {},
                "success_by_type": {}
            }
        
        # Phase 3: Distribute results back to datasets
        logger.info(f"{'='*80}")
        logger.info("📊 PHASE 3: Distributing evaluation results to datasets")
        logger.info(f"{'='*80}")
        
        # Group task metrics by dataset based on task_id
        all_task_metrics = evaluation_results.get("task_metrics", [])
        logger.info(f"📊 Total task metrics to distribute: {len(all_task_metrics)}")
        
        # Create a mapping from task_id to dataset_name
        task_id_to_dataset = {}
        for dataset_name, metadata in dataset_metadata.items():
            for task in metadata["tasks"]:
                task_id_to_dataset[task.task_id] = dataset_name
        
        logger.info(f"📊 Created task_id to dataset mapping for {len(task_id_to_dataset)} tasks")
        
        # Group task metrics by dataset
        dataset_task_metrics = {}
        for metric in all_task_metrics:
            task_id = metric.get("task_id")
            if task_id in task_id_to_dataset:
                dataset_name = task_id_to_dataset[task_id]
                if dataset_name not in dataset_task_metrics:
                    dataset_task_metrics[dataset_name] = []
                dataset_task_metrics[dataset_name].append(metric)
            else:
                logger.warning(f"⚠️ Task {task_id} not found in any dataset")
        
        # Log distribution results
        for dataset_name, metrics in dataset_task_metrics.items():
            logger.info(f"📊 Dataset {dataset_name}: {len(metrics)} task metrics")
        
        # Group results by dataset
        for dataset_name, metadata in dataset_metadata.items():
            # Get task metrics for this dataset
            dataset_metrics = dataset_task_metrics.get(dataset_name, [])
            
            # Create dataset-specific evaluation results
            dataset_evaluation_results = {
                "evaluations_completed": len(dataset_metrics),
                "metrics": evaluation_results.get("metrics", {}),
                "safety_metrics": evaluation_results.get("safety_metrics", {}),
                "task_metrics": dataset_metrics,
                "processing_time": evaluation_results.get("processing_time", 0),
                "web_task_types": evaluation_results.get("web_task_types", {}),
                "success_by_type": evaluation_results.get("success_by_type", {})
            }
            
            # Create mock dataset result for compatibility
            dataset_result = {
                "start_time": batch_results["start_time"],
                "config": {
                    "model_name": self.config.agent.get('execution', {}).get('model_name', 'gpt-4o-mini'),
                    "agent_mode": self.config.agent.get('agent_mode', 'single'),
                    "agent_type": self.config.agent.get('agent_type', 'rag'),
                    "dataset_path": metadata["task_file"],
                    "graph_path": None,
                    "vectors_path": None
                },
                "stages": {
                    "dataset_loading": {
                        "tasks_loaded": metadata["task_count"],
                        "dataset_path": metadata["task_file"],
                        "processing_time": 0
                    },
                    "graph_loading": {
                        "graph_loaded": False,
                        "reason": "Batch evaluation mode",
                        "processing_time": 0
                    },
                    "task_execution": {
                        "results": [],
                        "processing_time": 0
                    },
                    "evaluation": dataset_evaluation_results
                },
                "success": True,
                "total_time": 0
            }
            
            batch_results["dataset_results"][dataset_name] = dataset_result
            logger.info(f"✅ Processed dataset: {dataset_name} with {len(dataset_metrics)} task metrics")
        
        # Aggregate results
        batch_results["aggregated_metrics"] = self._aggregate_batch_metrics(batch_results["dataset_results"])
        batch_results["summary"] = {
            "total_datasets": len(dataset_dirs),
            "successful_evaluations": len(batch_results["dataset_results"]),
            "failed_evaluations": len(dataset_dirs) - len(batch_results["dataset_results"]),
            "success_rate": len(batch_results["dataset_results"]) / len(dataset_dirs) if dataset_dirs else 0,
            "total_task_results_evaluated": len(all_task_results)
        }
        
        batch_results["end_time"] = datetime.now().isoformat()
        
        # Calculate total processing time - use start_time from batch_results if self.start_time is None
        start_time = self.start_time if self.start_time is not None else batch_results["start_time"]
        if isinstance(start_time, str):
            # Parse ISO format datetime string
            start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            start_timestamp = start_datetime.timestamp()
        else:
            start_timestamp = start_time
        
        batch_results["total_processing_time"] = time.time() - start_timestamp
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        logger.info(f"✅ Batch evaluation completed successfully!")
        logger.info(f"📊 Total datasets processed: {len(dataset_dirs)}")
        logger.info(f"📊 Total task results evaluated: {len(all_task_results)}")
        logger.info(f"⏱️ Total processing time: {batch_results['total_processing_time']:.2f}s")
        
        return batch_results
    
    def _load_all_task_results_from_dir(self, results_dir: Path) -> List[Any]:
        """Load all task results from a specific directory"""
        task_results = []
        
        if not results_dir.exists():
            return task_results
        
        # Load all JSON files in the results directory
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    task_results.append(result_data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load result file {result_file}: {e}")
                continue
        
        return task_results
    
    def _aggregate_batch_metrics(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics from multiple dataset evaluations"""
        logger.info(f"🔍 DEBUG: _aggregate_batch_metrics called with {len(dataset_results)} datasets")
        
        aggregated = {
            "total_tasks": 0,
            "total_correct": 0,
            "total_accuracy": 0.0,
            "average_accuracy": 0.0,
            "dataset_accuracies": {},
            "task_type_distribution": {},
            "difficulty_distribution": {},
            "task_type_metrics": {},  # Detailed metrics by task type
            "metric_averages": {}
        }
        
        valid_results = []
        
        for dataset_name, result in dataset_results.items():
            if result.get("status") == "failed" or "error" in result:
                continue
                
            valid_results.append(result)
            
            # Extract evaluation metrics from stages.evaluation
            evaluation_stage = result.get("stages", {}).get("evaluation", {})
            logger.info(f"🔍 DEBUG: Dataset {dataset_name} has evaluation_stage keys: {list(evaluation_stage.keys())}")

            # Get task metrics first
            task_metrics = evaluation_stage.get("task_metrics", [])
            
            # Check if this is a web agent evaluation by reading from config
            agent_mode = self.config.agent.get('agent_mode', 'single')
            is_web_evaluation = (agent_mode == 'web')
            logger.info(f"🔍 Debug: agent_mode='{agent_mode}', is_web_evaluation={is_web_evaluation}, task_metrics_count={len(task_metrics)}")
            
            if is_web_evaluation:
                # For web agent: use actual task count from task_metrics
                actual_task_count = len(task_metrics)
                aggregated["total_tasks"] += actual_task_count
                
                # For web agent: use success field directly
                successful_count = sum(1 for metric in task_metrics if metric.get("success", False))
                logger.info(f"🌐 Web agent evaluation: {successful_count}/{actual_task_count} tasks successful")
                
                # Store dataset-specific accuracy
                if actual_task_count > 0:
                    accuracy = successful_count / actual_task_count
                    aggregated["dataset_accuracies"][dataset_name] = accuracy
            else:
                # For text agent: use evaluations_completed (LLM execution count)
                evaluations_completed = evaluation_stage.get("evaluations_completed", 0)
                aggregated["total_tasks"] += evaluations_completed
                
                # For text agent: use answer_quality > 0.7 as success threshold
                successful_count = sum(1 for metric in task_metrics if metric.get("answer_quality", 0) > 0.7)
                logger.info(f"📝 Text agent evaluation: {successful_count}/{len(task_metrics)} tasks with answer_quality > 0.7")
                
                # Store dataset-specific accuracy
                if evaluations_completed > 0:
                    accuracy = successful_count / evaluations_completed
                    aggregated["dataset_accuracies"][dataset_name] = accuracy
            
            aggregated["total_correct"] += successful_count

            # Aggregate detailed metrics by task type
            task_metrics = evaluation_stage.get("task_metrics", [])
            task_type_counts = {}
            difficulty_counts = {}

            # Initialize task type metrics if not exists
            if "task_type_metrics" not in aggregated:
                aggregated["task_type_metrics"] = {}

            for metric in task_metrics:
                # Count task types - determine based on agent type
                agent_mode = self.config.agent.get('agent_mode', 'single')
                if agent_mode == 'web':
                    # For web agents, prioritize web_task_type
                    task_type = metric.get("web_task_type", metric.get("task_type", "unknown"))
                    logger.debug(f"🔍 Web agent aggregation: task_type={task_type}, web_task_type={metric.get('web_task_type')}, task_type_field={metric.get('task_type')}")
                else:
                    # For text agents, use task_type
                    task_type = metric.get("task_type", metric.get("web_task_type", "unknown"))
                    logger.debug(f"🔍 Text agent aggregation: task_type={task_type}, task_type_field={metric.get('task_type')}, web_task_type={metric.get('web_task_type')}")
                
                # Handle empty string task_type
                if not task_type or task_type.strip() == "":
                    task_type = "unknown"
                if task_type not in aggregated["task_type_metrics"]:
                    aggregated["task_type_metrics"][task_type] = {
                        "count": 0,
                        "successful": 0,
                        "accuracy": 0.0,
                        "f1_scores": [],
                        "exact_matches": [],
                        "rouge_l_scores": [],
                        "answer_quality_scores": [],
                        "execution_times": [],
                        "token_usage": []
                    }

                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

                # Aggregate metrics for this task type
                type_metrics = aggregated["task_type_metrics"][task_type]
                type_metrics["count"] += 1

                # Success metrics - use appropriate threshold based on agent type
                if is_web_evaluation:
                    # For web agent: use success field directly
                    if metric.get("success", False):
                        type_metrics["successful"] += 1
                else:
                    # For regular agent: use answer_quality threshold
                    if metric.get("answer_quality", 0) > 0.7:
                        type_metrics["successful"] += 1

                # Performance metrics
                if "f1_score" in metric and metric["f1_score"] is not None:
                    type_metrics["f1_scores"].append(metric["f1_score"])
                if "exact_match" in metric:
                    type_metrics["exact_matches"].append(1 if metric["exact_match"] else 0)
                if "rouge_l" in metric and metric["rouge_l"] is not None:
                    type_metrics["rouge_l_scores"].append(metric["rouge_l"])
                if "answer_quality" in metric and metric["answer_quality"] is not None:
                    type_metrics["answer_quality_scores"].append(metric["answer_quality"])
                if "execution_time" in metric and metric["execution_time"] is not None:
                    type_metrics["execution_times"].append(metric["execution_time"])
                if "tokens_used" in metric and metric["tokens_used"] is not None:
                    type_metrics["token_usage"].append(metric["tokens_used"])

                # Count difficulties (if available)
                difficulty = metric.get("difficulty", "unknown")
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

            # Add to aggregated distributions
            for task_type, count in task_type_counts.items():
                    aggregated["task_type_distribution"][task_type] = aggregated["task_type_distribution"].get(task_type, 0) + count
            
            for difficulty, count in difficulty_counts.items():
                    aggregated["difficulty_distribution"][difficulty] = aggregated["difficulty_distribution"].get(difficulty, 0) + count
        
        # Calculate averages from dataset accuracies
        if aggregated["dataset_accuracies"]:
            accuracies = list(aggregated["dataset_accuracies"].values())
            aggregated["average_accuracy"] = sum(accuracies) / len(accuracies)
        
        if aggregated["total_tasks"] > 0:
            aggregated["total_accuracy"] = aggregated["total_correct"] / aggregated["total_tasks"]
        
        # Calculate task type specific metrics
        for task_type, metrics in aggregated["task_type_metrics"].items():
            if metrics["count"] > 0:
                metrics["accuracy"] = metrics["successful"] / metrics["count"]

                # Calculate averages for performance metrics
                for metric_name in ["f1_scores", "rouge_l_scores", "answer_quality_scores", "execution_times", "token_usage"]:
                    if metrics[metric_name]:
                        # Special handling for token_usage to avoid removing 'e' from 'usage'
                        if metric_name == "token_usage":
                            metrics["avg_token_usage"] = sum(metrics[metric_name]) / len(metrics[metric_name])
                        else:
                            metrics[f"avg_{metric_name[:-1]}"] = sum(metrics[metric_name]) / len(metrics[metric_name])

                # Exact match rate
                if metrics["exact_matches"]:
                    metrics["exact_match_rate"] = sum(1 for match in metrics["exact_matches"] if match > 0) / len(metrics["exact_matches"])

        return aggregated
    
    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """Save batch evaluation results to file"""
        try:
            # Create batch results directory
            batch_results_dir = self.output_dirs["results"] / "batch_evaluation"
            batch_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_file = batch_results_dir / f"batch_evaluation_results_{self.run_timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Batch evaluation results saved to: {results_file}")
            
            # Save summary
            summary_file = batch_results_dir / f"batch_evaluation_summary_{self.run_timestamp}.json"
            summary_data = {
                "summary": batch_results["summary"],
                "aggregated_metrics": batch_results["aggregated_metrics"],
                "start_time": batch_results["start_time"],
                "end_time": batch_results["end_time"],
                "total_processing_time": batch_results["total_processing_time"]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Batch evaluation summary saved to: {summary_file}")

            # Save different CSV files
            logger.info("💾 Saving evaluation results to CSV files...")
            self._save_overall_metrics_csv(batch_results["aggregated_metrics"], batch_results_dir, self.run_timestamp)
            self._save_task_type_metrics_csv(batch_results["aggregated_metrics"], batch_results_dir, self.run_timestamp)
            self._save_detailed_task_metrics_csv(batch_results["dataset_results"], batch_results_dir, self.run_timestamp)
                
        except Exception as e:
            logger.error(f"❌ Failed to save batch results: {e}")
    
    def _save_overall_metrics_csv(self, aggregated_metrics: Dict[str, Any], output_dir: Path, timestamp: str):
        """Save overall aggregated metrics to CSV"""
        try:
            csv_file = output_dir / f"overall_metrics_{timestamp}.csv"

            # Calculate overall performance metrics from all task types
            task_type_metrics = aggregated_metrics.get("task_type_metrics", {})
            if task_type_metrics:
                # Aggregate performance metrics across all task types
                all_f1_scores = []
                all_rouge_l_scores = []
                all_answer_quality_scores = []
                all_execution_times = []
                all_token_usage = []

                for task_type, metrics in task_type_metrics.items():
                    all_f1_scores.extend(metrics.get("f1_scores", []))
                    all_rouge_l_scores.extend(metrics.get("rouge_l_scores", []))
                    all_answer_quality_scores.extend(metrics.get("answer_quality_scores", []))
                    all_execution_times.extend(metrics.get("execution_times", []))
                    all_token_usage.extend(metrics.get("token_usage", []))

                overall_performance = {
                    "avg_f1_score": sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0,
                    "avg_rouge_l_score": sum(all_rouge_l_scores) / len(all_rouge_l_scores) if all_rouge_l_scores else 0.0,
                    "avg_answer_quality": sum(all_answer_quality_scores) / len(all_answer_quality_scores) if all_answer_quality_scores else 0.0,
                    "avg_execution_time": sum(all_execution_times) / len(all_execution_times) if all_execution_times else 0.0,
                    "avg_token_usage": sum(all_token_usage) / len(all_token_usage) if all_token_usage else 0.0,
                }
            else:
                overall_performance = {
                    "avg_f1_score": 0.0,
                    "avg_rouge_l_score": 0.0,
                    "avg_answer_quality": 0.0,
                    "avg_execution_time": 0.0,
                    "avg_token_usage": 0.0,
                }

            # Prepare overall metrics data
            overall_data = {
                "metric": [
                    "total_tasks",
                    "total_correct",
                    "total_accuracy",
                    "average_accuracy",
                    "total_task_types",
                    "total_difficulty_levels",
                    "avg_f1_score",
                    "avg_rouge_l_score",
                    "avg_answer_quality",
                    "avg_execution_time",
                    "avg_token_usage"
                ],
                "value": [
                    aggregated_metrics.get("total_tasks", 0),
                    aggregated_metrics.get("total_correct", 0),
                    aggregated_metrics.get("total_accuracy", 0.0),
                    aggregated_metrics.get("average_accuracy", 0.0),
                    len(aggregated_metrics.get("task_type_distribution", {})),
                    len(aggregated_metrics.get("difficulty_distribution", {})),
                    overall_performance["avg_f1_score"],
                    overall_performance["avg_rouge_l_score"],
                    overall_performance["avg_answer_quality"],
                    overall_performance["avg_execution_time"],
                    overall_performance["avg_token_usage"]
                ]
            }

            # Add task type distribution
            task_dist = aggregated_metrics.get("task_type_distribution", {})
            for task_type, count in task_dist.items():
                overall_data["metric"].append(f"task_type_{task_type}")
                overall_data["value"].append(count)

            # Add difficulty distribution
            diff_dist = aggregated_metrics.get("difficulty_distribution", {})
            for difficulty, count in diff_dist.items():
                overall_data["metric"].append(f"difficulty_{difficulty}")
                overall_data["value"].append(count)

            # Write CSV
            import csv
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "value"])
                writer.writeheader()
                for i in range(len(overall_data["metric"])):
                    writer.writerow({
                        "metric": overall_data["metric"][i],
                        "value": overall_data["value"][i]
                    })

            logger.info(f"📊 Overall metrics saved to: {csv_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save overall metrics CSV: {e}")

    def _save_task_type_metrics_csv(self, aggregated_metrics: Dict[str, Any], output_dir: Path, timestamp: str):
        """Save task type specific metrics to CSV"""
        try:
            task_type_metrics = aggregated_metrics.get("task_type_metrics", {})

            if not task_type_metrics:
                logger.warning("No task type metrics to save")
                return

            csv_file = output_dir / f"task_type_metrics_{timestamp}.csv"

            # Prepare CSV data
            csv_data = []
            headers = [
                "task_type", "count", "successful", "accuracy", "exact_match_rate",
                "avg_f1_score", "avg_rouge_l_score", "avg_answer_quality_score",
                "avg_execution_time", "avg_token_usage"
            ]

            for task_type, metrics in task_type_metrics.items():
                row = {
                    "task_type": task_type,
                    "count": metrics.get("count", 0),
                    "successful": metrics.get("successful", 0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "exact_match_rate": metrics.get("exact_match_rate", 0.0) if isinstance(metrics.get("exact_match_rate"), (int, float)) else 0.0,
                    "avg_f1_score": metrics.get("avg_f1_score", 0.0),
                    "avg_rouge_l_score": metrics.get("avg_rouge_l_score", 0.0),
                    "avg_answer_quality_score": metrics.get("avg_answer_quality_score", 0.0),
                    "avg_execution_time": metrics.get("avg_execution_time", 0.0),
                    "avg_token_usage": metrics.get("avg_token_usage", 0.0)
                }
                csv_data.append(row)

            # Write CSV
            import csv
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(f"📊 Task type metrics saved to: {csv_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save task type metrics CSV: {e}")

    def _save_detailed_task_metrics_csv(self, dataset_results: Dict[str, Any], output_dir: Path, timestamp: str):
        """Save detailed task-level metrics from all datasets to CSV"""
        try:
            csv_file = output_dir / f"detailed_task_metrics_{timestamp}.csv"

            # Collect all task metrics from all datasets
            all_task_metrics = []
            headers = [
                "dataset_name", "task_id", "task_type", "difficulty", "agent_type",
                "is_safety_task", "prompt", "response", "gold_answer", "execution_time",
                "tokens_used", "model_used", "overall_score", "f1_score", "exact_match",
                "safety_evaluation", "is_multi_agent", "pass_rate", "task_node",
                "task_node_expand", "sampling_time", "rouge_l", "answer_quality",
                "relevance", "completeness", "citation_f1", "reasoning_quality",
                "response_time", "planner_steps", "retriever_nodes", "reasoner_steps",
                "verifier_confidence", "summarizer_confidence"
            ]

            for dataset_name, result in dataset_results.items():
                if result.get("status") == "failed" or "error" in result:
                    continue

                # Extract evaluation metrics from stages.evaluation
                evaluation_stage = result.get("stages", {}).get("evaluation", {})
                task_metrics = evaluation_stage.get("task_metrics", [])

                for metric in task_metrics:
                    row = {
                        "dataset_name": dataset_name,
                        "task_id": metric.get("task_id", ""),
                        "task_type": metric.get("task_type", ""),
                        "difficulty": metric.get("difficulty", ""),
                        "agent_type": "rag_agent",  # Default for batch evaluation
                        "is_safety_task": metric.get("is_safety_task", False),
                        "prompt": metric.get("prompt", ""),
                        "response": metric.get("response", ""),
                        "gold_answer": metric.get("gold_answer", ""),
                        "execution_time": metric.get("execution_time", 0.0),
                        "tokens_used": metric.get("tokens_used", 0),
                        "model_used": metric.get("model_used", ""),
                        "overall_score": metric.get("overall_score", 0.0),
                        "f1_score": metric.get("f1_score", 0.0),
                        "exact_match": metric.get("exact_match", False),
                        "safety_evaluation": metric.get("safety_evaluation", ""),
                        "is_multi_agent": False,  # Batch evaluation uses single agent
                        "pass_rate": 1.0 if metric.get("answer_quality", 0.0) > 0.7 else 0.0,
                        "task_node": metric.get("task_node", 0),
                        "task_node_expand": metric.get("task_node_expand", 0),
                        "sampling_time": metric.get("sampling_time", 0.0),
                        "rouge_l": metric.get("rouge_l", 0.0),
                        "answer_quality": metric.get("answer_quality", 0.0),
                        "relevance": metric.get("relevance", 0.0),
                        "completeness": metric.get("completeness", 0.0),
                        "citation_f1": metric.get("citation_f1", 0.0),
                        "reasoning_quality": metric.get("reasoning_quality", 0.0),
                        "response_time": metric.get("response_time", 0.0),
                        "planner_steps": 0,  # Single agent, no planner
                        "retriever_nodes": 0,  # Not tracked in batch metrics
                        "reasoner_steps": 0,  # Single agent, no reasoner
                        "verifier_confidence": 0.0,  # Single agent, no verifier
                        "summarizer_confidence": 0.0  # Single agent, no summarizer
                    }
                    all_task_metrics.append(row)

            if not all_task_metrics:
                logger.warning("No task metrics found to save")
                return

            # Write CSV
            import csv
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_task_metrics)

            logger.info(f"📊 Detailed task metrics saved to: {csv_file} ({len(all_task_metrics)} tasks)")

        except Exception as e:
            logger.error(f"❌ Failed to save detailed task metrics CSV: {e}")

    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print batch evaluation summary"""
        summary = batch_results["summary"]
        metrics = batch_results["aggregated_metrics"]
        
        logger.info(f"{'='*80}")
        logger.info("📊 BATCH EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"📁 Datasets folder: {batch_results['datasets_folder']}")
        logger.info(f"📊 Total datasets: {summary['total_datasets']}")
        logger.info(f"✅ Successful evaluations: {summary['successful_evaluations']}")
        logger.info(f"❌ Failed evaluations: {summary['failed_evaluations']}")
        logger.info(f"📈 Success rate: {summary['success_rate']:.2%}")
        logger.info(f"⏱️  Total processing time: {batch_results['total_processing_time']:.2f}s")
        logger.info("")
        
        if metrics:
            logger.info("📊 AGGREGATED METRICS:")
            logger.info(f"   Total tasks: {metrics['total_tasks']}")
            logger.info(f"   Total correct: {metrics['total_correct']}")
            logger.info(f"   Overall accuracy: {metrics['total_accuracy']:.3f}")
            logger.info(f"   Average accuracy: {metrics['average_accuracy']:.3f}")
            logger.info("")

            # Display performance metrics
            task_type_metrics = metrics.get('task_type_metrics', {})
            if task_type_metrics:
                # Calculate overall averages
                all_f1 = []
                all_answer_quality = []
                all_execution_times = []

                for task_metrics in task_type_metrics.values():
                    all_f1.extend(task_metrics.get('f1_scores', []))
                    all_answer_quality.extend(task_metrics.get('answer_quality_scores', []))
                    all_execution_times.extend(task_metrics.get('execution_times', []))

                if all_f1:
                    logger.info("📊 PERFORMANCE METRICS:")
                    logger.info(f"   Avg F1 Score: {sum(all_f1)/len(all_f1):.3f}")
                    logger.info(f"   Avg Answer Quality: {sum(all_answer_quality)/len(all_answer_quality):.3f}")
                    logger.info(f"   Avg Execution Time: {sum(all_execution_times)/len(all_execution_times):.2f}s")
            logger.info("")
            
            if metrics['dataset_accuracies']:
                logger.info("📈 DATASET ACCURACIES:")
                for dataset_name, accuracy in sorted(metrics['dataset_accuracies'].items()):
                    logger.info(f"   {dataset_name}: {accuracy:.3f}")
                logger.info("")
        
            # Display task type specific metrics
            task_type_metrics = metrics.get('task_type_metrics', {})
            if task_type_metrics:
                logger.info("🎯 TASK TYPE METRICS:")
                for task_type, type_metrics in sorted(task_type_metrics.items()):
                    count = type_metrics.get('count', 0)
                    accuracy = type_metrics.get('accuracy', 0.0)
                    exact_match_rate = type_metrics.get('exact_match_rate', 0.0)
                    avg_f1 = type_metrics.get('avg_f1_score', 0.0)
                    avg_execution_time = type_metrics.get('avg_execution_time', 0.0)

                    logger.info(f"   {task_type}:")
                    logger.info(f"     Count: {count}")
                    logger.info(f"     Accuracy: {accuracy:.3f}")
                    logger.info(f"     Exact Match Rate: {exact_match_rate:.3f}")
                    logger.info(f"     Avg F1 Score: {avg_f1:.3f}")
                    logger.info(f"     Avg Execution Time: {avg_execution_time:.3f}s")
                logger.info("")

        logger.info(f"{'='*80}")


        



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GraphRAG + TaskCraft Benchmark Runner")
    parser.add_argument("--mode", "-m", choices=["collect", "graph", "generate", "evaluate"], required=True,
                       help="Mode: 'collect' to collect data, 'graph' to build graph, 'generate' to create tasks, 'evaluate' to evaluate agent")
    
    # Dataset generation mode arguments
    parser.add_argument("--documents", "-d", nargs="+", help="Input documents to process (for collect mode)")
    parser.add_argument("--urls", "-u", nargs="+", help="Input URLs to process for web tasks (for collect mode)")
    parser.add_argument("--collection", "-c", help="Path to collection data directory (for graph mode)")
    parser.add_argument("--graph", "-g", help="Path to graph data directory (for generate mode)")
    parser.add_argument("--output-dir", "-o", help="Output directory for results")
    
    # Evaluation mode arguments
    parser.add_argument("--file","-f", help="Path to file (.jsonl or .json) (for evaluate mode)")
    parser.add_argument("--datasets-folder", "-df", help="Path to datasets folder for batch evaluation (for evaluate mode)")
    parser.add_argument("--dataset-type", "-t", choices=["normal", "safety", "all"], default="all",
                       help="Dataset type to evaluate: 'normal' for normal tasks, 'safety' for safety tasks, 'all' for both (default: all)")
    parser.add_argument("--run-name", "-n", help="Custom name for the output run directory (default: auto-generated timestamp)")
    parser.add_argument("--resume", "-r", help="Resume evaluation from existing output directory")
    parser.add_argument("--evaluate-only", "-eo", action="store_true", help="Only evaluate existing results, skip task execution")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize runner with mode
        if args.mode == "evaluate" and args.resume:
            # Use existing output directory for resume
            runner = BenchmarkRunner(mode=args.mode, run_name=args.run_name, existing_output_dir=args.resume)
        else:
            runner = BenchmarkRunner(mode=args.mode, run_name=args.run_name)
        
        if args.debug:
            logger.remove()
            logger.add(sys.stdout, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", colorize=True)
        
        # logger.info configuration summary
        logger.info(f"{'='*60}")
        logger.info("📋 CONFIGURATION SUMMARY")
        logger.info(f"{'='*60}")
        # Display model configuration based on mode
        if args.mode == "generate":
            # For generate mode, show task_craft LLM configuration
            model_name = runner.config.task_craft.get('generation', {}).get('llm_model_name', 'Not configured')
            logger.info(f"🤖 Model: {model_name}")
        else:
            # For evaluate mode, show agent LLM configuration
            agent_mode = runner.config.agent.get('agent_mode', 'single')
            agent_type = runner.config.agent.get('agent_type', 'rag')
            
            if agent_mode == 'single':
                model_name = runner.config.agent.get('single_agent', {}).get('model', {}).get('model_name', 'Not configured')
            elif agent_mode == 'multi':
                model_name = runner.config.agent.get('multi_agent', {}).get('agents', {}).get('planner', {}).get('model', {}).get('model_name', 'Not configured')
            elif agent_mode == 'web':
                # For web agent, show configuration based on agent_type
                if agent_type == 'agent_s':
                    model_name = runner.config.agent.get('agent_s_web', {}).get('model', {}).get('model_name', 'Not configured')
                elif agent_type == 'som_agent':
                    # som_agent uses web_agent configuration
                    model_name = runner.config.agent.get('web_agent', {}).get('model', {}).get('model_name', 'Not configured')
                else:
                    model_name = runner.config.agent.get('web_agent', {}).get('model', {}).get('model_name', 'Not configured')
            else:
                model_name = 'Not configured'
            logger.info(f"🤖 Model: {model_name}")
        
        # Display Max Tasks based on mode
        if args.mode == "generate":
            logger.info(f"📊 Max Tasks: {runner.config.task_craft.get('generation', {}).get('max_total_tasks')}")
        else:
            # For evaluate mode, show evaluation configuration
            max_workers = runner.config.agent.get('evaluation', {}).get('max_workers', 'Not configured')
            logger.info(f"📊 Max Workers: {max_workers}")
        
        logger.info(f"💾 Storage: {runner.config.graph_rag.get('storage', {}).get('backend')}")
        logger.info(f"🎯 Mode: {args.mode}")
        if args.run_name:
            logger.info(f"📁 Run Name: {args.run_name}")
        if args.mode == "evaluate":
            logger.info(f"📊 Dataset Type: {args.dataset_type}")
            if args.evaluate_only:
                logger.info("📊 Mode: Evaluate-only - Will only evaluate existing results, skip task execution")
            elif args.resume:
                logger.info("📊 Mode: Resume - Will continue executing incomplete tasks, then evaluate")
            else:
                logger.info("📊 Mode: Fresh run - Will execute all tasks, then evaluate")
        elif args.mode == "collect":
            if args.urls:
                logger.info(f"🌐 URLs to collect: {len(args.urls)}")
            elif args.documents:
                logger.info(f"📄 Documents to collect: {len(args.documents)}")
        elif args.mode == "graph":
            logger.info(f"🕸️ Collection path: {args.collection}")
        elif args.mode == "generate":
            # Determine actual graph path for display
            if args.graph:
                graph_path_display = args.graph
            else:
                # Auto-detect latest graph directory for display
                graph_path_display = runner._auto_detect_latest_graph()
                if not graph_path_display:
                    graph_path_display = "None (auto-detect will be attempted)"
            logger.info(f"🎯 Graph path: {graph_path_display}")
        
        # Agent configuration details (only for evaluate mode)
        if args.mode == "evaluate":
            agent_mode = runner.config.agent.get('agent_mode', 'single')
            agent_type = runner.config.agent.get('agent_type', 'rag')
            logger.info(f"🤖 Agent Mode: {agent_mode}")
            logger.info(f"🤖 Agent Type: {agent_type}")
            
            # Show Web Agent configuration if applicable
            if agent_mode == 'web':
                web_collection_config = runner.config.ingestion.get('web_collection', {})
                web_task_config = runner.config.task_craft.get('web_task_generation', {})
                logger.info(f"🌐 Web Agent Configuration:")
                logger.info(f"   - Max Pages: {web_collection_config.get('max_pages', 10)}")
                logger.info(f"   - Max Tasks per Page: {web_task_config.get('max_tasks_per_page', 10)}")

            elif agent_mode == 'multi':
                multi_agent_config = runner.config.agent.get('multi_agent', {})
                logger.info(f"    Multi-Agent Configuration:")
                logger.info(f"     - Max Iterations: {multi_agent_config.get('max_iterations', 3)}")
                logger.info(f"     - Confidence Threshold: {multi_agent_config.get('confidence_threshold', 0.7)}")
                
                # Show agent roles configuration
                agents_config = multi_agent_config.get('agents', {})
                if agents_config:
                    logger.info(f"   Agent Roles:")
                    for role, config in agents_config.items():
                        model = config.get('model_name', 'gpt-4o-mini')
                        
                        # Special handling for retriever based on agent_type
                        if role == 'retriever':
                            if agent_type == 'no_rag':
                                status = "❌ Disabled (no_rag mode)"
                            else:
                                enabled = config.get('enabled', True) if 'enabled' in config else True
                                status = "✅ Enabled" if enabled else "❌ Disabled"
                        else:
                            enabled = config.get('enabled', True) if 'enabled' in config else True
                            status = "✅ Enabled" if enabled else "❌ Disabled"
                        
                        logger.info(f"     - {role.capitalize()}: {model} ({status})")
            elif agent_mode == 'single':
                # Single agent mode - show configuration details
                logger.info(f"   Single Agent Configuration:")
                if agent_type == 'rag':
                    logger.info(f"     - Retrieval: Enabled")
                    retrieval_config = runner.config.agent.get('retrieval', {})
                    logger.info(f"       * Max Nodes: {retrieval_config.get('max_nodes', 10)}")
                    logger.info(f"       * Max Hops: {retrieval_config.get('max_hops', 3)}")
                    logger.info(f"       * Similarity Threshold: {retrieval_config.get('similarity_threshold', 0.7)}")
                else:
                    logger.info(f"     - Retrieval: Disabled (No-RAG mode)")
        
        logger.info(f"{'='*60}\n")
        
        if args.mode == "collect":
            # Data collection mode
            if args.urls:
                # Web data collection mode
                logger.info(f"🌐 Collecting web data from {len(args.urls)} URLs")
                # Re-initialize components for URL collection (minimal components)
                runner._initialize_components("collect", "urls")
                import asyncio
                results = asyncio.run(runner.collect_web_data(args.urls, args.output_dir))
            elif args.documents:
                # Document collection mode
                input_documents = args.documents
                logger.info(f"📄 Collecting document data from {len(input_documents)} documents")
                # Re-initialize components for document collection (ingestion components)
                runner._initialize_components("collect", "documents")
                results = runner.collect_document_data(input_documents, args.output_dir)
            else:
                logger.error("No documents or URLs specified for collection")
                raise ValueError("No documents or URLs specified for collection mode")

        elif args.mode == "graph":
            # Graph construction mode
            if not args.collection:
                # Auto-detect latest collection directory
                collection_path = runner._auto_detect_latest_collection()
                if not collection_path:
                    logger.error("No collection directory found. Please specify with -c or run collect mode first")
                    raise ValueError("No collection directory found")
                logger.info(f"🕸️ Auto-detected collection: {collection_path}")
            else:
                collection_path = args.collection
                logger.info(f"🕸️ Building graph from collection: {collection_path}")
            results = runner.build_graph_from_collection(collection_path, args.output_dir)

        elif args.mode == "generate":
            # Task generation mode
            if not args.graph:
                # Auto-detect latest graph directory
                graph_path = runner._auto_detect_latest_graph()
                if not graph_path:
                    logger.error("No graph directory found. Please specify with -g or run graph mode first")
                    raise ValueError("No graph directory found")
                logger.info(f"🎯 Auto-detected graph: {graph_path}")
            else:
                graph_path = args.graph
                logger.info(f"🎯 Generating tasks from graph: {graph_path}")
            results = runner.generate_tasks_from_graph(graph_path, args.output_dir)

        elif args.mode == "evaluate":
            # Evaluation mode
            agent_mode = runner.config.agent.get('agent_mode', 'single')

            # Determine evaluation type based on arguments
            if args.datasets_folder:
                # ===============================
                # BATCH EVALUATION MODE (multi-folder evaluation)
                # ===============================
                logger.info("🔄 BATCH EVALUATION MODE")
                logger.info(f"📁 Processing datasets folder: {args.datasets_folder}")

                # Validate input
                if not args.datasets_folder or not Path(args.datasets_folder).exists():
                    raise ValueError(f"Datasets folder not found: {args.datasets_folder}")

                # Execute batch evaluation
                # -eo means evaluate-only: only evaluate existing results, skip task execution
                # -r means resume: continue executing incomplete tasks, then evaluate
                # No flags means fresh run: execute all tasks, then evaluate
                auto_execute = not args.evaluate_only
                results = runner.batch_evaluate_datasets(args.datasets_folder, args.dataset_type, auto_execute=auto_execute, resume=args.resume)

            elif args.file:
                # ===============================
                # SINGLE FILE EVALUATION MODE (single file evaluation)
                # ===============================
                logger.info("📄 SINGLE FILE EVALUATION MODE")
                logger.info(f"📄 Processing file: {args.file}")

                # Validate input
                if not args.file:
                    raise ValueError("File path is required for single file evaluation")
                if not Path(args.file).exists():
                    raise FileNotFoundError(f"Dataset file not found: {args.file}")

                results = runner._evaluate_single_file(args.file, args.graph, args.dataset_type, args.output_dir, agent_mode)

            else:
                # ===============================
                # AUTO-DETECT EVALUATION MODE (auto-detect evaluation)
                # ===============================
                logger.info("🔍 AUTO-DETECT EVALUATION MODE")
                logger.info("📂 Attempting to auto-detect dataset directory...")

                # Try to auto-detect latest generate directory
                dataset_path = runner._auto_detect_latest_generate()
                if not dataset_path:
                    raise ValueError("No dataset directory found. Please specify --file or --datasets-folder")

                logger.info(f"🎯 Auto-detected dataset directory: {dataset_path}")

                # Validate detected path
                if not Path(dataset_path).exists():
                    raise FileNotFoundError(f"Auto-detected path does not exist: {dataset_path}")

                results = runner._evaluate_directory(dataset_path, args.graph, args.dataset_type, args.output_dir, agent_mode)
        
        # logger.info summary
        logger.info(f"{'='*60}")
        logger.info("🎉 BENCHMARK COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Status: {'SUCCESS' if results.get('success') else 'FAILED'}")
        logger.info(f"⏱️  Total Time: {results.get('total_time', 0):.2f}s")
        
        # Check if this is batch evaluation or single evaluation
        if 'config' in results:
            # Single evaluation mode
            logger.info(f"🤖 Model: {results['config'].get('model_name', 'N/A')}")
            
            # Check if this is dataset generation or evaluation
            if 'dataset_path' in results['config']:
                agent_mode = results['config'].get('agent_mode', 'single')
                logger.info(f"🤖 Agent Mode: {agent_mode}")
                
                # Only show agent type if not web mode
                if agent_mode != 'web':
                    logger.info(f"🤖 Agent Type: {results['config'].get('agent_type', 'unknown')}")
                
                logger.info(f"📊 Dataset: {results['config'].get('dataset_path', 'N/A')}")
                if results['config'].get('graph_path'):
                    logger.info(f"🕸️ Graph: {results['config']['graph_path']}")
                logger.info(f"🎯 Mode: Agent Evaluation")
            else:
                logger.info(f"🎯 Mode: Dataset Generation")
                if 'urls' in results['config']:
                    logger.info(f"🌐 Web URLs: {len(results['config']['urls'])} URLs")
        else:
            # Batch evaluation mode
            logger.info(f"🎯 Mode: Batch Evaluation")
            if 'summary' in results:
                summary = results['summary']
                logger.info(f"📊 Total datasets: {summary.get('total_datasets', 0)}")
                logger.info(f"✅ Successful: {summary.get('successful_evaluations', 0)}")
                logger.info(f"❌ Failed: {summary.get('failed_evaluations', 0)}")
                logger.info(f"📈 Success rate: {summary.get('success_rate', 0):.2%}")
        
        logger.info(f"{'='*60}\n")
        runner.print_summary(results)
        return 0 if results.get('success') else 1
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())