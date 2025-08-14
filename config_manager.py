"""
Simplified configuration management system - Unify multiple YAML configuration files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Simple logging functions
def log_info(msg):
    print(f"[INFO] {msg}")

def log_warning(msg):
    print(f"[WARNING] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration"""
    
    # Project information
    project: Dict[str, Any]
    
    # API configuration
    apis: Dict[str, Any]
    
    # Path configuration
    paths: Dict[str, Any]
    
    # Global settings
    global_config: Dict[str, Any]
    
    # Orchestration settings
    orchestration: Dict[str, Any]
    
    # Module configurations
    ingestion: Dict[str, Any]
    graph_rag: Dict[str, Any]
    task_craft: Dict[str, Any]
    agent: Dict[str, Any]
    evaluation: Dict[str, Any]
    safety: Dict[str, Any]
    datasets: Dict[str, Any]
    finetuning: Dict[str, Any]


class ConfigManager:
    """Configuration manager - Load and manage all configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config = None
        # Load .env file first
        self._load_env_file()
        self._load_all_configs()
        self._load_environment_variables()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load main configuration
            main_config = self._load_yaml_file("main_config.yaml")
            
            # Load module configurations
            configs = {
                "project": main_config.get("project", {}),
                "apis": main_config.get("apis", {}),
                "paths": main_config.get("paths", {}),
                "global_config": main_config.get("global", {}),
                "orchestration": main_config.get("orchestration", {}),
                "ingestion": self._load_yaml_file("ingestion_config.yaml"),
                "graph_rag": self._load_yaml_file("graph_rag_config.yaml"),
                "task_craft": self._load_yaml_file("task_craft_config.yaml"),
                "agent": self._load_yaml_file("agent_config.yaml"),
                "evaluation": self._load_yaml_file("evaluation_config.yaml"),
                "safety": self._load_yaml_file("safety_config.yaml"),
                "datasets": self._load_yaml_file("datasets_config.yaml"),
                "finetuning": self._load_yaml_file("finetuning_config.yaml")
            }
            
            self.config = BenchmarkConfig(**configs)
            log_info("All config.yaml loaded successfully.")
            
        except Exception as e:
            log_error(f"Configuration loading failed: {e}")
            self._create_default_config()
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a single YAML file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            log_warning(f"Configuration file does not exist: {filename}, using default configuration")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log_error(f"Failed to load configuration file {filename}: {e}")
            return {}
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            log_info("Loaded environment variables from .env file")
        else:
            log_warning(".env file not found, using system environment variables")
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config = BenchmarkConfig(
            project={
                "name": "GraphRAG TaskCraft Benchmark",
                "version": "1.0.0",
                "description": "Universal agent framework"
            },
            apis={
                "openai": {"api_key": None, "base_url": None},
                "anthropic": {"api_key": None, "base_url": None},
                "huggingface": {"api_key": None, "cache_dir": "models/huggingface"}
            },
            paths={
                "data_dir": "data",
                "logs_dir": "logs", 
                "models_dir": "models",
                "outputs_dir": "outputs",
                "cache_dir": "cache"
            },
            global_config={
                "random_seed": 42,
                "debug_mode": False,
                "max_workers": 4
            },
            ingestion={},
            graph_rag={},
            task_craft={},
            agent={},
            evaluation={},
            safety={},
            datasets={},
            finetuning={}
        )
    
    def _load_environment_variables(self):
        """Load sensitive configuration from environment variables"""
        
        # API keys and URLs
        if self.config.apis.get("openai"):
            self.config.apis["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            self.config.apis["openai"]["base_url"] = os.getenv("OPENAI_BASE_URL")
            self.config.apis["openai"]["organization"] = os.getenv("OPENAI_ORGANIZATION")
            
        if self.config.apis.get("anthropic"):
            self.config.apis["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
            self.config.apis["anthropic"]["base_url"] = os.getenv("ANTHROPIC_BASE_URL")
            
        if self.config.apis.get("huggingface"):
            self.config.apis["huggingface"]["api_key"] = os.getenv("HUGGINGFACE_API_KEY")
            self.config.apis["huggingface"]["cache_dir"] = os.getenv("HUGGINGFACE_CACHE_DIR", "models/huggingface")
        
        # Environment variable overrides
        env_overrides = {
            "DEBUG_MODE": ("global_config", "debug_mode"),
            "LOG_LEVEL": ("orchestration", "monitoring", "log_level"),
            "BATCH_SIZE": ("orchestration", "batch_processing", "batch_size"),
            "MAX_TOKENS": ("agent", "execution", "max_tokens"),
            "MODEL_NAME": ("agent", "execution", "model_name"),
        }
        
        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_config(config_path, env_value)
    
    def _set_nested_config(self, path: tuple, value: Any):
        """Set nested configuration value"""
        try:
            config_dict = getattr(self.config, path[0])
            
            # Convert value type
            if isinstance(value, str):
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
            
            # Set nested value
            current = config_dict
            for key in path[1:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[path[-1]] = value
            log_info(f"Environment variable override: {'.'.join(path)} = {value}")
            
        except Exception as e:
            log_warning(f"Failed to set environment variable {path}: {e}")
    
    def get_config(self) -> BenchmarkConfig:
        """Get complete configuration"""
        return self.config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get specific module configuration"""
        return getattr(self.config, section_name, {})
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration"""
        issues = []
        warnings = []
        
        # 检查API密钥
        if not self.config.apis.get("openai", {}).get("api_key") and not self.config.apis.get("anthropic", {}).get("api_key"):
            warnings.append("没有设置LLM API密钥，将使用模拟模式")
        
        # 检查路径
        for path_name, path_value in self.config.paths.items():
            path_obj = Path(path_value)
            if not path_obj.parent.exists():
                warnings.append(f"路径的父目录不存在: {path_value}")
        
        # 检查数值范围
        if hasattr(self.config, 'agent') and isinstance(self.config.agent, dict):
            execution_config = self.config.agent.get('execution', {})
            temperature = execution_config.get('temperature', 0.1)
            if not 0 <= temperature <= 2:
                issues.append("Agent temperature 应该在 0-2 之间")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 50)
        print("Configuration Summary")
        print("=" * 50)
        
        print(f"Project: {self.config.project.get('name', 'Unknown')} v{self.config.project.get('version', '1.0')}")
        
        # API Status
        api_status = []
        for api_name, api_config in self.config.apis.items():
            has_key = bool(api_config.get('api_key'))
            api_status.append(f"{api_name}: {'✅' if has_key else '❌'}")
        print(f"API Status: {', '.join(api_status)}")
        
        # Module Status
        modules = ['ingestion', 'graph_rag', 'task_craft', 'agent', 'evaluation', 'safety']
        enabled_modules = []
        for module in modules:
            module_config = getattr(self.config, module, {})
            if module_config:
                enabled_modules.append(module)
        print(f"Enabled Modules: {', '.join(enabled_modules)}")
        
        # Validation Results
        validation = self.validate_config()
        if validation['valid']:
            print("Configuration Validation: ✅ Passed")
        else:
            print("Configuration Validation: ❌ Failed")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_dir: str = "configs") -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    
    return _config_manager

def get_config(config_dir: str = "configs") -> BenchmarkConfig:
    """Get benchmark configuration"""
    return get_config_manager(config_dir).get_config()


if __name__ == "__main__":
    # Test configuration system
    config_manager = ConfigManager()
    config_manager.print_config_summary()
    
    # Get module configuration examples
    print("\n" + "=" * 50)
    print("Module Configuration Examples")
    print("=" * 50)
    
    agent_config = config_manager.get_section('agent')
    print(f"Agent Configuration: {agent_config}")
    
    ingestion_config = config_manager.get_section('ingestion')
    print(f"Ingestion Configuration: {ingestion_config}")
    
    print("\nConfiguration system test completed ✅")
