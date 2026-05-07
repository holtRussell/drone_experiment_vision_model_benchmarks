"""
Single-pass experiment runner
Processes images through vision pipelines, then sends to LLM
"""
import time
from typing import Dict, Any, List, Optional
from PIL import Image
from pathlib import Path

from src.data.loader import VisDroneLoader
from src.data.ground_truth import VisDroneGroundTruth
from src.pipelines import (
    YoloLocalPipeline,
    YoloMcpPipeline,
    VlmDirectPipeline,
    VlmMultiAgentPipeline
)
from src.controller.llm_client import LLMClient
from src.prompts.templates import PromptTemplates
from src.logging import TimingStats, StructuredLogger, ResourceMonitor, TimeSeriesMonitor
from src.cache import CacheManager
from src.representation import create_ir, validate_ir


PIPELINE_MAP = {
    "yolo_local": YoloLocalPipeline,
    "yolo_mcp": YoloMcpPipeline,
    "vlm_direct": VlmDirectPipeline,
    "vlm_multiagent": VlmMultiAgentPipeline,
}


class ExperimentRunner:
    """Single-pass experiment runner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mock_mode = self.config.get('experiment', {}).get('mock_llm', False)
        self.llm_client = LLMClient(mock_mode=self.mock_mode)
        self.prompts = PromptTemplates()
        self.ground_truth = VisDroneGroundTruth()
        self.timing = TimingStats()
        self.logger = StructuredLogger()
        self.monitor = ResourceMonitor()
        self.timeseries_monitor = TimeSeriesMonitor(sample_interval_ms=100)
        self.cache = CacheManager(
            enabled=config.get('experiment', {}).get('cache', {}).get('enabled', True)
        )
        
        self.results = []
    
    def get_pipeline(self, pipeline_name: str) -> Any:
        """Instantiate a pipeline by name"""
        pipeline_class = PIPELINE_MAP.get(pipeline_name)
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        # Pipeline config is in full_config['experiment']['pipeline_configs'][pipeline_name]
        pipeline_configs = self.config.get('experiment', {}).get('pipeline_configs', {})
        pipeline_config = pipeline_configs.get(pipeline_name, {})
        return pipeline_class(config=pipeline_config)
    
    def run_image(
        self,
        image_data: Dict[str, Any],
        pipeline_name: str,
        repetition: int = 0
    ) -> Dict[str, Any]:
        """
        Run single pass: Vision Pipeline -> IR -> LLM -> Output
        """
        image_id = image_data['image_id']
        image_path = image_data['image_path']
        
        result = {
            "image_id": image_id,
            "pipeline": pipeline_name,
            "repetition": repetition,
        }
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get ground truth
        if image_data.get('annotation_path'):
            gt = self.ground_truth.parse_annotation(image_data['annotation_path'])
            result['ground_truth'] = gt
        
        # Start time-series resource monitoring
        self.timeseries_monitor = TimeSeriesMonitor(sample_interval_ms=100)
        self.timeseries_monitor.start()
        
        # Step 1: Vision Processing
        self.timing.start('vision_processing')
        
        # Check vision cache
        cached_ir, cache_hit = self.cache.get_vision_cache(image_id, pipeline_name)
        
        if cache_hit:
            ir = cached_ir
        else:
            pipeline = self.get_pipeline(pipeline_name)
            ir = pipeline.process(image)
            self.cache.set_vision_cache(image_id, pipeline_name, ir)
        
        vision_latency = self.timing.stop('vision_processing')
        
        result['intermediate_representation'] = ir
        result['vision_latency_ms'] = vision_latency
        result['vision_cache_hit'] = cache_hit
        
        # Log vision metrics with enhanced IR data
        self._log_vision_metrics(image_id, pipeline_name, ir, vision_latency)
        
        # Step 2: LLM Processing (IR + Query)
        self.timing.start('llm_processing')
        
        # Atomic queries
        atomic_responses = {}
        for obj_type in ['cars', 'pedestrians', 'bicycles']:
            query = self.prompts.get_atomic_query(obj_type)
            ir_prompt = self.prompts.get_ir_prompt(ir, query)
            
            # Check LLM cache
            cached_response, llm_cache_hit = self.cache.get_llm_cache(ir_prompt)
            
            if cached_response:
                response = cached_response
            else:
                response = self.llm_client.request_text(ir_prompt, self.prompts.get_system_prompt())
                self.cache.set_llm_cache(ir_prompt, response)
            
            atomic_responses[obj_type] = response
        
        # Composite query
        composite_query = self.prompts.get_composite_query()
        composite_ir_prompt = self.prompts.get_ir_prompt(ir, composite_query)
        composite_response = self.llm_client.request_text(composite_ir_prompt, self.prompts.get_system_prompt())
        
        llm_latency = self.timing.stop('llm_processing')
        
        result['atomic_responses'] = atomic_responses
        result['composite_response'] = composite_response
        result['llm_latency_ms'] = llm_latency
        
        # Stop time-series monitoring and log results
        self.timeseries_monitor.stop()
        timeseries_samples = self.timeseries_monitor.get_samples()
        timeseries_summary = self.timeseries_monitor.get_summary()
        
        # Log time-series resources
        self.logger.log(
            event_type="timeseries_resources",
            data={
                "samples": timeseries_samples,
                "summary": timeseries_summary
            },
            image_id=image_id,
            pipeline=pipeline_name
        )
        
        # Resource usage (snapshot)
        result['resources'] = self.monitor.get_snapshot()
        result['timeseries_summary'] = timeseries_summary
        
        # Total latency
        result['total_latency_ms'] = vision_latency + llm_latency
        
        self.results.append(result)
        
        return result
    
    def _log_vision_metrics(
        self,
        image_id: str,
        pipeline: str,
        ir: Dict[str, Any],
        latency_ms: float
    ):
        """Log enhanced vision metrics from IR"""
        metadata = ir.get("metadata", {})
        detections = metadata.get("detections", [])
        
        # Calculate confidence stats
        confidence_scores = metadata.get("confidence_scores", [])
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Get per-class confidences
        avg_conf_per_class = metadata.get("avg_confidence_per_class", {})
        
        self.logger.log(
            event_type="vision_metrics",
            data={
                "cars": ir.get("cars", 0),
                "pedestrians": ir.get("pedestrians", 0),
                "bicycles": ir.get("bicycles", 0),
                "total_detections": len(detections),
                "confidence_scores": confidence_scores,
                "avg_confidence": avg_confidence,
                "avg_confidence_per_class": avg_conf_per_class,
                "image_resolution": metadata.get("image_resolution"),
                "model_input_size": metadata.get("model_input_size"),
                "latency_ms": latency_ms,
                "cache_hit": ir.get("cache_hit", False)
            },
            image_id=image_id,
            pipeline=pipeline
        )
    
    def run_experiment(self):
        """Run full experiment"""
        dataset_path = self.config['experiment']['dataset_path']
        num_images = self.config['experiment']['num_images']
        pipelines = self.config['experiment']['pipelines']
        repetitions = self.config['experiment']['repetitions']
        
        loader = VisDroneLoader(dataset_path)
        images = list(loader)[:num_images]
        
        for pipeline_name in pipelines:
            for img_data in images:
                for rep in range(repetitions):
                    print(f"Processing {img_data['image_id']} with {pipeline_name} (rep {rep+1}/{repetitions})")
                    self.run_image(img_data, pipeline_name, rep)
        
        return self.results
