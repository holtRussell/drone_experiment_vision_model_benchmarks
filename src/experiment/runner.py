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
        
        # Step 1: Vision Processing with separate monitoring
        self.vision_monitor = TimeSeriesMonitor(sample_interval_ms=100)
        self.vision_monitor.start()
        
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
        
        # Stop vision monitoring and log
        self.vision_monitor.stop()
        vision_samples = self.vision_monitor.get_samples()
        vision_summary = self.vision_monitor.get_summary()
        
        result['intermediate_representation'] = ir
        result['vision_latency_ms'] = vision_latency
        result['vision_cache_hit'] = cache_hit
        
        # Get ground truth
        ground_truth = result.get('ground_truth', {})
        
        # For VLM pipelines, don't log vision metrics yet (will log after VLM parsing)
        if pipeline_name not in ['vlm_direct', 'vlm_multiagent']:
            self._log_vision_metrics(image_id, pipeline_name, ir, vision_latency, ground_truth)
        
        # Log vision-only resource usage
        self.logger.log(
            event_type="vision_resources",
            data={
                "samples": vision_samples,
                "summary": vision_summary
            },
            image_id=image_id,
            pipeline=pipeline_name
        )
        
        # Step 2: LLM Processing (IR + Query) with separate monitoring
        self.llm_monitor = TimeSeriesMonitor(sample_interval_ms=100)
        self.llm_monitor.start()
        
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
        
        # Stop LLM monitoring and log
        self.llm_monitor.stop()
        llm_samples = self.llm_monitor.get_samples()
        llm_summary = self.llm_monitor.get_summary()
        
        result['atomic_responses'] = atomic_responses
        result['composite_response'] = composite_response
        result['llm_latency_ms'] = llm_latency
        
        # For VLM pipelines, create IR from atomic + composite responses
        if pipeline_name in ['vlm_direct', 'vlm_multiagent']:
            from src.representation.vlm_parser import parse_atomic_responses
            vlm_ir = parse_atomic_responses(
                atomic_responses=atomic_responses,
                composite_response=composite_response,
                pipeline_name=pipeline_name
            )
            # Update result with new IR
            result['intermediate_representation'] = vlm_ir
            # Update vision metrics with new IR
            vision_latency = vlm_ir.get('metadata', {}).get('latency_ms', 0) or vision_latency
            self._log_vision_metrics(image_id, pipeline_name, vlm_ir, vision_latency, ground_truth)
        else:
            # For YOLO pipelines, log with original IR
            self._log_vision_metrics(image_id, pipeline_name, ir, vision_latency, ground_truth)
        
        # Get token usage from LLM client
        token_usage = getattr(self.llm_client, 'last_token_usage', None)
        result['token_usage'] = token_usage
        
        # Log LLM-only resource usage
        self.logger.log(
            event_type="llm_resources",
            data={
                "samples": llm_samples,
                "summary": llm_summary
            },
            image_id=image_id,
            pipeline=pipeline_name
        )
        
        # Log LLM tokens
        if token_usage:
            self.logger.log(
                event_type="llm_tokens",
                data=token_usage,
                image_id=image_id,
                pipeline=pipeline_name
            )
        
        # Resource usage (snapshot)
        result['resources'] = self.monitor.get_snapshot()
        
        # Total latency
        result['total_latency_ms'] = vision_latency + llm_latency
        
        self.results.append(result)
        
        return result
    
    def _log_vision_metrics(
        self,
        image_id: str,
        pipeline: str,
        ir: Dict[str, Any],
        latency_ms: float,
        ground_truth: Optional[Dict[str, int]] = None
    ):
        """Log enhanced vision metrics from IR"""
        metadata = ir.get("metadata", {})
        detections = metadata.get("detections", [])
        
        # Calculate confidence stats
        confidence_scores = metadata.get("confidence_scores", [])
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Get per-class confidences
        avg_conf_per_class = metadata.get("avg_confidence_per_class", {})
        
        log_data = {
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
            "cache_hit": ir.get("cache_hit", False),
            "metadata": metadata  # Include full metadata for VLM parsing info
        }
        
        # Add ground truth if available (flatten into log data)
        if ground_truth:
            log_data.update({
                "gt_cars": ground_truth.get("cars", 0),
                "gt_pedestrians": ground_truth.get("pedestrians", 0),
                "gt_bicycles": ground_truth.get("bicycles", 0),
            })
        
        self.logger.log(
            event_type="vision_metrics",
            data=log_data,
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
