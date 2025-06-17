import time
import logging
from contextlib import contextmanager
from typing import Dict, Optional
import threading

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics for agent operations"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    @contextmanager
    def start_operation(self, agent_name: str, operation: str):
        """Context manager for tracking operation performance"""
        operation_id = f"{agent_name}_{operation}_{int(time.time())}"
        start_time = time.time()
        
        class OperationContext:
            def __init__(self, monitor, op_id):
                self.monitor = monitor
                self.operation_id = op_id
                self.tokens_generated = 0
                self.success = True
                self.error_message = None
            
            def set_tokens_generated(self, tokens: int):
                self.tokens_generated = tokens
            
            def set_error(self, error: str):
                self.success = False
                self.error_message = error
        
        context = OperationContext(self, operation_id)
        
        try:
            yield context
        except Exception as e:
            context.set_error(str(e))
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self.lock:
                if agent_name not in self.metrics:
                    self.metrics[agent_name] = []
                
                self.metrics[agent_name].append({
                    "operation": operation,
                    "duration": duration,
                    "tokens_generated": context.tokens_generated,
                    "success": context.success,
                    "error": context.error_message,
                    "timestamp": start_time
                })
            
            logger.info(f"{agent_name}.{operation} completed in {duration:.2f}s "
                       f"({context.tokens_generated} tokens)")
    
    def get_metrics(self, agent_name: Optional[str] = None) -> Dict:
        """Get performance metrics"""
        with self.lock:
            if agent_name:
                return self.metrics.get(agent_name, [])
            return self.metrics.copy()
    
    def get_summary(self) -> str:
        """Get a summary of performance metrics"""
        with self.lock:
            summary = "Performance Summary:\n"
            
            for agent_name, operations in self.metrics.items():
                total_ops = len(operations)
                successful_ops = sum(1 for op in operations if op["success"])
                total_time = sum(op["duration"] for op in operations)
                total_tokens = sum(op["tokens_generated"] for op in operations)
                
                summary += f"\n{agent_name}:\n"
                summary += f"  Operations: {successful_ops}/{total_ops} successful\n"
                summary += f"  Total Time: {total_time:.2f}s\n"
                summary += f"  Total Tokens: {total_tokens}\n"
                
                if total_time > 0:
                    summary += f"  Avg Speed: {total_tokens/total_time:.1f} tokens/sec\n"
            
            return summary

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
