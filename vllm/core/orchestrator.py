import os
import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray


if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    
    
class Orchestrator:
    ''' The Orchestrator class is responsible for managing the distributed cluster and workers.
        In the case of multiple workers, the cluster is initialized with Ray 
    '''
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # initialize the cluster
        self.distributed_init_method: str = ''
        self.placement_group: "PlacementGroup" = None
        self.workers = []
    
    def initialize_cluster(self) -> None:
        self.distributed_init_method, self.placement_group = initialize_cluster(
            self.parallel_config)
        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(self.placement_group)
        else:
            self._init_workers(self.distributed_init_method)
    
    def _init_workers(self, distributed_init_method: str) -> None:
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self._set_worker_ids()
        self.run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self.run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
            kv_buffer_size=self.cache_config.kv_buffer_size,
            max_kv_slots=self.cache_config.max_kv_slots,
        )
    
    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        
        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            if self.parallel_config.tensor_parallel_size == 1:
                num_gpus = self.cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self.run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                              model_config,
                              parallel_config,
                              scheduler_config,
                              None,
                              None,
                          ))
        self._set_worker_ids()
        self.run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self.run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
            kv_buffer_size=self.cache_config.kv_buffer_size,
            max_kv_slots=self.cache_config.max_kv_slots,
        )
    
    def _set_worker_ids(self) -> None:
        all_outputs = []
        for i, worker in enumerate(self.workers):
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, 'set_worker_id')
                output = executor(i)
                all_outputs.append(output)
            else:
                executor = getattr(worker, 'set_worker_id')
                output = executor(i)
                all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
    
    def _run_workers_in_batch(
        self,
        workers,
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        return all_outputs
    
    def run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                self.workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [self.workers]

        for workers in work_groups:
            all_outputs.extend(
                self._run_workers_in_batch(workers, method, *args, **kwargs))

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
    
    def run_workers_bool_all(
        self,
        method: str,
        *args,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        all_outputs = self.run_workers(
            method,
            *args,
            get_all_outputs=True,
            max_concurrent_workers=max_concurrent_workers,
            **kwargs,
        )
        assert all(isinstance(x, bool) for x in all_outputs)
        return all(all_outputs)

    def run_workers_bool_any(
        self,
        method: str,
        *args,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        all_outputs = self.run_workers(
            method,
            *args,
            get_all_outputs=True,
            max_concurrent_workers=max_concurrent_workers,
            **kwargs,
        )
        assert all(isinstance(x, bool) for x in all_outputs)
        return any(all_outputs)