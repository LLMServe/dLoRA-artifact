from typing import List, Optional, Union, Any
import copy
from functools import partial
import concurrent.futures
import time
import torch

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.config import ExecType

from vllm.engine.ray_utils import RayWorker, ray
from ray.air.util.torch_dist import init_torch_dist_process_group


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        num_groups: Number of groups(engines).
        num_model_per_group: Number of models in each group(engine).
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        pipeline_parallel_size: The number of GPUs to use for distributed
            execution with pipeline parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        seed: The seed to initialize the random number generator for sampling.
        exec_type: Execution type.
    """

    def __init__(
        self,
        model: str,
        num_groups: int,
        num_model_per_group: int = 1,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        exec_type: int = 3,
        **kwargs,
    ) -> None:
        self.llm_engine: List[LLMEngine] = []
        self.num_groups = num_groups
        self.exec_type = ExecType(exec_type)
        self.engine_id = 0
        self.num_model_per_group = num_model_per_group
        engine_init_active_mapping = {}
        for i in range(num_groups):
            engine_args = EngineArgs(
                model=model,
                tokenizer=tokenizer,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                num_groups=num_groups,
                num_models=num_model_per_group,
                dtype=dtype,
                seed=seed,
                exec_type=exec_type,
                num_model_per_group=num_model_per_group,
                **kwargs,
            )
            self.llm_engine.append(LLMEngine.from_engine_args(engine_args, i))
            start_model_id = i * engine_args.gpu_capacity
            engine_init_active = []
            for j in range(engine_args.gpu_capacity):
                engine_init_active.append((start_model_id + j) % num_model_per_group)
            engine_init_active_mapping[i] = engine_init_active

        if engine_args.worker_use_ray:
            workers = []
            for engine in self.llm_engine:
                all_workers = engine._init_workers_ray()
                workers.extend(all_workers)
            init_torch_dist_process_group(workers, backend="nccl")
            from vllm.worker.worker import Worker

            model_config = copy.deepcopy(self.llm_engine[0].model_config)
            parallel_config = copy.deepcopy(self.llm_engine[0].parallel_config)
            scheduler_config = copy.deepcopy(self.llm_engine[0].scheduler_config)
            lora_config = copy.deepcopy(self.llm_engine[0].lora_config)
            self._run_workers(workers,
                            engine_args,
                            "init_worker",
                            get_all_outputs=True,
                            worker_init_fn=lambda: Worker(
                                model_config,
                                parallel_config,
                                scheduler_config,
                                lora_config,
                                None,
                                None,
                                exec_type=self.exec_type,
                                num_model_per_group=self.num_model_per_group,
                            ))
            self._run_workers(
                workers,
                engine_args,
                "init_model",
                engine_id=0,
                get_all_outputs=True,
            )
        else:
            for engine in self.llm_engine:
                engine.active = engine_init_active_mapping[engine.engine_id]
                engine._init_workers(engine.distributed_init_method)

        for engine in self.llm_engine:
            engine.init_cont(engine._get_available_gpu_memory()[0], [i for i in range(num_model_per_group)])
            
        self.request_counter = Counter()

    def _run_workers(
        self,
        workers: List[RayWorker],
        engine_args: EngineArgs,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in workers:
            if engine_args.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if engine_args.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine[0].tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        for engine in self.llm_engine:
            engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        model_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            model_ids: A list of targeted model IDs of prompts. If None, we
                use model 0.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            model_id = model_ids[i] if model_ids is not None else 0
            self._add_request(prompt, sampling_params, token_ids, model_id)
        return self._run_engines(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        model_id: int,
    ) -> None:
        request_id = str(next(self.request_counter))
        if self.exec_type != ExecType.LORA:
            engine_id = model_id // self.num_model_per_group
        else:
            engine_id = self.engine_id
            self.engine_id = (self.engine_id + 1) % self.num_groups
        self.llm_engine[engine_id].add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids, model_id=model_id % self.num_model_per_group)
        
    def _run_engines(self, use_tqdm: bool):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [executor.submit(self._run_engine, use_tqdm=use_tqdm, engine_id=i) for i in range(self.num_groups)]
            concurrent.futures.wait(tasks)

    def _run_engine(self, use_tqdm: bool,
                    engine_id: int = 0) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine[engine_id].get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine[engine_id].has_unfinished_requests():
            step_outputs, _ = self.llm_engine[engine_id].step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        print(engine_id, f"Avg JCT: {self.llm_engine[engine_id]._get_avg_jct():.2f}s")
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
