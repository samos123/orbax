# Deep Dive: Orbax Colocated Checkpoint Loading

## 1. Overview

In large-scale machine learning, loading multi-terabyte checkpoints can become a significant bottleneck if performed solely by the driver or a single host. "Colocated checkpoint loading" in Orbax addresses this by distributing the file I/O and preprocessing work across the CPUs of the worker machines that are physically attached to (colocated with) the accelerator devices (TPUs/GPUs).

This mechanism allows Orbax to:
1.  **Parallelize I/O**: Read data from storage (GCS, etc.) comfortably in parallel across all available hosts.
2.  **Avoid Device Bottlenecks**: Perform decompression and array restoration on the CPU before moving data to the expensive accelerator memory.
3.  **Minimize Latency**: Keep data local to the machine where it will eventually be used.

## 2. Architecture Components

The implementation relies on three main components working in concert:

1.  **`ColocatedPythonDispatcher`** (`dispatchers.py`): The orchestrator that wraps function calls to execute on colocated CPUs.
2.  **`jax.experimental.colocated_python`**: The underlying JAX primitive that enables running Python code on specific mesh devices (specifically CPUs).
3.  **`ArrayHandler`** (`jax_array_handlers.py`): The high-level component that decides *when* to use the dispatcher and invokes it during `deserialize`.

## 3. Detailed Implementation Walkthrough

### Step 1: Dispatcher Selection

The entry point for using colocated loading is typically strictly configuration-driven. In `pathways_handler_registry.py`, the `get_pathways_array_handler` function factory initializes the dispatcher if the `CheckpointingImpl.COLOCATED_PYTHON` option is selected.

```python
# orbax/checkpoint/_src/serialization/pathways_handler_registry.py

def get_pathways_array_handler(..., checkpointing_impl=None, ...):
    checkpointing_impl = checkpointing_impl or CheckpointingImpl.from_options(
        use_colocated_python=True,
    )
    match checkpointing_impl:
        case CheckpointingImpl.COLOCATED_PYTHON:
            logging.info('Using ColocatedPythonDispatcher')
            dispatcher = dispatchers.ColocatedPythonDispatcher()
        # ...
    
    return _get_array_hander_with_dispatcher(dispatcher, ...)
```

### Step 2: The Core Mechanism (`ColocatedPythonDispatcher`)

This class resides in `orbax/checkpoint/_src/multihost/dispatchers.py`. It implements the `Dispatcher` interface.

#### 2.1 Sharding Conversion (`_colocated_cpu_sharding`)
Before execution, the dispatcher must translate the *device* sharding (e.g., TPU mesh) into a corresponding *host* sharding (CPU mesh). This ensures that the Python function runs on the CPU responsible for the corresponding TPU.

```python
# orbax/checkpoint/_src/multihost/dispatchers.py

def _colocated_cpu_sharding(self, sharding: jax.sharding.Sharding) -> jax.sharding.Sharding:
    if isinstance(sharding, jax.sharding.SingleDeviceSharding):
        # Maps TPU device -> Colocated CPU device
        cpu_devices = cp.colocated_cpu_devices(list(sharding.device_set))
        return jax.sharding.SingleDeviceSharding(
            cpu_devices[0], memory_kind=sharding.memory_kind
        )
    elif isinstance(sharding, jax.sharding.NamedSharding):
        # Maps TPU Mesh -> Colocated CPU Mesh
        cpu_mesh = cp.colocated_cpu_devices(sharding.mesh)
        return jax.sharding.NamedSharding(
            cpu_mesh, sharding.spec, memory_kind=sharding.memory_kind
        )
```

#### 2.2 The Dispatch Method
The `dispatch` method executes a function `func` on the worker CPUs. It handles:
1.  **Input Transfer**: Moving arguments from the driver/wherever they are to the worker CPUs.
2.  **Execution**: Wrapping the function with `@cp.colocated_python`.
3.  **Output Spec**: Defining where the results should go (back to TPU).

```python
# orbax/checkpoint/_src/multihost/dispatchers.py

def dispatch(self, func, *, input_arrays=None, result_specs=None, func_kwargs=None, ...):
    # 1. Prepare dummy input to determine WHICH devices to run on
    if input_arrays is None:
        input_arrays = _get_dummy_input_array_from_result_specs(result_specs)

    # 2. Transform args/kwargs to live on CPU
    cpu_kwargs = self._transform_pytree_shardings(func_kwargs)

    # 3. Define the wrapper using JAX's colocated_python decorator
    @cp.colocated_python
    def _cp_wrapper(inp: PyTree) -> PyTree:
        # This code runs ON THE WORKER CPU
        _vlog_dispatch(func, 'ColocatedPythonDispatcher')
        # ... call the actual function ...
        return func(*args, **cpu_kwargs)

    # 4. Prepare result specs (where the output should end up eventually)
    cpu_result_specs = self._transform_pytree_shardings(result_specs)
    _cp_wrapper.specialize(out_specs_fn=lambda _: cpu_result_specs)

    # 5. Execute: Transfer inputs -> Run Wrapper -> Get Layout-Constrained Output
    result = _cp_wrapper(self.to_colocated_python(input_arrays))
    
    # 6. Final Transfer: Move from CPU to TPU
    # The result from _cp_wrapper is on CPU (because of cpu_result_specs).
    # We now push it to the final requested device/sharding (e.g. TPU).
    return self._to_final_specs(result, result_specs)
```

### Step 3: ArrayHandler Integration

In `orbax/checkpoint/_src/serialization/jax_array_handlers.py`, `ArrayHandler.deserialize` checks if a dispatcher is present. If so, it delegates the entire deserialization logic to it.

```python
# orbax/checkpoint/_src/serialization/jax_array_handlers.py

async def deserialize(self, infos, args=None):
    # ... metadata reading ...
    
    if self._dispatcher is None:
        # Standard local/driver-based load
        ret = await _deserialize_arrays(...)
    else:
        # DISPATCHED LOAD
        
        # 1. Update restore args if needed
        args = await self._maybe_read_metadata_and_update_restore_args(infos, args)
        
        # 2. Construct abstract arrays (ShapeDtypeStruct) representing the FINAL TPU placement
        result_specs = _get_abstract_arrays(args, shardings)
        
        # 3. Dispatch the _sync_deserialize_arrays function
        # This function will run on the worker CPUs!
        ret = self._dispatcher.dispatch(
            _sync_deserialize_arrays, 
            result_specs=result_specs,
            func_kwargs={
                'infos': infos,
                'args': args,
                'shardings': shardings, # These will be converted to CPU shardings by the dispatcher
                'metadata_key': self._metadata_key,
                'array_metadata_store': self._array_metadata_store,
            },
        )
        # 4. Block until the distributed computation is ready
        jax.block_until_ready(ret)
        
    return ret
```

### Step 4: Worker-Side Execution (`_sync_deserialize_arrays`)

The function actually running on the worker CPU is `_sync_deserialize_arrays`. It is a synchronous wrapper around the asyncio standard deserialization logic.

```python
# orbax/checkpoint/_src/serialization/jax_array_handlers.py

def _sync_deserialize_arrays(infos, args, shardings, ...):
    return asyncio_utils.run_sync(
        _deserialize_arrays(
            infos,
            args,
            shardings, # Note: These are now CPU shardings!
            metadata_key,
            array_metadata_store,
        )
    )
```

**Crucial Detail**: When `_deserialize_arrays` runs inside the dispatched context:
1.  The `shardings` passed to it have been transformed by `dispatcher._transform_pytree_shardings` to be **CPU shardings**.
2.  `tensorstore` operations (inside `_deserialize_arrays`) thus read data directly into host memory on that specific worker node.
3.  Because the function is running under `@cp.colocated_python`, JAX knows this is local to that process group.

## Summary of Data Flow

1.  **Driver**: Calls `ArrayHandler.deserialize`.
2.  **Driver**: `ColocatedPythonDispatcher` calculates CPU mesh from TPU mesh.
3.  **Driver**: `dispatch` triggers `@cp.colocated_python`.
4.  **Worker CPU**: `_sync_deserialize_arrays` runs.
5.  **Worker CPU**: `TensorStore` reads checkpoint bytes from storage -> Worker RAM.
6.  **Worker CPU**: `ColocatedPythonDispatcher` receives the CPU arrays.
7.  **Transfer**: `_to_final_specs` moves data from Worker RAM -> Worker TPU HBM (`jax.device_put`).

## 4. Memory Usage & Streaming Behavior

**Key Finding**: The current implementation is **Buffered (per-shard)**, not Streamed.

### Behavior
When loading a checkpoint using this mechanism, the system performs a **full materialization** of the host-local shard into Host RAM before transferring it to the accelerator (TPU/GPU).

1.  **Read Phase**: The `TensorStore` operations in `_deserialize_arrays` allocate a NumPy array (`np.zeros`) large enough to hold the **entire local shard** for each parameter.
2.  **Buffer Phase**: Data is read from storage into this host-memory buffer.
3.  **Transfer Phase**: The entire buffer is moved to the accelerator using `jax.device_put`.
4.  **Cleanup**: The host-memory buffer is released only after the `jax.Array` has been created.

### Implications
-   **Host RAM Requirement**: The worker machine must have enough CPU RAM to hold the *largest* local shard of the model parameters. 
-   **No "Chunked" Transfer**: Data is not streamed in small chunks (e.g., 100MB at a time) through to the TPU. It is "store-and-forward" at the granularity of the full shard.
-   **Concurrency**: While individual shards are buffered, `asyncio.gather` is used to process multiple parameters concurrently, which can increase peak host memory usage if many large parameters are loaded simultaneously.

## 5. Potential Improvements: Enabling True Streaming

To answer the question *"Would it be possible to do some form of streaming where we release memory after each tensorstore spec?"*: **Yes, but it requires code changes.**

### The Bottleneck
Currently, `ArrayHandler.deserialize` invokes the dispatcher once for all requested parameters. The dispatcher (and the underlying `asyncio.gather` on the worker) waits for **all** arrays to be materialized in host memory before returning the list of `jax.Array`s to the driver. This accumulates peak memory usage equal to the sum of all local shards in the batch.

### Solution Strategy
To achieve "streaming" (or at least granular batching) where memory is released after each parameter (or small group):

1.  **Batching at the Driver**: The `ArrayHandler.deserialize` method should break the list of `infos` (parameters) into smaller chunks (e.g., 1 parameter at a time, or 1GB batches).
2.  **Sequential Dispatch**: It should call `self._dispatcher.dispatch(...)` sequentially for each chunk.
3.  **Early Release**: After each dispatch call returns, the resulting `jax.Array`s (which are now effectively on the TPU, assuming the dispatcher moved them there) can have their CPU references dropped or handled by JAX's memory management.

Basically, instead of:
```python
# Current: All-at-once
all_results = dispatcher.dispatch(deserialize_all, args=all_params)
```
We would need:
```python
# Proposed: Streaming/Batching
all_results = []
for params_batch in batched(all_params):
    batch_results = dispatcher.dispatch(deserialize_batch, args=params_batch)
    all_results.extend(batch_results)
    # At this point, Host RAM for batch_results should be releasable 
    # as the data is already transferred to TPU by the dispatcher's output logic.
```
This would cap peak Host RAM usage to the size of the largest single batch (or single parameter).
