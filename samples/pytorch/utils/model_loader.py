import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import concurrent.futures


class ModelLoader(nn.Module):
    def __init__(self, use_multithreading=False):
        super(ModelLoader, self).__init__()
        self.models = nn.ModuleList()
        self.chained = False
        self.use_multithreading = use_multithreading

    def add_models(self, *models, chained=False):
        """Add models to the loader."""
        self.models.extend(models)
        self.chained = chained

    def get_device(self, model):
        """Get the device of a specific model."""
        if len(next(model.parameters(), [])) == 0:
            return torch.device("cpu")  # Default to CPU if no parameters
        return next(model.parameters()).device

    def get_dtype(self, model):
        """Get the dtype of a specific model."""
        return next(model.parameters(), torch.tensor(0.0)).dtype

    def set_eval_mode(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()

    def apply_ipex_optimization(self):
        """Apply IPEX optimization to all models."""
        for i, model in enumerate(self.models):
            self.models[i] = ipex.optimize(model)

    def warm_up(self, iterations=5, batch_size=None, channel=None, width=None, height=None):
        """Warm up the models by running a few dummy inferences."""

        # Check if batch_size, channel, width, and height are set properly
        if None in [batch_size, channel, width, height]:
            raise ValueError("Batch size, channel, width, and height must be provided.")

        for model in self.models:
            device = self.get_device(model)
            dtype = self.get_dtype(model)
            dummy_input = torch.randn(
                batch_size, channel, width, height, device=device, dtype=dtype
            )
            for _ in range(iterations):
                _ = model(dummy_input)

    def forward(self, x):
        """Run inference through the models."""
        return self._chained_forward(x) if self.chained else self._parallel_forward(x)

    def _chained_forward(self, x):
        """Run inference in a chained manner."""
        for model in self.models:
            x = model(x)
        return x

    def _parallel_forward(self, x):
        """Run inference in parallel or sequentially."""
        # TODO: Current implementation is not ideal, after some discussions
        # Streams API for XPU, and multiprocessing for CPU and
        # support for mixed models needs to be revisited
        if self.use_multithreading:
            # Use multithreading but due to GIL issue this is just concurrency
            # Read more here:
            # https://towardsdatascience.com/multithreading-multiprocessing-python-180d0975ab29
            # Still useful for I/O bound workload

            # Multi threading somehow needs to re-initialize `torch.no_grad()`
            def infer(model, input_tensor):
                with torch.no_grad():
                    return model(input_tensor)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(infer, model, x) for model in self.models]
                # Wait for all futures to complete
                concurrent.futures.wait(futures)
                # Retrieve the results in the order they were submitted
                outputs = [future.result() for future in futures]
        else:
            # Run sequentially
            outputs = [model(x) for model in self.models]
        return outputs
