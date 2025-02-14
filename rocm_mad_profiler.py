import torch
import os
 
class MADProfiler(object):
    def __init__(self, backend=None, nvtx_tracing=True, index=None):
       self.backend = backend
       self.nvtx_tracing = nvtx_tracing
 
       # Need for torch.multiprocessing.spawn type of launches
       self.index = index
 
    def __enter__(self):
        if self.backend == "rpd":
            from rpdTracerControl import rpdTracerControl
            from rpdTracerControl import isChildProcess
            filename = "trace.rpd"
            appendMode = False # append mode has to be false for tables to be created in RPD file
            if os.path.exists( filename ):
                appendMode = True
            if not isChildProcess():
                # filename needs to be set before forking child processess or using RPDT_FILENAME or loadTracer
                rpdTracerControl.setFilename(name = filename, append=appendMode)
            self.rpd_profile = rpdTracerControl()
            self.rpd_profile.start()
            if self.nvtx_tracing:
                self.nvtx_prof = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                self.nvtx_prof.__enter__()
        elif self.backend == "torch":
             def trace_handler(prof):
                 filename = "trace.json"
                 if "LOCAL_RANK" in os.environ:
                     local_rank = int(os.environ["LOCAL_RANK"])
                     filename = "trace_" + str(local_rank) + ".json"
                 elif self.index is not None:
                     filename = "trace_" + str(self.index) + ".json"
                 prof.export_chrome_trace(filename)
             self.torch_profile = torch.profiler.profile(
                     activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                     record_shapes=True,
                     on_trace_ready=trace_handler
                 )
             self.torch_profile.start()
        torch.cuda.synchronize()
 
    def __exit__(self, *args):
        torch.cuda.synchronize()
        if self.backend == "rpd":
            self.rpd_profile.stop()
            if self.nvtx_tracing:
                self.nvtx_prof.__exit__(None, None, None)
        elif self.backend == "torch":
            self.torch_profile.stop()
 
    def start(self):
        self.__enter__()
 
    def stop(self, *args):
        self.__exit__(args)
 
 
def mad_profile(func, backend=None, nvtx_tracing=False):
    def wrapper():
        with MADProfiler(backend, nvtx_tracing):
            func()
    return wrapper
