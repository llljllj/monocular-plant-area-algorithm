import sys
import types
import importlib.machinery

# 导入sam3的初始代码
def _ensure_triton_stub():
    """为不支持triton的平台提供最小stub"""
    try:
        import triton  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    
    # 创建triton模块stub
    triton_stub = types.ModuleType("triton")
    triton_spec = importlib.machinery.ModuleSpec("triton", None)
    triton_stub.__spec__ = triton_spec
    triton_stub.__file__ = None
    triton_stub.__path__ = []

    def _jit(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    def _cdiv(a, b):
        return (a + b - 1) // b

    triton_stub.jit = _jit
    triton_stub.cdiv = _cdiv

    tl_stub = types.ModuleType("triton.language")
    tl_stub.__spec__ = importlib.machinery.ModuleSpec("triton.language", None)
    tl_stub.constexpr = object()
    tl_stub.dtype = object()
    triton_stub.language = tl_stub
    
    # 添加 triton.compiler
    tc_stub = types.ModuleType("triton.compiler")
    tc_stub.__spec__ = importlib.machinery.ModuleSpec("triton.compiler", None)
    tc_stub.compiler = types.ModuleType("triton.compiler.compiler") # Hack for nested compiler
    triton_stub.compiler = tc_stub
    
    # 添加 triton.backends 及其子模块
    tb_stub = types.ModuleType("triton.backends")
    tb_stub.__spec__ = importlib.machinery.ModuleSpec("triton.backends", None)
    triton_stub.backends = tb_stub
    
    tbc_stub = types.ModuleType("triton.backends.compiler")
    tbc_stub.__spec__ = importlib.machinery.ModuleSpec("triton.backends.compiler", None)
    tb_stub.compiler = tbc_stub
    
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = tl_stub
    sys.modules["triton.compiler"] = tc_stub
    sys.modules["triton.backends"] = tb_stub
    sys.modules["triton.backends.compiler"] = tbc_stub
    print("[INFO] Triton不可用，使用stub替代")
