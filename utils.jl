using InteractiveUtils, CassetteOverlay
using MLIR: IR, API

macro code_ircode(ex0...)
    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(@__MODULE__, :(Base.code_ircode), ex0)
    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end

function registerAllDialects!()
    ctx = IR.context()
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    # handle = API.mlirGetDialectHandle__jlir__()
    # API.mlirDialectHandleInsertDialect(handle, registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

function mlir_opt(mod::IR.MModule, pipeline::String)
    pm = IR.PassManager()
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), pipeline)
    status = API.mlirPassManagerRunOnOp(pm, IR.get_operation(mod).operation)
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end

function lowerModuleToLLVM(mod::IR.MModule)
    pm = IR.PassManager()

    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertAffineToStandard())
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "normalize-memrefs")
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "affine-expand-index-ops")
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertVectorToLLVM())
    IR.add_owned_pass!(pm, API.mlirCreateConversionSCFToControlFlow())
    IR.add_owned_pass!(pm, API.mlirCreateConversionMemRefToLLVMConversionPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertFuncToLLVM())
    IR.add_owned_pass!(pm, API.mlirCreateConversionArithToLLVMConversionPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionReconcileUnrealizedCasts())
    status = API.mlirPassManagerRun(pm, mod)

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end

function lowerModuleToNVVM(mod::IR.MModule)
    pm = IR.PassManager()

    pm_func = API.mlirPassManagerGetNestedUnder(pm, "func.func")

    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionConvertNVGPUToNVVMPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuKernelOutlining())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertFuncToLLVMPass())

    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionArithToLLVMConversionPass())

    pm_gpu = API.mlirPassManagerGetNestedUnder(pm, "gpu.module")

    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateTransformsStripDebugInfo())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionArithToLLVMConversionPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertGpuOpsToNVVMOps())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertNVVMToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionReconcileUnrealizedCasts())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuNVVMAttachTarget())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionGpuToLLVMConversionPass())
    
    
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuModuleToBinaryPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertFuncToLLVMPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionReconcileUnrealizedCasts())

    status = API.mlirPassManagerRunOnOp(pm, IR.get_operation(mod))

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end
  
function jit(mod::IR.MModule; opt=0)
    paths = Base.unsafe_convert.(Ref(API.MlirStringRef), [MLIR.API.mlir_c_runner_utils, MLIR.API.mlir_runner_utils])
    jit = API.mlirExecutionEngineCreate(
        mod,
        opt,
        length(paths), # numPaths
        paths, # libPaths
        true # enableObjectDump
    )
    function lookup(name)
        addr = API.mlirExecutionEngineLookup(jit, name)
        (addr == C_NULL) && error("Lookup failed.")
        return addr
    end
    return lookup
end

cb = IR.Block() # Fixed block for demonstration purposes.
IR.lose_ownership!(cb)
currentblock() = cb

@MethodTable MLIRCompilation
@overlay MLIRCompilation function IR.create_operation(
        name, loc;
        results=nothing,
        operands=nothing,
        owned_regions=nothing,
        successors=nothing,
        attributes=nothing,
        result_inference=isnothing(results))
    @info "Overlayed!!!"
    op = @nonoverlay IR.create_operation(
        name, loc;
        results,
        operands,
        owned_regions,
        successors,
        attributes,
        result_inference)
    push!(currentblock(), op)
    return op
end