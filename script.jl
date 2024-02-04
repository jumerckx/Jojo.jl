using TestEnv
TestEnv.activate()

using MLIR
includet("utils.jl")
using Jojo
import Jojo: MemRef, @mlirfunction
using Jojo.Types
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith, index, linalg, transform, builtin

using MLIR: IR, API, Dialects
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir`` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.get_result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=IR.Attribute(1, IR.IndexType())) |> IR.get_result
    new_index = arith.subi(i, oneoff) |> IR.get_result
    T(Dialects.memref.load(A, [new_index], result=IR.MLIRType(T)) |> IR.get_result)
end

square(a) = a*a
f(a, b) = (a>b) ? a+b : square(a)
h(a, i) = a[i]

# Base.code_ircode(f, (i64, i64), interp=Jojo.MLIRInterpreter())
module_f = Jojo.code_mlir(f, Tuple{i64, i64}, do_simplify=true)
addr_f = jit(lowerModuleToLLVM(module_f); opt=3)("_mlir_ciface_f")
@ccall $addr_f(3::Int, 2::Int)::Int


# Base.code_ircode(h, Tuple{memref{i64, 1}, Types.index}, interp=Jojo.MLIRInterpreter())
module_h = Jojo.code_mlir(h, Tuple{memref{i64, 1}, Types.index})
addr_h = jit(lowerModuleToLLVM(module_h); opt=3)("_mlir_ciface_h")
a = [42, 43, 44]
@ccall $addr_h(MemRef(a)::Ref{MemRef}, 1::Int)::Int