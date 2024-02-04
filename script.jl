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

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, result=IR.MLIRType(Bool), predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir`` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.get_result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=Attribute(1, IR.IndexType())) |> IR.get_result
    new_index = arith.subi(i, oneoff) |> IR.get_result
    T(Dialects.memref.load(A, [new_index]) |> IR.get_result)
end
@noinline Jojo.mlir_bool_conversion(a::i1)::Bool = Jojo.new_intrinsic()

square(a) = a*a
f(a, b) = (a>b) ? a+b : square(a)
g(a::AbstractVector) = a[2]
h(a, i) = a[i]

Base.code_ircode(f, (i64, i64), Jojo.MLIRInterpreter())
@time m = Jojo.code_mlir(f, Tuple{i64, i64}, do_simplify=true)
lowerModuleToLLVM(m)

Jojo.code_mlir(Tuple{Complex{i64}, Complex{i64}}) do a, b
    a + b
end |> lowerModuleToLLVM

