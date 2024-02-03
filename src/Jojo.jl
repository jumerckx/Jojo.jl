module Jojo

export code_mlir

using MLIR.IR
using MLIR: API
using MLIR.Dialects: arith, func, cf, memref, index, builtin, llvm
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

const JojoScalarType = Union{Bool, Int64, UInt64, Int32, UInt32, Float16, Float32, Float64, UInt64}
const JojoType = Union{JojoScalarType, Array{JojoScalarType}}

include("abstract.jl")
include("memref.jl")
include("pass.jl")
include("overlay.jl")
include("codegencontext.jl")
include("ValueTypes.jl")

struct InstructionContext{I}
    args::Vector
    result_type::Type
    loc::Location
end

function cmpi_pred(predicate)
    function(lhs, rhs; loc=Location())
        arith.cmpi(lhs, rhs; predicate, location=loc)
    end
end

function single_op_wrapper(fop)
    (cg::CodegenContext, ic::InstructionContext)->IR.get_result(push!(currentblock(cg), fop(indextoi64.(Ref(cg), get_value.(Ref(cg), ic.args))...)))
end

indextoi64(cg::CodegenContext, x; loc=IR.Location()) = x
function indextoi64(cg::CodegenContext, x::Value; loc=IR.Location())
    mlirtype = IR.get_type(x)
    if API.mlirTypeIsAIndex(mlirtype)
        return push!(currentblock(cg), arith.index_cast(
            x;
            out=MLIRType(Int), location=loc)
            ) |> IR.get_result
    else
        return x
    end
end
function i64toindex(cg, x::Value; loc=IR.Location())
    mlirtype = IR.get_type(x)
    if API.mlirTypeIsAInteger(mlirtype)
        return push!(currentblock(cg), arith.index_cast(
            x;
            out=IR.IndexType(), location=loc
        )) |> IR.get_result
    else
        return x
    end
end

# all these emit functions are remnants of a previous version and might be not useful anymore:
function emit(cg::CodegenContext, ic::InstructionContext{Base.getfield})
    object = get_value(cg, first(ic.args))
    field = ic.args[2]
    if field isa QuoteNode; field=field.value; end
    return cg, getfield(object, field)
end
function emit(cg::CodegenContext, ic::InstructionContext{Core.tuple})
    inputs = get_value.(Ref(cg), ic.args)
    outputs = IR.get_type.(inputs)
    
    op = push!(currentblock(cg), builtin.unrealized_conversion_cast(
        inputs;
        outputs,
        location=ic.loc
    ))
    return cg, Tuple(IR.get_result.(Ref(op), 1:fieldcount(ic.result_type)))
end
function emit(cg::CodegenContext, ic::InstructionContext{Core.ifelse})
    T = get_type(cg, ic.args[2])
    @assert T == get_type(cg, ic.args[3]) "Branches in Core.ifelse should have the same type."
    condition, true_value, false_value = get_value.(Ref(cg), ic.args)
    return cg, IR.get_result(push!(currentblock(cg), arith.select(condition, true_value, false_value; result=IR.get_type(true_value), location=ic.loc)))
end
function emit(cg::CodegenContext, ic::InstructionContext{Base.throw_boundserror})
    @warn "Ignoring potential boundserror while generating MLIR."
    return cg, nothing
end

"Generates a block argument for each phi node present in the block."
function prepare_block(ir, bb)
    b = Block()

    for sidx in bb.stmts
        stmt = ir.stmts[sidx]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        type = stmt[:type]
        IR.push_argument!(b, MLIRType(type), Location())
    end

    return b
end

"Values to populate the Phi Node when jumping from `from` to `to`."
function collect_value_arguments(ir, from, to)
    to = ir.cfg.blocks[to]
    values = []
    for s in to.stmts
        stmt = ir.stmts[s]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        edge = findfirst(==(from), inst.edges)
        if isnothing(edge) # use dummy scalar val instead
            val = zero(stmt[:type])
            push!(values, val)
        else
            push!(values, inst.values[edge])
        end
    end
    values
end

unpack(T) = unpack(IR.MLIRValueTrait(T), T)
unpack(::IR.Convertible, T) = (T, )
function unpack(::IR.NonConvertible, T)
    @assert isbitstype(T) "Cannot unpack type $T that is not `isbitstype`"
    fc = fieldcount(T)
    if (fc == 0)
        if (sizeof(T) == 0)
            return []
        else
            error("Unable to unpack NonConvertible type $T any further")
        end
    end
    unpacked = []
    for i in 1:fc
        ft = fieldtype(T, i)
        append!(unpacked, unpack(ft))
    end
    return unpacked
end

"""
    code_mlir(f, types::Type{Tuple}) -> IR.Operation

Returns a `func.func` operation corresponding to the ircode of the provided method.
This only supports a few Julia Core primitives and scalar types of type $JojoType.

!!! note
    The Julia SSAIR to MLIR conversion implemented is very primitive and only supports a
    handful of primitives. A better to perform this conversion would to create a dialect
    representing Julia IR and progressively lower it to base MLIR dialects.
"""
function code_mlir(f, types; do_simplify=true, emit_region=false, ignore_returns=emit_region)
    ctx = context()
    ir, ret = Core.Compiler.code_ircode(f, types) |> only
    @assert first(ir.argtypes) isa Core.Const

    values = Vector(undef, length(ir.stmts))
    args = Vector(undef, length(types.parameters))
    for dialect in ("func", "cf")
        IR.get_or_load_dialect!(dialect)
    end

    blocks = [
        prepare_block(ir, bb)
        for bb in ir.cfg.blocks
    ]

    CodegenContext(;
        regions=[Region()],
        loop_thunks=[],
        blocks,
        entryblock=blocks[begin],
        currentblockindex=1,
        ir,
        ret,
        values,
        args
    ) do cg
        for (i, argtype) in enumerate(types.parameters)
            args = []
            for t in unpack(argtype)
                arg = IR.push_argument!(cg.entryblock, MLIRType(t), Location())
                push!(args, t(arg))
            end
            # TODO: what to do with padding?
            cg.args[i] = reinterpret(argtype, Tuple(args))
        end

        for (block_id, bb) in enumerate(cg.ir.cfg.blocks)
            cg.currentblockindex = block_id
            @info "number of regions: $(length(cg.regions))"
            @show currentblock(cg)
            push!(currentregion(cg), currentblock(cg))
            n_phi_nodes = 0

            for sidx in bb.stmts
                stmt = cg.ir.stmts[sidx]
                inst = stmt[:inst]
                @info "Working on: $(inst)"
                if inst == nothing
                    inst = Core.GotoNode(block_id+1)
                    line = Core.LineInfoNode(Jojo, :code_mlir, Symbol(@__FILE__), Int32(@__LINE__), Int32(@__LINE__))
                else
                    line = cg.ir.linetable[stmt[:line]]
                end

                if Meta.isexpr(inst, :call)
                    val_type = stmt[:type]
                    called_func, args... = inst.args

                    if called_func isa GlobalRef # TODO: should probably use something else here
                        called_func = getproperty(called_func.mod, called_func.name)
                    end
                    args = map(args) do arg
                        if arg isa GlobalRef
                            arg = getproperty(arg.mod, arg.name)
                        elseif arg isa QuoteNode
                            arg = arg.value
                        end
                        return arg
                    end

                    getintrinsic(gr::GlobalRef) = Core.Compiler.abstract_eval_globalref(gr)
                    getintrinsic(inst::Expr) = getintrinsic(first(inst.args))
                    getintrinsic(mod::Module, name::Symbol) = getintrinsic(GlobalRef(mod, name))

                    loc = Location(string(line.file), line.line, 0)
                    ic = InstructionContext{called_func}(args, val_type, loc)
                    # return cg, ic
                    @show typeof(ic)
                    cg, res = emit(cg, ic)

                    values[sidx] = res
                elseif Meta.isexpr(inst, :invoke)
                    val_type = stmt[:type]
                    _, called_func, args... = inst.args
                    if called_func isa Core.SSAValue
                        called_func = get_value(cg, called_func)
                    end

                    if called_func isa GlobalRef # TODO: should probably use something else here
                        called_func = getproperty(called_func.mod, called_func.name)
                    end
                    args = map(args) do arg
                        if arg isa GlobalRef
                            arg = getproperty(arg.mod, arg.name)
                        elseif arg isa QuoteNode
                            arg = arg.value
                        end
                        return arg
                    end
                    loc = Location(string(line.file), line.line, 0)
                    if val_type == Core.Const(nothing)
                        val_type = Nothing
                    end
                    ic = InstructionContext{called_func}(args, val_type, loc)

                    argvalues = get_value.(Ref(cg), ic.args)
                    @show called_func, argvalues
                    
                    out = mlircompilationpass() do
                        called_func(argvalues...)
                    end

                    values[sidx] = out


                elseif inst isa PhiNode
                    values[sidx] = IR.get_argument(currentblock(cg), n_phi_nodes += 1)
                elseif inst isa PiNode
                    values[sidx] = get_value(values, inst.val)
                elseif inst isa GotoNode
                    args = Value[get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, inst.label))...]
                    dest = cg.blocks[inst.label]
                    loc = Location(string(line.file), line.line, 0)
                    push!(currentblock(cg), cf.br(args; dest, location=loc))
                elseif inst isa GotoIfNot
                    false_args = Value[get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, inst.dest))...]
                    cond = get_value(cg, inst.cond)
                    @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
                    trueDest = setdiff(bb.succs, inst.dest) |> only
                    true_args = Value[get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, trueDest))...]
                    trueDest = cg.blocks[trueDest]
                    falseDest = cg.blocks[inst.dest]

                    location = Location(string(line.file), line.line, 0)
                    # @show cond
                    # if inst.cond.id == 54; return 1; end
                    cond_br = cf.cond_br(cond, true_args, false_args; trueDest, falseDest, location)
                    push!(currentblock(cg), cond_br)
                elseif inst isa ReturnNode
                    ignore_returns && continue
                    line = cg.ir.linetable[stmt[:line]]
                    loc = Location(string(line.file), line.line, 0)
                    if isdefined(inst, :val)
                        if (inst.val isa GlobalRef)  && (getproperty(inst.val.mod, inst.val.name) == nothing)
                            returnvalue = []
                        else
                            v = get_value(cg, inst.val)
                            returnvalue = reinterpret(Tuple{unpack(typeof(v))...}, v)
                        end
                    else
                        returnvalue = [IR.get_result(push!(currentblock(cg), llvm.mlir_undef(; res=MLIRType(cg.ret), location=loc)))]
                    end
                    push!(currentblock(cg), func.return_(returnvalue; location=loc))
                elseif Meta.isexpr(inst, :new)
                    args = get_value.(Ref(cg), inst.args[2:end])
                    values[sidx] = reinterpret(inst.args[1], Tuple(args))
                elseif Meta.isexpr(inst, :code_coverage_effect)
                    # Skip
                elseif Meta.isexpr(inst, :boundscheck)
                    @warn "discarding boundscheck"
                    cg.values[sidx] = IR.get_result(push!(currentblock(cg), arith.constant(value=true)))
                elseif Meta.isexpr(inst, :GlobalRef)

                else
                    # @warn "unhandled ir $(inst)"
                    # return inst
                    @warn "unhandled ir $(inst) of type $(typeof(inst))"
                    if inst isa GlobalRef
                        inst = getproperty(inst.mod, inst.name)
                    end
                    cg.values[sidx] = inst
                end
            end
        end
        
        func_name = nameof(f)
        
        # add fallthrough to next block if necessary
        for (i, b) in enumerate(cg.blocks)
            if (i != length(cg.blocks) && IR.mlirIsNull(API.mlirBlockGetTerminator(b)))
                @warn "Block $i did not have a terminator, adding one."
                args = []
                dest = cg.blocks[i+1]
                loc = IR.Location()
                push!(b, cf.br(args; dest, location=loc))
            end
        end

        if emit_region
            println("emitting region")
            @show currentregion(cg)
            println(currentregion(cg).region.ptr)
            return currentregion(cg)
        else
            input_types = MLIRType[
                IR.get_type(IR.get_argument(cg.entryblock, i))
                for i in 1:IR.num_arguments(cg.entryblock)
            ]
            result_types = MLIRType.(unpack(ret))

            ftype = MLIRType(input_types => result_types)
            op = IR.create_operation(
                "func.func",
                Location();
                attributes = [
                    NamedAttribute("sym_name", IR.Attribute(string(func_name))),
                    NamedAttribute("function_type", IR.Attribute(ftype)),
                    NamedAttribute("llvm.emit_c_interface", IR.Attribute(API.mlirUnitAttrGet(IR.context())))
                ],
                owned_regions = Region[currentregion(cg)],
                result_inference=false,
            )
            IR.verifyall(op)    
            
            mod = IR.MModule(IR.Location())
            push!(IR.get_body(mod), op)
            if IR.verify(op) && do_simplify
                simplify(mod)
            end
            return mod
        end
    end
end

end # module Jojo
