using Core: MethodInstance, CodeInstance, OpaqueClosure
const CC = Core.Compiler
using CodeInfoTools

## code instance cache

struct CodeCache
    dict::IdDict{MethodInstance,Vector{CodeInstance}}

    CodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
end

Base.empty!(cc::CodeCache) = empty!(cc.dict)

function CC.setindex!(cache::CodeCache, ci::CodeInstance, mi::MethodInstance)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end


## world view of the cache

using Core.Compiler: WorldView

function CC.haskey(wvc::WorldView{CodeCache}, mi::MethodInstance)
    CC.get(wvc, mi, nothing) !== nothing
end

function CC.get(wvc::WorldView{CodeCache}, mi::MethodInstance, default)
    # check the cache
    for ci in get!(wvc.cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            # TODO: if (code && (code == jl_nothing || jl_ir_flag_inferred((jl_array_t*)code)))
            src = if ci.inferred isa Vector{UInt8}
                ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                       mi.def, C_NULL, ci.inferred)
            else
                ci.inferred
            end
            return ci
        end
    end

    return default
end

function CC.getindex(wvc::WorldView{CodeCache}, mi::MethodInstance)
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{CodeCache}, ci::CodeInstance, mi::MethodInstance)
    src = if ci.inferred isa Vector{UInt8}
        ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                mi.def, C_NULL, ci.inferred)
    else
        ci.inferred
    end
    CC.setindex!(wvc.cache, ci, mi)
end


## custom interpreter

struct MLIRInterpreter <: CC.AbstractInterpreter
    world::UInt

    code_cache::CodeCache
    inf_cache::Vector{CC.InferenceResult}

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function MLIRInterpreter(world::UInt;
                           code_cache::CodeCache,
                           inf_params::CC.InferenceParams,
                           opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    # inf_params.ipo_constant_propagation = false
    inf_params = CC.InferenceParams(ipo_constant_propagation=false)

    inf_cache = Vector{CC.InferenceResult}()

    return MLIRInterpreter(world,
                             code_cache, inf_cache,
                             inf_params, opt_params)
end
MLIRInterpreter() = MLIRInterpreter(
    Base.get_world_counter();
    code_cache=global_ci_cache,
    inf_params=CC.InferenceParams(),
    opt_params=CC.OptimizationParams()
)

CC.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
CC.get_world_counter(interp::MLIRInterpreter) = interp.world
CC.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
CC.code_cache(interp::MLIRInterpreter) = WorldView(interp.code_cache, interp.world)

# No need to do any locking since we're not putting our results into the runtime cache
CC.lock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing
CC.unlock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing

function CC.add_remark!(interp::MLIRInterpreter, sv::CC.InferenceState, msg)
    @debug "Inference remark during compilation of MethodInstance of $(sv.linfo): $msg"
end

CC.may_optimize(interp::MLIRInterpreter) = true
CC.may_compress(interp::MLIRInterpreter) = true
CC.may_discard_trees(interp::MLIRInterpreter) = true
CC.verbose_stmt_info(interp::MLIRInterpreter) = false


## utils

# create a MethodError from a function type
# TODO: fix upstream
function unsafe_function_from_type(ft::Type)
    if isdefined(ft, :instance)
        ft.instance
    else
        # HACK: dealing with a closure or something... let's do somthing really invalid,
        #       which works because MethodError doesn't actually use the function
        Ref{ft}()[]
    end
end
function MethodError(ft::Type{<:Function}, tt::Type, world::Integer=typemax(UInt))
    Base.MethodError(unsafe_function_from_type(ft), tt, world)
end
MethodError(ft, tt, world=typemax(UInt)) = Base.MethodError(ft, tt, world)

const global_ci_cache = CodeCache()

mlir_bool_conversion(x::Bool) = x
function mlir_bool_conversion(x::T) where T
    error("Cannot convert type $T")
end
mlir_bool_conversion(x::Int) = x != 0

import Core.Compiler: retrieve_code_info, maybe_validate_code, InferenceState, InferenceResult
# Replace usage sites of `retrieve_code_info`, OptimizationState is one such, but in all interesting use-cases
# it is derived from an InferenceState. There is a third one in `typeinf_ext` in case the module forbids inference.
function InferenceState(result::InferenceResult, cached::Symbol, interp::MLIRInterpreter)
    src = retrieve_code_info(result.linfo, interp.world)
    src === nothing && return nothing
    maybe_validate_code(result.linfo, src, "lowered")
    src = transform(interp, result.linfo, src)
    maybe_validate_code(result.linfo, src, "transformed")

    return InferenceState(result, src, cached, interp)
end

function static_eval(mod, name)
    if Base.isbindingresolved(mod, name) && Base.isdefined(mod, name)
        return Some(getfield(mod, name))
    else
        return nothing
    end
end
static_eval(gr::GlobalRef) = static_eval(gr.mod, gr.name)

struct DestinationOffsets
    indices::Vector{Int}
    DestinationOffsets() = new([])
end
function Base.insert!(d::DestinationOffsets, insertion::Int)
    candidateindex = d[insertion]+1
    if (length(d.indices) == 0)
        push!(d.indices, insertion)
    elseif candidateindex == length(d.indices)+1
        push!(d.indices, insertion)
    elseif (candidateindex == 1) || (d.indices[candidateindex-1] != insertion)
        insert!(d.indices, candidateindex, insertion)
    end
    return d
end
Base.getindex(d::DestinationOffsets, i::Int) = searchsortedlast(d.indices, i, lt= <=)

function insert_bool_conversions_pass(mi, src)
    offsets = DestinationOffsets()

    b = CodeInfoTools.Builder(src)
    for (v, st) in b
        if st isa Core.GotoIfNot
            arg = st.cond isa Core.SSAValue ? var(st.cond.id + offsets[st.cond.id]) : st.cond
            # arg = st.cond isa Core.SlotNumber ? st.cond : var(st.cond.id + offsets[st.cond.id])
            b[v] = Statement(Expr(:call, GlobalRef(Jojo, :mlir_bool_conversion), arg))
            # b[v] = Statement(Expr(:call, GlobalRef(Main, :mlir_bool_conversion)))
            push!(b, Core.GotoIfNot(v, st.dest))
            insert!(offsets, v.id)
        elseif st isa Core.GotoNode
            b[v] = st
        end
    end

    # fix destinations and conditions
    for i in 1:length(b.to)
        st = b.to[i].node
        if st isa Core.GotoNode
            b.to[i] = Core.GotoNode(st.label + offsets[st.label])
        elseif st isa Core.GotoIfNot
            b.to[i] = Statement(Core.GotoIfNot(st.cond, st.dest + offsets[st.dest]))
        end
    end
    finish(b)
end

function transform(interp, mi, src)
    src = insert_bool_conversions_pass(mi, src)
    return src
end
