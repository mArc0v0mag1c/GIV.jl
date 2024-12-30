function create_dualcoef(coef::Vector{V}) where {V}
    Ncoef = length(coef)
    DT = Dual{:delta,V,Ncoef}
    dualcoef = Vector{DT}(undef, Ncoef)
    for i in 1:Ncoef
        partialvalues = zeros(V, Ncoef)
        partialvalues[i] = one(V)
        # convert partialvalues to NTuple
        partialvalues = Tuple(partialvalues)
        partials = Partials{Ncoef,V}(partialvalues)
        dualcoef[i] = DT(coef[i], partials)
    end
    return dualcoef
end

"""
    Use to block any inference related with estimates without proper standard errors.
"""
function create_nandualcoef(::Dual{T,V,N}, coef) where {T,V,N}
    DT = Dual{T,V,N}
    partials = Partials{N,V}(Tuple(zeros(V, N) * NaN))
    return [DT(x, partials) for x in coef]
end

StatsAPI.stderror(givmodel::GIVModel) = sqrt.(diag(givmodel.vcov))
StatsAPI.stderror(ζ::Dual{:delta,V,N}, givmodel::GIVModel) where {V,N} = stderror(ζ, givmodel.vcov)
function StatsAPI.stderror(ζ::Dual{:delta,V,N}, vcov) where {V,N}
    partialsvec = [ζ.partials.values...]
    return sqrt(dot(partialsvec, vcov * partialsvec))
end
StatsAPI.stderror(ζ::AbstractArray{DT}, vcov) where {DT<:Dual{:delta,T,N}} where {T,N} =
    stderror.(ζ, Ref(vcov))

point_estimate(ζ::Dual{:delta,V,N}) where {V,N} = ζ.value
point_estimate(ζ::AbstractArray{DT}) where {DT<:Dual{:delta,T,N}} where {T,N} = point_estimate.(ζ)