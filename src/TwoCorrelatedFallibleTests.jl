

struct CorrConstrainedCustom{p, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::Vector{T}
  S::MVector{p,T}
  C::MVector{p,T}
  constraint::MVector{p,T}
end
CorrConstrainedCustom{T}(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) = CorrConstrainedCustom{length(x), T}(logit.(x), x)


struct CorrErrors{p,T} <: parameters{T,1}
  x::Vector{T}
  π::ProbabilityVector{p,T}
  S::ProbabilityVector{2,T}
  C::ProbabilityVector{2,T}
  covsc::CorrConstrainedCustom{2,T}
end
@generated Base.size(A::CorrErrors{p,T} where {T<:Real}) where {p} = (6+p,)
@generated Base.length(A::CorrErrors{p,T} where {T<:Real}) where {p} = 6+p


struct CorrErrorsData{p} <: Data
  n::SVector{p,SVector{8,Int}}
  πBeta::SVector{p,Distributions.Beta{Float64}}
  S1Beta::Distributions.Beta{Float64}
  C1Beta::Distributions.Beta{Float64}
  S2Beta::Distributions.Beta{Float64}
  C2Beta::Distributions.Beta{Float64}
end



function CorrErrorsData(n::Vararg{Vector{Int},p}; απ::Vector = ones(p), βπ::Vector = ones(p), αS1::Real = 1.0, βS1::Real = 1.0, αC1::Real = 1.0, βC1::Real = 1.0, αS2::Real = 1.0, βS2::Real = 1.0, αC2::Real = 1.0, βC2::Real = 1.0) where {p}
  CorrErrorsData{p}(SVector{p}([SVector{8}(n_i) for n_i ∈ n]), SVector{p}([Beta(απ[i], βπ[i]) for i ∈ 1:p]), Beta(αS1, βS1), Beta(αC1, βC1), Beta(αS2, βS2), Beta(αC2, βC2))
end

function common_p(Θ::CorrErrors{p,Float64}) where {p}
  [[Θ.π[i]*(prod(Θ.S.x)+Θ.covsc[1]) + (1-Θ.π[i])*((1-Θ.C[1])*(1-Θ.C[2])+Θ.covsc[2]), Θ.π[i]*(Θ.S[1]*(1-Θ.S[2])-Θ.covsc[1]) + (1-Θ.π[i])*((1-Θ.C[1])*Θ.C[2]-Θ.covsc[2]), Θ.π[i]*(Θ.S[2]*(1-Θ.S[1])-Θ.covsc[1]) + (1-Θ.π[i])*(Θ.C[1]*(1-Θ.C[2])-Θ.covsc[2]), Θ.π[i]*((1-Θ.S[1])*(1-Θ.S[2])+Θ.covsc[1]) + (1-Θ.π[i])*(Θ.C[1]*Θ.C[2]+Θ.covsc[2])] for i ∈ 1:p]
end
function p_i(Θ::CorrErrors{p,Float64}, i::Int) where {p}
  [[Θ.π[j]*Θ.S[i] + (1-Θ.π[j])*(1-Θ.C[i]), Θ.π[j]*(1-Θ.S[i]) + (1-Θ.π[j])*Θ.C[i]] for j ∈ 1:p]
end
function gen_data(Θ::CorrErrors{p,Float64}, n_common::Int, n_1_only::Int, n_2_only::Int) where {p}
  double_test = common_p(Θ)
  p_1_only = p_i(Θ, 1)
  p_2_only = p_i(Θ, 2)
  [vcat(rand(Multinomial(n_common, double_test[i])), rand(Multinomial(n_1_only, p_1_only[i])), rand(Multinomial(n_2_only, p_2_only[i]))) for i ∈ 1:p]
end



function update_constraint!(x::CorrConstrainedCustom)
  x.constraint[1] = minimum(x.S) - prod(x.S)
  x.constraint[2] = minimum(x.C) - prod(x.C)
end
function update!(x::CorrConstrainedCustom)
  update_constraint!(x)
  x.x .= logistic.(x.Θ) .* x.constraint
end
function log_jacobian!(x::CorrConstrainedCustom)
  sum(log.(x.x) .+ log.(1 .- x.x ./ x.constraint))
end
type_length{p,T}(::Type{CorrConstrainedCustom{p,T}}) = p
Base.getindex(x::CorrConstrainedCustom, i::Int) = x.x[i]
function Base.setindex!(x::CorrConstrainedCustom, v::Real, i::Int)
  update_constraint!(x)
  x.x[i] = v
  x.Θ[i] = logit(v / x.constraint[i])
end
function construct{p, T}(::Type{CorrConstrainedCustom{p,T}}, Θv::Vector{T}, i::Int, S::ProbabilityVector{2,T}, C::ProbabilityVector{2,T})
  v = view(Θv, i + (1:p))
  constraint = [minimum(S.x) - prod(S.x), minimum(C.x) - prod(C.x)]
  CorrConstrainedCustom{p, T}(v, logistic.(v) .* constraint, S.x, C.x, constraint)
end
function construct{p, T}(::Type{CorrConstrainedCustom{p,T}}, Θv::Vector{T}, i::Int, vals::Vector{T}, S::ProbabilityVector{2,T}, C::ProbabilityVector{2,T})
  constraint = [minimum(S.x) - prod(S.x), minimum(C.x) - prod(C.x)]
  pv = CorrConstrainedCustom{p, T}(view(Θv, i + (1:p)), vals, S.x, C.x, constraint)
  pv.Θ .= logit.(vals ./ constraint)
  pv
end





function construct{T <: CorrErrors}(::Type{T})
  if isa(T, UnionAll)
    T2 = T{Float64}
  else
    T2 = T
  end
  field_count = length(T2.types)
  indices = cumsum([type_length(T2.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T2.parameters[end], indices[end])

  π = construct(T2.types[2], Θ, indices[1])
  S = construct(T2.types[3], Θ, indices[2])
  C = construct(T2.types[4], Θ, indices[3])
  covsc = construct(T2.types[5], Θ, indices[4], S, C)

  T2(Θ, π, S, C, covsc)
end
function construct{T <: CorrErrors}(::Type{T}, Θ::Vector)

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])

  π = construct(T.types[2], Θ, indices[1])
  S = construct(T.types[3], Θ, indices[2])
  C = construct(T.types[4], Θ, indices[3])
  covsc = construct(T.types[5], Θ, indices[4], S, C)

  T(Θ, π, S, C, covsc)
end
function construct{T <: CorrErrors}(::Type{T}, A::Vararg)

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T.parameters[end], indices[end])

  π = construct(T.types[2], Θ, indices[1], A[1])
  S = construct(T.types[3], Θ, indices[2], A[2])
  C = construct(T.types[4], Θ, indices[3], A[3])
  covsc = construct(T.types[5], Θ, indices[4], A[4], S, C)

  T(Θ, π, S, C, covsc)
end





function log_priors(Θ::CorrErrors{p,T}, data::Data) where {p,T}
  out = logpdf(data.S1Beta, Θ.S[1])
  out += logpdf(data.C1Beta, Θ.C[1])
  out += logpdf(data.S2Beta, Θ.S[2])
  out += logpdf(data.C2Beta, Θ.C[2])
  for i ∈ 1:p
    out += logpdf(data.πBeta[i], Θ.π[i])
  end
  out# - sum(log, Θ.covsc.constraint)
end

function log_likelihood(Θ::CorrErrors, n::SVector{8,Int}, π::Real)
  out = n[1] * log( π * (Θ.S[1]*Θ.S[2]+Θ.covsc.x[1]) + (1-π) * ((1-Θ.C[1])*(1-Θ.C[2])+Θ.covsc.x[2]))
  out += n[2] * log( π * (Θ.S[1]*(1-Θ.S[2])-Θ.covsc.x[1]) + (1-π) * ((1-Θ.C[1])*Θ.C[2]-Θ.covsc.x[2]))
  out += n[3] * log( π * ((1-Θ.S[1])*Θ.S[2]-Θ.covsc.x[1]) + (1-π) * (Θ.C[1]*(1-Θ.C[2])-Θ.covsc.x[2]))
  out += n[4] * log( π * ((1-Θ.S[1])*(1-Θ.S[2])+Θ.covsc.x[1]) + (1-π) * (Θ.C[1]*Θ.C[2]+Θ.covsc.x[2]))

  out += n[5] * log( π*Θ.S[1] + (1-π)*(1-Θ.C[1]) )
  out += n[6] * log( π*(1-Θ.S[1]) + (1-π)*Θ.C[1] )

  out += n[7] * log( π*Θ.S[2] + (1-π)*(1-Θ.C[2]) )
  out + n[8] * log( π*(1-Θ.S[2]) + (1-π)*Θ.C[2] )
end

function log_density(Θ::CorrErrors{p,T}, data::CorrErrorsData) where {p,T}
   out = log_priors(Θ, data)
   for i ∈ 1:p
     out += log_likelihood(Θ, data.n[i], Θ.π[i])
   end
   out
end


function negative_log_density{T, P <: CorrErrors}(Θ::Vector{T}, ::Type{P}, data::Data)
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - log_density(param, data)
end
function negative_log_density!(Θ::CorrErrors, data::Data)
#  update!(Θ)
  nld = -log_jacobian!(Θ)
  nld - log_density(Θ, data)
end
