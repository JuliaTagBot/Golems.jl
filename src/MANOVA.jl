



struct TF_RE_MANOVA_Data_unbalanced <: ANOVA_Data_complete

end

struct TF_RE_MANOVA_Data_incomplete <: ANOVA_Data_incomplete

end

function TF_RE_MANOVA_Data(y::Array{Float64,2}, yb::Array{Int64,1}, yt::Array{Int64,1}, p = size(y,2); μ0 = zeros(p), Σμ = diagm(fill(100.0, p)), Γβ = eye(p), Γτ = eye(p), Γθ = eye(p), Γϵ = eye(p), νβ::Real = p+1.0, ντ::Real = p+1.0, νθ::Real = p+1.0, νϵ::Real = p+1.0)
  n = size(y,1)
  b = maximum(yb)
  t = maximum(yt)

  y .-= μ0'

#  Σy = zeros(Float64, b, t, p)
#  Σyyt = zeros(Float64, b, t, p, p)
  N = zeros(Int64, b, t)
  for i ∈ 1:n
    N[yb[i], yt[i]] += 1
#    Σy[yb[i], yt[i], :] .+= y[i,:]
#    Σyyt[yb[i], yt[i], :, :] .+= y[i,:] * y[i,:]'
  end
  if all(x -> x == N[1], N)
    return TF_RE_MANOVA_Data_balanced(y, yb, yt, n, N[1], b, t, p, Σμ, Γβ, Γτ, Γθ, Γϵ, νβ, ντ, νθ, νϵ)
  elseif any(x -> x == 0, N)
    return TF_RE_MANOVA_Data_incomplete(y, yb, yt, n, N, b, t, p, Σμ, Γβ, Γτ, Γθ, Γϵ, νβ, ντ, νθ, νϵ)
  else
    return TF_RE_MANOVA_Data_unbalanced(y, yb, yt, n, N, b, t, p, Σμ, Γβ, Γτ, Γθ, Γϵ, νβ, ντ, νθ, νϵ)
  end
end



function TF_RE_MANOVA_Data_unbalanced(y::Array{Float64,2}, yb::Array{Int64,1}, yt::Array{Int64,1}, n::Int, N::Array{Int,2}, b = maximum(yp), t = maximum(yo), p = size(y,2), Σμ = eye(p), Γβ = eye(p), Γτ = eye(p), Γθ = eye(p), Γϵ = eye(p), νβ::Real = p+1.0, ντ::Real = p+1.0, νθ::Real = p+1.0, νϵ::Real = p+1.0)
  bt = b*t
  y_bar = zeros(N)
  unique_n = union(N)
  for i ∈ 1:n
    y_bar[yb[i], yt[i],:] .+= y[i,:]
  end
  y_bar ./= N
  yδ = Array{Float64}(p)
  for i ∈ 1:n
    yδ .= y[i,:] .- y_bar[yb[i], yt[i],:]
    Γϵ .+= yδ * yδ'
  end
  Eb = eigenVJ(b); Et = eigenVJ(t);
  @tensor yE[b2,t2,d] := y_bar[b,t,d] * Eb[b,b2] * Et[t,t2]

end

function TF_RE_MANOVA_Data_incomplete(y::Array{Float64,2}, yb::Array{Int64,1}, yt::Array{Int64,1}, n::Int, N::Array{Int,2}, b = maximum(yp), t = maximum(yo), p = size(y,2), Σμ = eye(p), Γβ = eye(p), Γτ = eye(p), Γθ = eye(p), Γϵ = eye(p), νβ::Real = p+1.0, ντ::Real = p+1.0, νθ::Real = p+1.0, νϵ::Real = p+1.0)
  bt = b*t
  zero_indices = find(N .== 0)
  nonzero_indices = find(N .!= 0)
  index_translator = similar(N)
  for i ∈ eachindex(nonzero_indices)
    index_translator[nonzero_indices[i]] = i
  end
  unique_n = union(N)
  y_bar = zeros(length(nonzero_indices), p)
  for i ∈ 1:n
    y_bar[index_translator[yb[i], yt[i]],:] .+= y[i,:]
  end
  for i ∈ eachindex(nonzero_indices)
    y_bar[i] ./= N[nonzero_indices[i]]
  end
  yδ = Array{Float64}(p)
  for i ∈ 1:n
    yδ .= y[i,:] .- y_bar[index_translator[yb[i], yt[i]],:]
    Γϵ .+= yδ * yδ'
  end
  E = kron(eigenVJ(b), eigenVJ(t))
  E11 = E[nonzero_indices, nonzero_indices]
  E12 = E[nonzero_indices, zero_indices]
  E21 = E[zero_indices, nonzero_indices]
  E22 = E[zero_indices, zero_indices]
  E11_i = inv(E11)

  Ei_y = E11_i * y_bar
  E11_iE12 = E11_i * E12


end


struct TF_RE_MANOVA{p,T} <: parameters{T,1}
  x::Vector{T}
  Σβ::CovarianceMatrix{p,T}
  Στ::CovarianceMatrix{p,T}
  Σθ::CovarianceMatrix{p,T}
  Σϵ::CovarianceMatrix{p,T}

end

function ConstrainedParameters.update_Σ!(Θ::TF_RE_MANOVA{p,<:Real}) where {p}
  update_Σ!(Θ.Σβ)
  update_Σ!(Θ.Στ)
  update_Σ!(Θ.Σθ)
  update_Σ!(Θ.Σϵ)
end


struct TF_RE_MANOVA_Data_balanced <: ANOVA_Data_complete#p
  b::Int
  t::Int
  bm1::Int
  tm1::Int
  bt::Int
  btmbmtp1::Int
  N::Int
  Nm1bt::Int
  p::Int
  Λcomp::Vector{Array{Float64,2}}
  Σμ::Array{Float64,2}
  btΣμ::Array{Float64,2}
  Γβ::Array{Float64,2}
  Γτ::Array{Float64,2}
  Γθ::Array{Float64,2}
  Γϵ::Array{Float64,2}
  yE1::Array{Float64,2}
  yE2::Array{Float64,2}
  yE3::Array{Float64,2}
  yE4::Array{Float64,2}
  νβp1::Float64
  ντp1::Float64
  νθp1::Float64
  νϵp1::Float64
end
function TF_RE_MANOVA_Data_balanced(y::Array{Float64,2}, yb::Array{Int64,1}, yt::Array{Int64,1}, n::Int, N::Int, b = maximum(yp), t = maximum(yo), p = size(y,2), Σμ = eye(p), Γβ = eye(p), Γτ = eye(p), Γθ = eye(p), Γϵ = eye(p), νβ::Real = p+1.0, ντ::Real = p+1.0, νθ::Real = p+1.0, νϵ::Real = p+1.0)
  bt = b*t
  y_bar = zeros(b, t, p)
  for i ∈ 1:n
    y_bar[yb[i], yt[i],:] .+= y[i,:]
  end
  y_bar ./= N
  yδ = Array{Float64}(p)
  for i ∈ 1:n
    yδ .= y[i,:] .- y_bar[yb[i], yt[i],:]
    for j ∈ 1:p, k ∈ 1:p#faster than yδ * yδ' ???
      Γϵ[k,j] += yδ[j] * yδ[k]
    end
  end
  Eb = eigenVJ(b); Et = eigenVJ(t);
  @tensor yE[b2,t2,d] := y_bar[b,t,d] * Eb[b,b2] * Et[t,t2]
  yE1 = Array{Float64,2}(p,p)
  for i ∈ 1:p, j ∈ 1:p
    yE1[j,i] = yE[1,1,i] * yE[1,1,j]
  end
  yE2 = zeros(p,p)
  for k ∈ 2:t, i ∈ 1:p, j ∈ 1:p
    yE2[j,i] += yE[1,k,i] * yE[1,k,j]
  end
  yE3 = zeros(p,p)
  for k ∈ 2:b, i ∈ 1:p, j ∈ 1:p
    yE3[j,i] += yE[k,1,i] * yE[k,1,j]
  end
  yE4 = zeros(p,p)
  for k ∈ 2:b, l ∈ 2:t, i ∈ 1:p, j ∈ 1:p
    yE4[j,i] += yE[k,l,i] * yE[k,l,j]
  end
  TF_RE_MANOVA_Data_balanced(b, t, b-1, t-1, bt, bt-b-t+1, N, bt*(N-1), p, [Array{Float64,2}(p,p) for i ∈ 1:4], Σμ, bt .* Σμ, Γβ, Γτ, Γθ, Γϵ, yE1, yE2, yE3, yE4, νβ+1, ντ+1, νθ+1, νϵ+1)
end

function outer_prod(x::Vector{T}, y::Vector{T}) where {T}
  out = Array{T}(length(x), length(y))
  for i ∈ eachindex(x), j ∈ eachindex(y)
    out[j,i] = x[i] * y[j]
  end
  out
end
function outer_prod(x::Vector{T}) where {T}
  p = length(x)
  out = Array{T}(p,p)
  for i ∈ 1:p
    out[i,i] = x[i]^2
    for j ∈ 1:i-1
      out[j,i] = x[i] * x[j]
    end
  end
  out
end

function compΛ!(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_balanced) where {p}
  data.Λcomp[4] .= Θ.Σθ.Σ .+ Θ.Σϵ.Σ ./ data.N
  data.Λcomp[3] .= data.t .* Θ.Σβ.Σ .+ data.Λcomp[4]
  data.Λcomp[2] .= data.b .* Θ.Στ.Σ .+ data.Λcomp[4]
  data.Λcomp[1] .= data.b .* Θ.Στ.Σ .+ data.Λcomp[3] .+ data.btΣμ
end
function compΛ(Θ::TF_RE_MANOVA{p,T}, data::TF_RE_MANOVA_Data_balanced) where {p,T}
  Λcomp = Vector{Array{T,2}}(4)
  Λcomp[4] = Θ.Σθ.Σ .+ Θ.Σϵ.Σ ./ data.N
  Λcomp[3] = data.t .* Θ.Σβ.Σ .+ Λcomp[4]
  Λcomp[2] = data.b .* Θ.Στ.Σ .+ Λcomp[4]
  Λcomp[1] = data.b .* Θ.Στ.Σ .+ Λcomp[3] .+ data.btΣμ
  Λcomp
end
function compQuad(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_balanced) where {p}
  htraceSymProd( data.yE1, data.Λcomp[1] ) + htraceSymProd( data.yE2, data.Λcomp[2] ) + htraceSymProd( data.yE3, data.Λcomp[3] ) + htraceSymProd( data.yE4, data.Λcomp[4] )
end
function compQuad(Θ::TF_RE_MANOVA, data::TF_RE_MANOVA_Data_balanced, Λcomp::Vector{Array{T,2}}) where {T<:Real}
  htraceSymProd( data.yE1, Λcomp[1] ) + htraceSymProd( data.yE2, Λcomp[2] ) + htraceSymProd( data.yE3, Λcomp[3] ) + htraceSymProd( data.yE4, Λcomp[4] )
end
#2out - b*t*p*log(n) gives the actual log determinant.
function compLogDet(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_balanced) where {p}
  logTriangleDet(data.Λcomp[1]) + data.tm1*logTriangleDet(data.Λcomp[2]) + data.bm1*logTriangleDet(data.Λcomp[3]) + data.btmbmtp1*logTriangleDet(data.Λcomp[4]) - data.Nm1bt*hlogdet(Θ.Σϵ)
end
function compLogDet(Θ::TF_RE_MANOVA{p,<:Real}, data::TF_RE_MANOVA_Data_balanced, Λcomp::Vector{Array{T,2}}) where {p,T<:Real}
  logTriangleDet(Λcomp[1]) + data.tm1*logTriangleDet(Λcomp[2]) + data.bm1*logTriangleDet(Λcomp[3]) + data.btmbmtp1*logTriangleDet(Λcomp[4]) - data.Nm1bt*hlogdet(Θ.Σϵ)
end
function compPrior(Θ::TF_RE_MANOVA, data::TF_RE_MANOVA_Data_balanced)
  data.νβp1*nhlogdet(Θ.Σβ) + data.ντp1*nhlogdet(Θ.Στ) + data.νθp1*nhlogdet(Θ.Σθ) + data.νϵp1*nhlogdet(Θ.Σϵ) - htrace_AΣinv(data.Γβ, Θ.Σβ) - htrace_AΣinv(data.Γτ, Θ.Στ) - htrace_AΣinv(data.Γθ, Θ.Σθ) - htrace_AΣinv(data.Γϵ, Θ.Σϵ)
end
function log_density(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_balanced) where {p}
  update_Σ!(Θ)
  compΛ!(Θ, data)
  for i ∈ eachindex(data.Λcomp)
    chol!(data.Λcomp[i])
    inv!(data.Λcomp[i])
  end
  out = compPrior(Θ, data) + compLogDet(Θ, data)
  for i ∈ eachindex(data.Λcomp)
    UUt!(data.Λcomp[i])
  end
  out - compQuad(Θ, data)
end
function log_density(Θ::TF_RE_MANOVA{p,T}, data::TF_RE_MANOVA_Data_balanced) where { p, T <: Real }
  update_Σ!(Θ)
  Λcomp = compΛ(Θ, data)
  for i ∈ eachindex(Λcomp)
    chol!(Λcomp[i])
    inv!(Λcomp[i])
  end
  out = compPrior(Θ, data) + compLogDet(Θ, data, Λcomp)
  for i ∈ eachindex(Λcomp)
    UUt!(Λcomp[i])
  end
  out - compQuad(Θ, data, Λcomp)
end


function compΛ!(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_unbalanced) where {p}
end

function log_density(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_unbalanced) where {p}
end

function log_density(Θ::TF_RE_MANOVA{p, T}, data::TF_RE_MANOVA_Data_unbalanced) where {p, T <: Real }
end



function compΛ!(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_incomplete) where {p}
end

function log_density(Θ::TF_RE_MANOVA{p,Float64}, data::TF_RE_MANOVA_Data_incomplete) where {p}
end

function log_density(Θ::TF_RE_MANOVA{p,T}, data::TF_RE_MANOVA_Data_incomplete) where {p, T <: Real }
end

#calc_p(l::Int) = round(Int, (√(2l + 1) - 1)/2 )

function negative_log_density(Θ::Vector{T}, ::Type{P}, data::Data) where {T, P<:TF_RE_MANOVA}
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - log_density(param, data)
end
function negative_log_density!(Θ::TF_RE_MANOVA, data::Data)
  nld = -log_jacobian!(Θ)
  nld - log_density(Θ, data)
end

function Model(::Type{TF_RE_MANOVA}, p::Int; l = 6, q::DataType = SparseQuadratureGrids.GenzKeister, seq::Vector{Int} = SparseQuadratureGrids.default(q))
  Θ = construct(TF_RE_MANOVA{p, Float64})
  Grid = GridContainer(length(Θ), l, q, seq)
  Model(Grid, Θ, TF_RE_MANOVA{p})
end
