abstract type ANOVA_Data <: Data end
abstract type ANOVA_Data_complete <: Data end
abstract type ANOVA_Data_incomplete <: Data end


struct TF_RE_ANOVA_Data_balanced <: ANOVA_Data_complete
  N::Int64
  P::Int64
  O::Int64
  Pm1::Int64
  Om1::Int64
  PO::Int64
  R::Int
  Rm1::Int
  μ_hat::Array{Float64,2}
  δ::Array{Float64,2}
  s2::Float64
  Pδsum::Vector{Float64}
  Λcomp::Vector{Float64}
  Λcomp_i::Vector{Float64}
  cauchy_spread2::Float64
end

function TF_RE_ANOVA_Data(y::Array{Float64,1}, yp::Array{Int64,1}, yo::Array{Int64,1}, cauchy_spread = 20.0)
  N = size(y,1)
  P = maximum(yp)
  O = maximum(yo)
  PO = P*O

  Σy = zeros(Float64, P, O)
  Σy2 = zeros(Float64, P, O)
  Ra = zeros(Int64, P, O)
  for i ∈ 1:N
    Ra[yp[i], yo[i]] += 1
    Σy[yp[i], yo[i]] += y[i]
    Σy2[yp[i], yo[i]] += y[i]^2
  end
  Requal = all(x -> x == Ra[1], Ra)
  Σyb = Σy ./ Ra
  μ_hat = mean(Σyb, [1,2])
  δ = Σyb .- μ_hat
  s2 = sum( Σy2 - Ra .* Σyb .^2 )

  Pδ = kron(eigenVJ(P), eigenVJ(O))' * vec(δ')
  Pδsum = zeros(4)
  Pδsum[1] = Pδ[1]^2
  for i ∈ 2:O
    Pδsum[2] += Pδ[i]^2
  end
  for i ∈ O+1:O:PO
    Pδsum[3] += Pδ[i]^2
    for j ∈ i+(1:O-1)
      Pδsum[4] += Pδ[j]^2
    end
  end

  if Requal
    TF_RE_ANOVA_Data_balanced( N, P, O, P-1, O-1, PO, Ra[1], Ra[1] - 1, μ_hat, δ, s2, Pδsum, Array{Float64}(4), Array{Float64}(4), cauchy_spread^2)
  else
    throw("Unbalanced data not yet implemented.")
  end

end

struct TF_RE_ANOVA{T} <: parameters{T,1}
  x::Vector{T}
  σ::PositiveVector{4,T}

end



function compΛ!(Θ::TF_RE_ANOVA{Float64}, data::TF_RE_ANOVA_Data_balanced)
  data.Λcomp[4] = Θ.σ.x[4] / data.R + Θ.σ.x[3]
  data.Λcomp[3] = data.Λcomp[4] + data.O * Θ.σ.x[1]
  data.Λcomp[2] = data.Λcomp[4] + data.P * Θ.σ.x[2]
  data.Λcomp[1] = data.Λcomp[3] + data.P * Θ.σ.x[2]
  data.Λcomp_i .= 1 ./ data.Λcomp
end

function log_density(Θ::TF_RE_ANOVA{Float64}, data::TF_RE_ANOVA_Data_balanced)
  compΛ!(Θ, data)

  - ( data.s2 / Θ.σ.x[4] + dot(data.Pδsum, data.Λcomp_i) + log(data.Λcomp[1]) + data.Om1*log(data.Λcomp[2]) + data.Pm1*(log(data.Λcomp[3]) + data.Om1*log(data.Λcomp[4])) + data.Rm1*data.PO*log(Θ.σ.x[4]) + sum(log, Θ.σ.x) ) / 2 - log( 1 + Θ.σ.x[2] / data.cauchy_spread2 )

end

function log_density{ T <: Real }(Θ::TF_RE_ANOVA{T}, data::TF_RE_ANOVA_Data_balanced)
  out = data.s2 / Θ.σ.x[4]
  Λcomp4 = Θ.σ.x[4] / data.R + Θ.σ.x[3]
  Λcomp3 = Λcomp4 + data.O * Θ.σ.x[1]
  Λcomp2 = Λcomp4 + data.P * Θ.σ.x[2]
  Λcomp1 = Λcomp3 + data.P * Θ.σ.x[2]

  out += data.Pδsum[1] / Λcomp1 + data.Pδsum[2] / Λcomp2 + data.Pδsum[3] / Λcomp3 + data.Pδsum[4] / Λcomp4

  - ( out + log(Λcomp1) + data.Om1*log(Λcomp2) + data.Pm1*(log(Λcomp3) + data.Om1*log(Λcomp4)) + data.Rm1*data.PO*log(Θ.σ.x[4]) + sum(log, Θ.σ.x) ) / 2  - log( 1 + Θ.σ.x[2] / data.cauchy_spread2 )

end

function negative_log_density{T}(Θ::Vector{T}, ::Type{TF_RE_ANOVA}, data::Data)
  param = construct(TF_RE_ANOVA{T}, Θ)
  nld = -log_jacobian!(param)
  nld - log_density(param, data)
end
function negative_log_density!(Θ::TF_RE_ANOVA, data::Data)
  nld = -log_jacobian!(Θ)
  nld - log_density(Θ, data)
end
